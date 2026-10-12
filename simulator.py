"""Synthetic football match simulator -- the data foundation of Touchline.

WHAT THIS IS
------------
A generative model of a football match. It produces a stream of observable
events (passes, shots, tackles, pressure, possession changes) *and*, separately,
the hidden latent state that caused them.

WHY SIMULATE AT ALL
-------------------
Two reasons, one legal and one scientific.

  * Legal: the hackathon requires synthetic, football-realistic data. Event-level
    football data is commercially locked up, and real club/player names bring
    trademark exposure we do not need.

  * Scientific: a simulator has *ground truth*. It knows which team was genuinely
    controlling the match at minute 63. Real data does not -- analysts only argue
    about it. That means we can score our explanations against a known answer,
    which is the whole differentiator of this project.

THE ONE RULE THAT MATTERS MOST
------------------------------
The latent state must CAUSE the observables, and must never BE one of them.

If control had no effect on anything visible, no model could ever recover it --
that is not a modelling failure, it is an impossibility, and it is worth
internalising: a quantity is learnable only if it leaves a trace.

Conversely, if we left `control_true` sitting in the event objects, every model
downstream would "predict" a value it was handed. That is data leakage, and it
is how machine-learning projects produce beautiful numbers that mean nothing.
So the latent state travels in a separate channel -- see `Snapshot` -- and only
ever reaches a model during training and scoring.

CAUSAL CHAIN WORTH NOTICING
---------------------------
    control -> pass completion -> sequence length -> final-third entries -> shots

Control does not touch shots directly. It propagates through the game. That is
also, incidentally, what makes the explanations non-trivial.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, NamedTuple

from schemas import (
    CardType,
    EventType,
    LatentState,
    MatchEvent,
    Outcome,
    PassType,
    Phase,
    StoppageBreakdown,
    StoppageCause,
)

# --------------------------------------------------------------------------- #
# Pitch geometry
# --------------------------------------------------------------------------- #
# Internally we work in a single "normalized" 0-100 frame, then convert to
# metres when we need a physical quantity like pass distance. Mixing the two
# silently is a classic source of bugs that are extremely hard to see, so the
# conversion lives in exactly one function (`_metres`).

PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0
REGULATION_S = 5400  # 90 minutes
HALF_S = REGULATION_S // 2  # 45 minutes
GOAL_CENTER_Y_M = PITCH_WID_M / 2


# --------------------------------------------------------------------------- #
# Profiles -- the "personality" of a synthetic team and player
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TeamProfile:
    """Synthetic team strengths. All values are deliberately fictional."""

    team_id: str
    name: str
    attack: float  # multiplier, ~0.7 poor to ~1.3 strong
    defence: float  # multiplier, higher = harder to score against
    tempo: float  # ball circulation speed
    press: float  # 0..1, how hard they press


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    name: str
    team_id: str
    position: str  # GK | DEF | MID | FWD
    number: int  # shirt number -- visible to a broadcast camera
    skill: float  # passing / technique, 0..1
    finishing: float  # 0..1


# --------------------------------------------------------------------------- #
# A Snapshot pairs the observable with the truth
# --------------------------------------------------------------------------- #


class Snapshot(NamedTuple):
    """One instant of the match: what you could see, and what was really going on.

    Keeping these in one object (rather than two parallel lists that can drift
    out of sync) means an event and its ground truth can never be mismatched.

    Downstream consumers receive *only* `.event`. The `.latent` half is opened
    by exactly two pieces of code: the trainer and the scorer.
    """

    event: MatchEvent
    latent: LatentState


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _lerp_table(x: float, table: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation over a small table of breakpoints.

    We use an explicit table rather than a fitted curve because it is
    *inspectable*. When the validation report says "our pass completion is 3
    points too high at long range", you can see exactly which number to change.
    Opaque is easy to write and impossible to maintain.
    """
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return table[-1][1]


# Mean completion probability by pass distance in metres. These breakpoints are
# the single most important tuning surface in the simulator; the validation
# report compares the resulting aggregate against published reference ranges.
_BASE_COMPLETION: list[tuple[float, float]] = [
    (0.0, 0.970),
    (5.0, 0.965),
    (15.0, 0.925),
    (25.0, 0.865),
    (40.0, 0.720),
    (60.0, 0.550),
    (80.0, 0.400),
]

# --------------------------------------------------------------------------- #
# Tuning constants -- deliberately drawn out to module level
# --------------------------------------------------------------------------- #
# Emphasising these as named constants rather than magic numbers buried in the
# loop is not cosmetic. Calibration is an iterative loop of "change one thing,
# re-measure", and you cannot reason about that loop if the knobs are scattered
# through the code. Every constant here is a hypothesis the harness tests.

# Probability that a completed pass is immediately followed by a sequence-ending
# event (foul, throw-in, clearance). Without this, sequences run until a pass is
# misplaced, which at 85% completion almost never happens.
P_SEQUENCE_BREAKUP = 0.175

# Shot probability once the ball is in the attacking third, scaled by how close
# to goal we are: p = BASE + SCALE * ((lx - 68) / 32) ** 2
SHOT_PROB_BASE = 0.016
SHOT_PROB_SCALE = 0.125

# How long the clock runs between events, in seconds. These are the numbers that
# actually control event density: total events = 5400s / mean seconds per event.
# Football's clock keeps running, so "time until the next event" is the right
# abstraction -- a foul costs 15-25s of real time, an open-play turnover costs 2.
CLOCK_PASS_S = (1, 4)
CLOCK_TURNOVER_OPEN_S = (1, 3)
CLOCK_TURNOVER_THROW_S = (8, 16)
CLOCK_TURNOVER_FOUL_S = (15, 25)
CLOCK_SHOT_GOAL_S = (30, 50)
CLOCK_SHOT_SAVED_S = (6, 12)
CLOCK_SHOT_OTHER_S = (8, 15)
CLOCK_CORNER_S = (25, 40)
CLOCK_OFFSIDE_S = (10, 20)
CLOCK_SUB_S = (8, 15)

# --- sequence-ending composition ------------------------------------------- #
# Proportion of sequence breakups that are fouls. A foul does NOT change
# possession (the fouled side takes a free kick), so it is purely a time cost.
P_BREAKUP_FOUL = 0.15
# In the attacking third, a share of out-of-play events are deflected behind for
# a corner, which the attacking side restarts.
P_BREAKUP_ATTACK_THIRD_CORNER = 0.11
# Fraction of BLOCKED shots that deflect behind for a corner.
P_CORNER_FROM_BLOCK = 0.30
# Offside risk on a progressive pass into the final third.
P_OFFSIDE = 0.018

# --- discipline ------------------------------------------------------------- #
P_YELLOW_CARD = 0.16  # per foul
P_RED_CARD = 0.012  # per foul

# --- substitutions ---------------------------------------------------------- #
MAX_SUBS_PER_TEAM = 5
SUB_WINDOWS_MIN = (46, 55, 62, 70, 78, 85)
P_SUB_PER_WINDOW = 0.45  # per team, per window

# --- added (stoppage) time -------------------------------------------------- #
# Seconds of added time each cause contributes. These are referee rules of thumb,
# deliberately NOT the play-clock cost of the event: a substitution burns ~10s of
# playing clock but delays the game by nearer half a minute, and it is the delay
# that gets added on. Keeping the two notions apart is what lets the ledger
# actually explain the added-time figure instead of just restating it.
STOPPAGE_LOST_S: dict[StoppageCause, int] = {
    StoppageCause.GOAL: 40,
    StoppageCause.SUBSTITUTION: 30,
    StoppageCause.CARD: 30,
}
BASE_STOPPAGE_S = 60  # baseline allowance for ordinary lost time
STOPPAGE_CLAMP_S = (60, 600)  # a half adds between 1:00 and 10:00

# --- contested possession --------------------------------------------------- #
# A turnover does not always hand the ball cleanly to a team: sometimes it
# becomes a loose ball both sides fight for. The probability rises with latent
# chaos, which is what makes contested time *learnable* from the observables
# rather than a decorative counter that increments for no reason.
P_CONTEST_BASE = 0.05
P_CONTEST_CHAOS = 0.18
CLOCK_LOOSE_BALL_S = (3, 9)

# Shot-quality (xG-like) model: quality = XG_BASE * exp(-XG_DISTANCE_DECAY * d)
# The shape was already correct on the first draft -- `xg_per_shot` measured
# 0.10 against a 0.08-0.14 target before any tuning. Only the scale needed a
# nudge to lift goals off the bottom of their range.
XG_BASE = 0.65
XG_DISTANCE_DECAY = 0.14

# --- penalty-area geometry -------------------------------------------------- #
# Normalised pitch coordinates. A shot is "inside the box" when its origin lies
# in the penalty area of the goal being attacked. The box is a real, visible
# region of the pitch, so `in_box` rides on the SHOT event -- a broadcaster can
# plainly see whether a strike came from inside the area.
BOX_DEPTH_M = 16.5   # goal-line to the edge of the box
BOX_WIDTH_M = 40.32  # width of the penalty area
BOX_X_MIN = 100.0 - 100.0 * BOX_DEPTH_M / PITCH_LEN_M              # ~84.3
BOX_Y_MIN = (PITCH_WID_M - BOX_WIDTH_M) / 2.0 / PITCH_WID_M * 100.0  # ~20.4
BOX_Y_MAX = 100.0 - BOX_Y_MIN

# --- shot outcomes ---------------------------------------------------------- #
# This share of shots hit the frame and miss (post or crossbar). A deliberately
# narrow band: roughly one woodwork strike per ~29-shot match.
P_SHOT_WOODWORK = 0.03

# --- big chances (latent) --------------------------------------------------- #
# A "big chance" is an analyst-coded, subjective stat in real football -- there
# is no label for it in Opta or Statsbomb, only a human's judgement. Here the
# simulator produces the label, because it *generated* the chance. The tier is
# decided on a LATENT quality that carries a little noise the observable xG does
# not, so a downstream classifier recovering the tier from events faces a real
# learning task instead of reading the answer off the shot_quality field.
BIG_CHANCE_QUALITY = 0.35
BIG_CHANCE_NOISE = 0.04  # uniform half-width; bounded, so invariants stay exact

# --- unforced errors (latent) ---------------------------------------------- #
# An "error leading to a chance" is the other analyst-coded stat we get to
# label. We model it as a failed pass deep in the possessing side's own third
# that hands the ball straight to the opposition with a gift chance. The
# probability scales with latent chaos -- that causal link is the trace that
# makes errors recoverable from the event stream, exactly like control and
# pass intent.
DEFENSIVE_THIRD_X = 32.0  # attacking-local: own defensive third is lx < 32
P_ERROR_BASE = 0.004
P_ERROR_CHAOS = 0.05
ERROR_CHANCE_BONUS_XG = 0.12  # the give-away gifts a better chance than the position



def in_box(lx: float, ly: float) -> bool:
    """True if an attacking-local point (lx, ly) is inside the penalty area."""
    return lx >= BOX_X_MIN and BOX_Y_MIN <= ly <= BOX_Y_MAX

# Fictional name pools. Deterministic selection means a given seed always
# produces the same league -- reproducibility is not a nicety here, it is what
# lets us compare two runs of an experiment and trust the difference.
_FIRST_NAMES = [
    "Ade", "Bruno", "Cato", "Dax", "Emil", "Farid", "Goran", "Hugo",
    "Ilias", "Jonas", "Kian", "Lars", "Mateo", "Niko", "Omar", "Piet",
    "Quinn", "Rui", "Sami", "Tomas", "Umar", "Viktor", "Wim", "Yusuf",
]
_LAST_NAMES = [
    "Afolabi", "Berger", "Costa", "Dahl", "Eriksen", "Ferrer", "Gomis",
    "Haas", "Ivanov", "Jansen", "Kovac", "Lindqvist", "Moreno", "Novak",
    "Olsen", "Petrov", "Quaresma", "Roux", "Silva", "Tavares", "Ueda",
    "Varga", "Weiss", "Zeman",
]


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #


class MatchSimulator:
    """Generates one synthetic match, deterministically, from a seed."""

    def __init__(
        self,
        home: TeamProfile,
        away: TeamProfile,
        match_id: str = "M0001",
        seed: int = 0,
    ) -> None:
        # A dedicated RNG instance rather than the global `random` module.
        # Using the global RNG means any other code that draws a random number
        # changes your results -- a genuinely nasty class of non-reproducibility.
        self.rng = random.Random(seed)

        self.home = home
        self.away = away
        self.match_id = match_id

        self.squads: dict[str, list[PlayerProfile]] = {
            home.team_id: self._build_squad(home),
            away.team_id: self._build_squad(away),
        }

        # --- evolving match state --- #
        self.clock_s = 0
        self.score = {home.team_id: 0, away.team_id: 0}
        self.shots = {home.team_id: 0, away.team_id: 0}
        self.shots_on_target = {home.team_id: 0, away.team_id: 0}
        self.shot_quality = {home.team_id: 0.0, away.team_id: 0.0}

        # Discipline, set-piece and rotation counters. Kept as explicit maps
        # rather than derived after the fact, so the calibration harness can
        # read them directly instead of re-scanning the event list.
        self.corners = {home.team_id: 0, away.team_id: 0}
        self.offsides = {home.team_id: 0, away.team_id: 0}
        self.fouls = {home.team_id: 0, away.team_id: 0}
        self.yellows = {home.team_id: 0, away.team_id: 0}
        self.reds = {home.team_id: 0, away.team_id: 0}
        self.substitutions = {home.team_id: 0, away.team_id: 0}
        self._send_off = {home.team_id: False, away.team_id: False}
        self._sub_window_idx = 0
        self.contests = 0  # loose-ball phases

        # Phase 0.6 objective stats (the cheap trio).
        self.shots_in_box = {home.team_id: 0, away.team_id: 0}
        self.woodwork = {home.team_id: 0, away.team_id: 0}
        self.saves = {home.team_id: 0, away.team_id: 0}
        self.blocks = {home.team_id: 0, away.team_id: 0}

        # --- clock and period --- #
        # `clock_s` is monotonic, so the display clock has to be derived. These
        # track where the current period began in wall-clock terms.
        self.period = 1
        self._half_start_s = 0

        # --- possession as TIME, not pass count --- #
        # Between events, exactly one of three things is true: home controls,
        # away controls, or the ball is loose. We attribute every second to one
        # of the three. This is what lets `possession` mean time-in-control
        # rather than "who passed most", which is a different and worse thing.
        self.controlled_s = {home.team_id: 0.0, away.team_id: 0.0}
        self.contested_s = 0.0
        self._phase_owner: str | None = home.team_id
        self._phase_start_s = 0

        # --- added time --- #
        self._period_ledger: dict[int, Counter[StoppageCause]] = {
            1: Counter(),
            2: Counter(),
        }
        self.added_time_s = {1: 0, 2: 0}
        self.stoppage_breakdown: dict[int, StoppageBreakdown] = {}

        # Latent state. Initialised neutral; it wanders from here.
        self.control = 0.0  # +1 home dominant, -1 away dominant
        self.chaos = 0.30

        self.snapshots: list[Snapshot] = []
        self._event_seq = 0

        # --- sequence tracking for latent attributions ---------------------- #
        # Index of the possessor's most recent completed pass, so a big-chance
        # shot can credit "chance created" to the pass that manufactured it.
        # Cleared whenever possession changes hands or a sequence resets.
        self._last_passer_idx: int | None = None
        # True while an unforced error has gifted the current possessor a chance.
        # Consumed (or expired) on the next shot -- see _shoot.
        self._error_chance_pending = False

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _build_squad(self, team: TeamProfile) -> list[PlayerProfile]:
        """Build a 4-3-3 from the frozen name pools."""
        layout = [("GK", 1), ("DEF", 4), ("MID", 3), ("FWD", 3)]
        # Classic 4-3-3 shirt numbers, parallel to `layout`. A viewer reads the
        # number off the back, and the overlay draws it on the actor disc -- so
        # this is observable, not a rendering convenience.
        numbers = {"GK": [1], "DEF": [2, 3, 4, 5], "MID": [6, 8, 10], "FWD": [7, 9, 11]}
        squad: list[PlayerProfile] = []
        for position, count in layout:
            for i in range(count):
                first = self.rng.choice(_FIRST_NAMES)
                last = self.rng.choice(_LAST_NAMES)
                # Attribute ranges differ by position, which is what makes the
                # event stream positionally plausible.
                if position == "FWD":
                    skill = self.rng.uniform(0.45, 0.80)
                    finishing = self.rng.uniform(0.45, 0.90)
                elif position == "MID":
                    skill = self.rng.uniform(0.50, 0.90)
                    finishing = self.rng.uniform(0.30, 0.65)
                elif position == "DEF":
                    skill = self.rng.uniform(0.45, 0.80)
                    finishing = self.rng.uniform(0.10, 0.35)
                else:  # GK
                    skill = self.rng.uniform(0.40, 0.70)
                    finishing = self.rng.uniform(0.0, 0.10)

                # Scale by team quality so strong teams have better players.
                quality = (team.attack + team.defence) / 2.0
                skill = _clamp(skill * quality, 0.05, 0.99)

                squad.append(
                    PlayerProfile(
                        player_id=f"{team.team_id}-{position}-{i}",
                        name=f"{first} {last}",
                        team_id=team.team_id,
                        position=position,
                        number=numbers[position][i],
                        skill=round(skill, 3),
                        finishing=round(finishing, 3),
                    )
                )
        return squad

    # ------------------------------------------------------------------ #
    # Coordinate helpers
    # ------------------------------------------------------------------ #

    def _attack_dir(self, team_id: str) -> int:
        """+1 if this team attacks toward x=100, -1 otherwise."""
        return 1 if team_id == self.home.team_id else -1

    def _to_absolute(self, team_id: str, lx: float, ly: float) -> tuple[float, float]:
        """Attacking-local frame -> fixed broadcast frame.

        This flip is the reason a broadcast overlay can draw both teams in the
        same physical space, and forgetting it is the classic cause of a debug
        session where a team appears to attack its own goal.
        """
        if self._attack_dir(team_id) == 1:
            return lx, ly
        return 100.0 - lx, 100.0 - ly

    def _goal_distance_m(self, lx: float, ly: float) -> float:
        """Distance to the centre of the goal being attacked, in metres."""
        mx = lx / 100.0 * PITCH_LEN_M
        my = ly / 100.0 * PITCH_WID_M
        return math.hypot(PITCH_LEN_M - mx, GOAL_CENTER_Y_M - my)

    # ------------------------------------------------------------------ #
    # Latent dynamics
    # ------------------------------------------------------------------ #

    def _update_latent(self) -> None:
        """Advance the hidden state: mean-reverting drift plus noise.

        This is an Ornstein-Uhlenbeck-style process: control wanders, but is
        continuously pulled back toward zero. Pure random walks drift to the
        rails and produce nonsense matches.
        """
        self.control = _clamp(0.985 * self.control + self.rng.gauss(0, 0.05), -1.0, 1.0)

        # Chaos is highest when the game is even. A team in full control makes
        # the game *less* chaotic, not more. Note this is deliberately
        # independent of `control` in sign -- it is a second axis, not a mirror
        # of the first, which is what makes "control vs chaos" a real 2D story.
        chaos_target = 0.25 + 0.55 * (1.0 - abs(self.control))
        self.chaos = _clamp(
            0.95 * self.chaos + 0.05 * chaos_target + self.rng.gauss(0, 0.02), 0.0, 1.0
        )

    def _shock_control(self, team_id: str, magnitude: float) -> None:
        """A discrete event that swings control hard -- a goal, a red card."""
        sign = 1.0 if team_id == self.home.team_id else -1.0
        self.control = _clamp(self.control + sign * magnitude, -1.0, 1.0)

    def _phase(self) -> Phase:
        if abs(self.control) > 0.45 and self.chaos < 0.45:
            return Phase.CONTROLLED
        if self.chaos > 0.6:
            return Phase.CHAOTIC
        return Phase.CONTESTED

    def _latent(
        self,
        pass_type_true: PassType | None = None,
        big_chance_true: bool | None = None,
        chance_created_true: bool | None = None,
        error_true: bool | None = None,
        from_error_true: bool | None = None,
    ) -> LatentState:
        dominant = (
            self.home.team_id if self.control >= 0 else self.away.team_id
        )
        return LatentState(
            match_id=self.match_id,
            clock_s=self.clock_s,
            control_true=round(self.control, 4),
            chaos_true=round(self.chaos, 4),
            dominant_team=dominant,
            phase=self._phase(),
            pass_type_true=pass_type_true,
            big_chance_true=big_chance_true,
            chance_created_true=chance_created_true,
            error_true=error_true,
            from_error_true=from_error_true,
        )

    # ------------------------------------------------------------------ #
    # Clock, possession-time and added-time bookkeeping
    # ------------------------------------------------------------------ #

    def _display_clock(self) -> tuple[int, int]:
        """Return (display_minute, stoppage_minute) for the current clock.

        The display clock is what a broadcaster shows: 0-45 in the first half,
        45-90 in the second, then 45+X / 90+X in added time. Deriving it from a
        monotonic `clock_s` keeps event ordering trivial and puts the fiddly
        base-45 arithmetic in exactly one place.
        """
        elapsed = self.clock_s - self._half_start_s
        regulation = min(elapsed, HALF_S)
        base = 0 if self.period == 1 else 45
        minute = base + int(regulation // 60)
        if elapsed > HALF_S:
            # The first added minute is the window [45:00, 46:00), so a clock
            # at exactly 45:00 is still regulation and must NOT be labelled
            # 45+1.
            stoppage = int((elapsed - HALF_S) // 60) + 1
        else:
            stoppage = 0
        return minute, stoppage

    def _set_phase(self, owner: str | None) -> None:
        """Attribute elapsed time to the phase that just ended.

        `owner=None` marks a contested (loose-ball) phase. Every second of the
        match lands in exactly one of three buckets, so the possession figures
        always sum to the elapsed time. That invariant matters: a decomposition
        that does not sum to the whole is worse than no decomposition at all,
        because it looks authoritative and is quietly wrong.
        """
        dt = self.clock_s - self._phase_start_s
        if dt > 0:
            if self._phase_owner is None:
                self.contested_s += dt
            else:
                self.controlled_s[self._phase_owner] += dt
        self._phase_owner = owner
        self._phase_start_s = self.clock_s

    def _note_stoppage(self, cause: StoppageCause) -> None:
        """Record one cause of added time for the half currently being played."""
        self._period_ledger[self.period][cause] += STOPPAGE_LOST_S[cause]

    def _added_time_breakdown(self, period: int) -> StoppageBreakdown:
        """Turn a half's stoppage ledger into an explainable added-time figure.

        This is the whole point of keeping a ledger: the added time is the sum of
        the causes, so the number and its justification are the same object.
        """
        by_cause = {
            cause: int(seconds) for cause, seconds in self._period_ledger[period].items()
        }
        by_cause[StoppageCause.ROUTINE] = (
            by_cause.get(StoppageCause.ROUTINE, 0) + BASE_STOPPAGE_S
        )

        raw = sum(by_cause.values())
        total = int(_clamp(raw, *STOPPAGE_CLAMP_S))
        if raw and total != raw:
            # Keep the ledger summing to the total we actually report, or the
            # explanation would contradict the number it explains.
            scale = total / raw
            by_cause = {c: int(round(v * scale)) for c, v in by_cause.items()}
            total = sum(by_cause.values())
        return StoppageBreakdown(total_s=total, by_cause=by_cause)

    # ------------------------------------------------------------------ #
    # Event emission
    # ------------------------------------------------------------------ #

    def _emit(
        self,
        *,
        event_type: EventType,
        team_id: str,
        player: PlayerProfile,
        outcome: Outcome,
        x: float,
        y: float,
        end_x: float | None = None,
        end_y: float | None = None,
        card_type: CardType | None = None,
        player_out_id: str | None = None,
        stoppage: StoppageBreakdown | None = None,
        pass_type_true: PassType | None = None,
        in_box: bool | None = None,
        big_chance_true: bool | None = None,
        chance_created_true: bool | None = None,
        error_true: bool | None = None,
        from_error_true: bool | None = None,
        **features: float | None,
    ) -> None:
        """Record one event together with the latent state that produced it."""
        self._event_seq += 1
        minute, stoppage_minute = self._display_clock()
        event = MatchEvent(
            event_id=f"{self.match_id}-{self._event_seq:05d}",
            match_id=self.match_id,
            clock_s=self.clock_s,
            minute=minute,
            stoppage_minute=stoppage_minute,
            period=self.period,
            type=event_type,
            team=team_id,
            player_id=player.player_id,
            player_name=player.name,
            player_number=player.number,
            outcome=outcome,
            card_type=card_type,
            player_out_id=player_out_id,
            stoppage=stoppage,
            x=round(x, 2),
            y=round(y, 2),
            end_x=None if end_x is None else round(end_x, 2),
            end_y=None if end_y is None else round(end_y, 2),
            in_box=in_box,
            **features,
        )
        self.snapshots.append(
            Snapshot(
                event=event,
                latent=self._latent(
                    pass_type_true=pass_type_true,
                    big_chance_true=big_chance_true,
                    chance_created_true=chance_created_true,
                    error_true=error_true,
                    from_error_true=from_error_true,
                ),
            )
        )

    # ------------------------------------------------------------------ #
    # Football logic
    # ------------------------------------------------------------------ #

    def _pick_player(self, team_id: str, weights: dict[str, float]) -> PlayerProfile:
        squad = self.squads[team_id]
        w = [weights.get(p.position, 0.1) for p in squad]
        return self.rng.choices(squad, weights=w, k=1)[0]

    def _pressure(self, team_id: str, lx: float) -> float:
        """Opponent pressure on the ball, 0..1.

        Higher in the attacking third, because that is where the opposition
        actually defends. The opponent's `press` rating scales the whole curve.
        """
        opponent = self.away if team_id == self.home.team_id else self.home
        base = 0.10 + 0.55 * (lx / 100.0) ** 1.6
        p = base * (0.6 + 0.8 * opponent.press) + self.rng.gauss(0, 0.06)
        return _clamp(p, 0.0, 0.95)

    def _completion_prob(
        self,
        team_id: str,
        player: PlayerProfile,
        distance_m: float,
        pressure: float,
        progressive: bool,
    ) -> float:
        """Probability this pass finds its man.

        Notice the causal link that gives the latent state something to leave a
        trace in: `control` raises the completion probability of the team that
        owns it, and lowers the opponent's. Without a link like this, control
        would be unlearnable and the whole project would be vacuous.
        """
        base = _lerp_table(distance_m, _BASE_COMPLETION)
        p = base
        p *= 1.0 - 0.28 * pressure  # pressure kills passes
        p *= 0.90 + 0.20 * player.skill  # better players pass better
        p *= 0.96 if progressive else 1.03  # risky passes complete less often
        signed_control = self.control if team_id == self.home.team_id else -self.control
        p *= 1.0 + 0.07 * signed_control  # the causal trace of "control"

        # A team with a player sent off is a man short: it keeps the ball less
        # well, and its opponent finds it easier to play through. Without this a
        # red card is a purely cosmetic event -- present in the stream, absent
        # from the football. An event the model ignores is a modelling bug, not
        # a simplification.
        opponent_id = self.away.team_id if team_id == self.home.team_id else self.home.team_id
        if self._send_off[team_id]:
            p *= 0.94
        if self._send_off[opponent_id]:
            p *= 1.03
        return _clamp(p, 0.03, 0.97)

    def _choose_pass(self, lx: float, ly: float) -> tuple[float, float, bool, PassType]:
        """Pick where a pass goes, in the attacking-local frame, and what KIND it is.

        Returns (new_lx, new_ly, progressive, pass_type). The kind is ground
        truth: it rides in LatentState, never in the event, because a real
        analyst can only *infer* intent from the trajectory. Phase 1 builds a
        classifier that recovers these labels from geometry alone and scores
        itself against them.

        Two hard lessons live in here, both visible as code:

        1. POSITION GATES INTENT. A 22-45 m "long" ball is only a long ball when
           it is played from deep. The same branch firing in the final third
           parks the ball on the goal line, and shot probability is quadratic in
           proximity to goal -- so that single oversight roughly doubled the
           scoreline. Hence the carefully nested branches: the final third is
           resolved first, and only a modest advance is ever legal there.

        2. A CROSS MUST AIM AT THE BOX. The first typed draft labelled wide
           final-third passes as crosses without changing where the ball went,
           which made the label a lie the classifier could never learn. The
           lateral pull toward the centre is what makes a cross geometrically
           different from a short pass -- and therefore learnable.
        """
        r = self.rng.random()
        wide = ly < 24.0 or ly > 76.0

        if lx > 72.0 and r < 0.30:
            # Final-third ball. From wide it is a cross into the box; from
            # central, a through ball into the area. Either way it travels only
            # a short distance -- you are already where the damage is done.
            advance = self.rng.uniform(4.0, 18.0)
            if wide:
                ptype = PassType.CROSS
                lateral = (50.0 - ly) * self.rng.uniform(0.30, 0.50)  # toward the box
            else:
                ptype = PassType.THROUGH_BALL
                lateral = self.rng.gauss(0.0, 12.0)
            progressive = True
        elif r < 0.18:
            # Long ball, typically played under pressure from deep.
            ptype = PassType.LONG
            advance = self.rng.uniform(22.0, 45.0)
            lateral = self.rng.gauss(0.0, 10.0)
            progressive = True
        elif r < 0.55:
            # Progressive build-up. A "through ball" in our taxonomy means a
            # pass that plays the ball INTO the final third. A forward pass that
            # stays in the middle of the park is a normal pass, not a through
            # ball -- labelling every progressive pass as one would have an
            # overlay reporting "~40% through balls", which any viewer would
            # reject, and rightly so.
            advance = self.rng.uniform(8.0, 22.0)
            lateral = self.rng.gauss(0.0, 9.0)
            progressive = True
            ptype = (
                PassType.THROUGH_BALL
                if lx + advance > 68.0
                else PassType.SHORT
            )
        else:
            # Safe circulation: sideways or backwards, which is why it completes.
            ptype = PassType.SHORT
            advance = self.rng.uniform(-10.0, 8.0)
            lateral = self.rng.gauss(0.0, 12.0)
            progressive = False

        return (
            _clamp(lx + advance, 3.0, 97.0),
            _clamp(ly + lateral, 4.0, 96.0),
            progressive,
            ptype,
        )

    def _shoot(self, team_id: str, lx: float, ly: float) -> bool:
        """Execute a shot and resolve its outcome.

        Returns True if the ATTACKING team keeps the ball (a corner won off a
        block), False if possession turns over. The caller must honour this:
        the first draft of the corner feature emitted the corner event and then
        let the main loop hand the ball to the opposition anyway. The stream
        looked fine -- every event type was present -- and the semantics were
        wrong. `test_events.py` is what caught it.

        Phase 0.6 additions, each justified by "feeds a claim OR is ground
        truth that scores one":
          * in_box is computed from the shot origin (observable -- you can see
            it on the screen) and feeds "kept them to long-range efforts".
          * SAVE and BLOCK get their own events so the keeper and the blocker
            are attributable, not anonymous aggregates.
          * big_chance_true lives in the latent channel: the tier is decided on
            a slightly noisy copy of latent quality, so recovering it from
            observables is a real learning task with a label.
          * if possession was gifted by an unforced error, the shot gets a
            quality bonus and from_error_true: the error's consequence is
            visible, the error itself is not.
        """
        team = self.home if team_id == self.home.team_id else self.away
        shooter = self._pick_player(
            team_id, {"FWD": 5.0, "MID": 2.0, "DEF": 0.4, "GK": 0.0}
        )

        dist_m = self._goal_distance_m(lx, ly)
        mx, my = lx / 100.0 * PITCH_LEN_M, ly / 100.0 * PITCH_WID_M

        # xG-like shot quality: decays with distance, punished for wide angles.
        angle_factor = math.exp(-(((my - GOAL_CENTER_Y_M) / 20.0) ** 2))
        quality = (
            XG_BASE * math.exp(-XG_DISTANCE_DECAY * dist_m) * angle_factor
            * (0.75 + 0.5 * shooter.finishing)
            * (0.9 + 0.2 * team.attack)
        )
        quality = _clamp(quality, 0.0, 0.95)

        # An unforced error gifts the opposition a better chance than the
        # position alone deserves. The bonus is applied to everything the shot
        # carries; the *gift* flag itself stays latent, because nothing on an
        # event stream ever says "this came from a mistake" -- a downstream
        # detector has to infer that from the giveaway it can see.
        from_error = self._error_chance_pending
        self._error_chance_pending = False
        if from_error:
            quality = _clamp(quality + ERROR_CHANCE_BONUS_XG, 0.0, 0.95)

        # Expected goals IS the probability of scoring -- that is the definition.
        # A "big chance" is decided on a latent copy of quality with bounded
        # noise, so the tier is correlated with, but not equal to, the xG any
        # observer could compute for themselves.
        latent_quality = _clamp(
            quality + self.rng.uniform(-BIG_CHANCE_NOISE, BIG_CHANCE_NOISE),
            0.0,
            1.0,
        )
        big_chance = latent_quality >= BIG_CHANCE_QUALITY

        roll = self.rng.random()
        if roll < quality:
            outcome = Outcome.GOAL
        elif roll < quality + 0.24:
            outcome = Outcome.SAVED
        elif roll < quality + 0.24 + P_SHOT_WOODWORK:
            outcome = Outcome.WOODWORK
        elif roll < quality + 0.54 + P_SHOT_WOODWORK:
            outcome = Outcome.OFF_TARGET
        else:
            outcome = Outcome.BLOCKED

        shot_speed = _clamp(
            self.rng.gauss(95.0, 18.0) * (0.85 + 0.3 * shooter.finishing),
            40.0,
            140.0,
        )

        self.shots[team_id] += 1
        self.shot_quality[team_id] += quality
        shot_in_box = in_box(lx, ly)
        if shot_in_box:
            self.shots_in_box[team_id] += 1
        if outcome in (Outcome.GOAL, Outcome.SAVED):
            self.shots_on_target[team_id] += 1
        elif outcome == Outcome.WOODWORK:
            # A strike on the frame is NOT a shot on target.
            self.woodwork[team_id] += 1

        sx, sy = self._to_absolute(team_id, lx, ly)
        ex, ey = self._to_absolute(team_id, 100.0, ly)

        self._emit(
            event_type=EventType.SHOT,
            team_id=team_id,
            player=shooter,
            outcome=outcome,
            x=sx,
            y=sy,
            end_x=ex,
            end_y=ey,
            in_box=shot_in_box,
            big_chance_true=big_chance,
            from_error_true=from_error,
            shot_quality=round(quality, 4),
            shot_speed_kmh=round(shot_speed, 1),
            ball_speed_kmh=round(shot_speed, 1),
            pressure_at_location=round(self._pressure(team_id, lx), 3),
        )

        # Credit the pass that manufactured the chance. Only the most recent
        # completed pass by this team counts -- it is the one that unlocked the
        # shot -- and it must have been played with no possession change since
        # (the index is cleared on every turnover/restart).
        if big_chance and self._last_passer_idx is not None:
            creator = self.snapshots[self._last_passer_idx]
            if creator.event.team == team_id and creator.event.type == EventType.PASS:
                self.snapshots[self._last_passer_idx] = Snapshot(
                    creator.event,
                    creator.latent.model_copy(update={"chance_created_true": True}),
                )
        self._last_passer_idx = None

        if outcome == Outcome.GOAL:
            self.score[team_id] += 1
            self._note_stoppage(StoppageCause.GOAL)
            # A goal is the hardest shock in football. It should visibly move
            # control, or the latent variable is not modelling the game.
            self._shock_control(team_id, 0.35)
        else:
            self._shock_control(team_id, 0.06)

        # Goals cost real time: celebration, restart, kick-off. Ignoring this
        # inflates the number of events the loop can fit into 90 minutes.
        if outcome == Outcome.GOAL:
            self.clock_s += self.rng.randint(*CLOCK_SHOT_GOAL_S)
        elif outcome == Outcome.SAVED:
            self.clock_s += self.rng.randint(*CLOCK_SHOT_SAVED_S)
        else:
            self.clock_s += self.rng.randint(*CLOCK_SHOT_OTHER_S)

        # The keeper and the blocker are named actors, so a save and a block get
        # their own observable events (and a count). Both belong to the team NOT
        # shooting, and both are emitted immediately after the shot they answer.
        defender_id = (
            self.away.team_id if team_id == self.home.team_id else self.home.team_id
        )
        if outcome == Outcome.SAVED:
            keeper = self._pick_player(defender_id, {"GK": 1.0})
            self.saves[defender_id] += 1
            self._emit(
                event_type=EventType.SAVE,
                team_id=defender_id,
                player=keeper,
                outcome=Outcome.COMPLETE,
                x=ex,
                y=ey,
            )
        elif outcome == Outcome.BLOCKED:
            blocker = self._pick_player(
                defender_id, {"DEF": 4.0, "MID": 2.0, "FWD": 0.5}
            )
            self.blocks[defender_id] += 1
            self._emit(
                event_type=EventType.BLOCK,
                team_id=defender_id,
                player=blocker,
                outcome=Outcome.COMPLETE,
                x=sx,
                y=sy,
            )

        # A blocked shot deflects behind for a corner reasonably often. A corner
        # is restarted by the ATTACKING side, so this retains possession -- which
        # is why the return value exists.
        if outcome == Outcome.BLOCKED and self.rng.random() < P_CORNER_FROM_BLOCK:
            self.clock_s += self._emit_corner(team_id, ex, ey)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Set pieces, discipline and rotation
    # ------------------------------------------------------------------ #

    def _emit_foul(self, attacking_id: str, abs_x: float, abs_y: float) -> int:
        """A foul by the defending side. Returns the clock cost in seconds.

        THE SEMANTICS MATTER, and the first draft got them exactly backwards:
          * The foul is committed by the team WITHOUT the ball.
          * The team WITH the ball keeps it, via a free kick.
        So a foul costs time but does NOT change possession. Getting this wrong
        makes every discipline statistic meaningless and silently corrupts the
        possession model -- which is precisely the kind of error that produces
        plausible-looking numbers that are subtly, systematically wrong.
        """
        defending_id = (
            self.away.team_id if attacking_id == self.home.team_id else self.home.team_id
        )
        offender = self._pick_player(defending_id, {"DEF": 4.0, "MID": 3.5, "FWD": 0.6})
        self.fouls[defending_id] += 1

        self._emit(
            event_type=EventType.FOUL,
            team_id=defending_id,
            player=offender,
            outcome=Outcome.LOST,
            x=abs_x,
            y=abs_y,
        )

        # Discipline. A red card is rare, but when it happens it must move the
        # match -- otherwise the latent model is ignoring a real event.
        roll = self.rng.random()
        if roll < P_RED_CARD:
            self.reds[defending_id] += 1
            self._send_off[defending_id] = True
            self._note_stoppage(StoppageCause.CARD)
            self._emit(
                event_type=EventType.CARD,
                team_id=defending_id,
                player=offender,
                outcome=Outcome.LOST,
                x=abs_x,
                y=abs_y,
                card_type=CardType.RED,
            )
            self._shock_control(attacking_id, 0.30)  # a man up swings control hard
        elif roll < P_RED_CARD + P_YELLOW_CARD:
            self.yellows[defending_id] += 1
            self._note_stoppage(StoppageCause.CARD)
            self._emit(
                event_type=EventType.CARD,
                team_id=defending_id,
                player=offender,
                outcome=Outcome.LOST,
                x=abs_x,
                y=abs_y,
                card_type=CardType.YELLOW,
            )

        return self.rng.randint(*CLOCK_TURNOVER_FOUL_S)

    def _emit_corner(self, team_id: str, abs_x: float, abs_y: float) -> int:
        """A corner: the attacking side restarts, so possession is retained."""
        self.corners[team_id] += 1
        taker = self._pick_player(team_id, {"MID": 4.0, "DEF": 2.0, "FWD": 2.0})
        self._emit(
            event_type=EventType.CORNER,
            team_id=team_id,
            player=taker,
            outcome=Outcome.COMPLETE,
            x=abs_x,
            y=abs_y,
        )
        return self.rng.randint(*CLOCK_CORNER_S)

    def _emit_offside(self, team_id: str, abs_x: float, abs_y: float) -> int:
        """Caught beyond the last defender: play stops and possession turns over."""
        self.offsides[team_id] += 1
        caught = self._pick_player(team_id, {"FWD": 6.0, "MID": 2.0, "DEF": 0.5})
        self._emit(
            event_type=EventType.OFFSIDE,
            team_id=team_id,
            player=caught,
            outcome=Outcome.LOST,
            x=abs_x,
            y=abs_y,
        )
        return self.rng.randint(*CLOCK_OFFSIDE_S)

    def _maybe_substitutions(self) -> None:
        """At each substitution window, each side may make a change."""
        for team_id in (self.home.team_id, self.away.team_id):
            if self.substitutions[team_id] >= MAX_SUBS_PER_TEAM:
                continue
            if self.rng.random() >= P_SUB_PER_WINDOW:
                continue

            coming_on = self._pick_player(team_id, {"MID": 4.0, "FWD": 3.0, "DEF": 2.0})
            going_off = self._pick_player(team_id, {"MID": 4.0, "FWD": 3.0, "DEF": 2.0})
            if coming_on.player_id == going_off.player_id:
                continue

            self.substitutions[team_id] += 1
            self._note_stoppage(StoppageCause.SUBSTITUTION)
            self._emit(
                event_type=EventType.SUBSTITUTION,
                team_id=team_id,
                player=coming_on,
                outcome=Outcome.COMPLETE,
                x=50.0,
                y=0.0,  # the bench is on the halfway-line touchline
                player_out_id=going_off.player_id,
            )
            self.clock_s += self.rng.randint(*CLOCK_SUB_S)
            self._shock_control(team_id, 0.04)  # fresh legs

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def simulate(self, *, live: bool = False) -> list[Snapshot]:
        """Generate the whole match. Deterministic for a given seed.

        The match is played as two halves. Each half plays its 45 minutes of
        regulation plus the added time its own stoppage ledger earns, so the
        added-time figure is *derived from what actually happened* rather than
        announced by fiat.

        `live=True` inserts the kickoff / half-time / full-time markers that a
        real-time consumer expects to receive.
        """
        home_id, away_id = self.home.team_id, self.away.team_id

        self._set_phase(home_id)  # the possession-time clock starts at kickoff

        if live:
            self._emit(
                event_type=EventType.KICKOFF,
                team_id=home_id,
                player=self.squads[home_id][0],
                outcome=Outcome.COMPLETE,
                x=50.0, y=50.0,
            )
            self._shock_control(home_id, 0.05)  # nominal home advantage

        # The ball begins in the home side's own third.
        self._possessor = home_id
        self._lx, self._ly = 30.0, 50.0

        # --- first half ---------------------------------------------------- #
        self.added_time_s[1] = self._play_period(1)
        if live:
            self._emit(
                event_type=EventType.HALF_TIME,
                team_id=home_id,
                player=self.squads[home_id][0],
                outcome=Outcome.COMPLETE,
                x=50.0, y=50.0,
                stoppage=self.stoppage_breakdown[1],
            )
            # BUG FIX worth understanding properly:
            # The first draft swapped self.home and self.away here, to model
            # teams changing ends. That was wrong twice over.
            #   1. Our frame is FIXED (home always attacks x=100) so a broadcast
            #      overlay can draw both sides in the same physical space for the
            #      full match. Changing ends is modelled by data (`_attack_dir`),
            #      so no swap is needed.
            #   2. Swapping the objects mutated *shared identity* mid-run. Every
            #      structure keyed on team_id -- score, shots, squads -- silently
            #      became wrong. The symptom was a mirrored scoreline.
            # Lesson: never mutate identity to model position.

        # --- second half --------------------------------------------------- #
        # The side that did not kick off restarts, from the centre circle, for
        # real. The clock base and period must be rolled over BEFORE the event
        # is drawn, or the display clock would still be reading half one and
        # stamp the kickoff as "45+..." of the first half.
        self.period = 2
        self._half_start_s = self.clock_s
        self._possessor = away_id
        self._lx, self._ly = 50.0, 50.0
        self._set_phase(away_id)
        if live:
            self._emit(
                event_type=EventType.KICKOFF,
                team_id=away_id,
                player=self.squads[away_id][0],
                outcome=Outcome.COMPLETE,
                x=50.0, y=50.0,
            )
        self.added_time_s[2] = self._play_period(2)
        if live:
            self._emit(
                event_type=EventType.FULL_TIME,
                team_id=home_id,
                player=self.squads[home_id][0],
                outcome=Outcome.COMPLETE,
                x=50.0, y=50.0,
                stoppage=self.stoppage_breakdown[2],
            )

        self._set_phase(None)  # close the final phase so every second is counted
        return self.snapshots

    def _play_period(self, period: int) -> int:
        """Play one half: 45 minutes of regulation plus its added time.

        Returns the added time in seconds. The added time is settled the first
        time the clock passes regulation, because the stoppage ledger is then
        complete -- which is also, roughly, when a referee decides it.
        """
        home_id, away_id = self.home.team_id, self.away.team_id
        self.period = period
        self._half_start_s = self.clock_s
        reg_end = self._half_start_s + HALF_S
        added_s: int | None = None

        while True:
            if added_s is None and self.clock_s >= reg_end:
                breakdown = self._added_time_breakdown(period)
                self.stoppage_breakdown[period] = breakdown
                added_s = breakdown.total_s
            if self.clock_s >= reg_end + (added_s or 0):
                break

            self._update_latent()

            # Substitution windows. Rotations are scheduled by the clock, not by
            # play, so they are checked once per iteration right after the clock
            # advances. Managers begin making changes from around the 46th minute.
            if (
                self._sub_window_idx < len(SUB_WINDOWS_MIN)
                and self.clock_s >= SUB_WINDOWS_MIN[self._sub_window_idx] * 60
            ):
                self._maybe_substitutions()
                self._sub_window_idx += 1

            team_id = self._possessor
            lx, ly = self._lx, self._ly

            # --- pressure event, occasionally --------------------------------- #
            pressure = self._pressure(team_id, lx)
            if pressure > 0.75 and self.rng.random() < 0.12:
                presser = self._pick_player(
                    away_id if team_id == home_id else home_id,
                    {"DEF": 4.0, "MID": 3.0, "FWD": 1.0},
                )
                px, py = self._to_absolute(team_id, lx, ly)
                self._emit(
                    event_type=EventType.PRESSURE,
                    team_id=presser.team_id,
                    player=presser,
                    outcome=Outcome.WON if self.rng.random() < 0.5 else Outcome.LOST,
                    x=px, y=py,
                    pressure_at_location=round(pressure, 3),
                )
                self.clock_s += self.rng.randint(1, 2)

            # --- shoot? -------------------------------------------------------- #
            if lx > 68.0:
                shot_prob = SHOT_PROB_BASE + SHOT_PROB_SCALE * ((lx - 68.0) / 32.0) ** 2
                if self.rng.random() < shot_prob:
                    retained = self._shoot(team_id, lx, ly)
                    if not retained:
                        winner_id = away_id if team_id == home_id else home_id
                        self._set_phase(winner_id)
                        self._possessor = winner_id
                        self._lx, self._ly = 100.0 - lx, 100.0 - ly
                    continue

            # --- otherwise, pass ----------------------------------------------- #
            passer = self._pick_player(
                team_id, {"DEF": 3.0, "MID": 5.0, "FWD": 2.0, "GK": 0.3}
            )
            tx, ty, progressive, pass_type = self._choose_pass(lx, ly)
            distance_m = math.hypot(
                (tx - lx) / 100.0 * PITCH_LEN_M, (ty - ly) / 100.0 * PITCH_WID_M
            )
            p_complete = self._completion_prob(team_id, passer, distance_m, pressure, progressive)
            difficulty = 100.0 * (1.0 - p_complete)
            completed = self.rng.random() < p_complete

            sx, sy = self._to_absolute(team_id, lx, ly)
            ex, ey = self._to_absolute(team_id, tx, ty)

            # Offside. A progressive pass into the final third, aimed at a runner
            # beyond the last defender, is flagged. Play stops and possession
            # turns over. This is checked BEFORE the pass is recorded: the flag
            # supersedes the pass, so only the offside appears in the stream.
            if progressive and lx > 70.0 and self.rng.random() < P_OFFSIDE:
                self.clock_s += self._emit_offside(team_id, ex, ey)
                # The sequence is dead: any pending creator/gift is void.
                self._last_passer_idx = None
                self._error_chance_pending = False
                winner_id = away_id if team_id == home_id else home_id
                self._set_phase(winner_id)
                self._possessor = winner_id
                self._lx, self._ly = 100.0 - lx, 100.0 - ly
                continue

            ball_speed = _clamp(distance_m * 2.4 + self.rng.gauss(0, 6), 8.0, 110.0)

            is_error = (
                not completed
                and lx < DEFENSIVE_THIRD_X
                and self.rng.random() < P_ERROR_BASE + P_ERROR_CHAOS * self.chaos
            )

            self._emit(
                event_type=EventType.PASS,
                team_id=team_id,
                player=passer,
                outcome=Outcome.COMPLETE if completed else Outcome.INCOMPLETE,
                x=sx, y=sy, end_x=ex, end_y=ey,
                pass_distance=round(distance_m, 2),
                pass_difficulty=round(difficulty, 2),
                pressure_at_location=round(pressure, 3),
                ball_speed_kmh=round(ball_speed, 1),
                pass_type_true=pass_type,
                error_true=is_error or None,
            )
            self.clock_s += self.rng.randint(1, 4)

            if completed:
                # The most recent completed pass is the candidate "creator" for
                # any big-chance shot that follows before possession changes.
                # It is cleared on every turnover and restart.
                self._last_passer_idx = len(self.snapshots) - 1
                self._lx, self._ly = tx, ty
                attack_third = tx > 68.0

                # A sequence can also end without a misplaced pass: a foul, the
                # ball running out of play, a clearance. Crucially these consume
                # *realistic amounts of clock*, and that is what actually controls
                # how many passes fit into 90 minutes. A sequence ending with no
                # matching time cost does nothing, because the loop simply starts
                # another sequence to fill the clock.
                if self.rng.random() < P_SEQUENCE_BREAKUP:
                    r = self.rng.random()

                    if r < P_BREAKUP_FOUL:
                        # FOUL by the defending side. Time passes, but the attack
                        # KEEPS the ball via the free kick, so possession is not
                        # touched here. (The original draft both blamed the wrong
                        # team and handed over the ball -- see _emit_foul.)
                        self.clock_s += self._emit_foul(team_id, ex, ey)
                        # A dead-ball restart resets the sequence: the previous
                        # pass did not "create" anything that follows.
                        self._last_passer_idx = None
                        continue

                    if attack_third and r < P_BREAKUP_FOUL + P_BREAKUP_ATTACK_THIRD_CORNER:
                        # Deflected behind by a defender: a corner, restarted by
                        # the attacking side, so possession is likewise retained.
                        self.clock_s += self._emit_corner(team_id, ex, ey)
                        # Set-piece restart: the creator chain restarts here.
                        self._last_passer_idx = None
                        continue

                    # Otherwise possession genuinely turns over. Two flavours
                    # with very different clock costs; that difference is what
                    # controls the total pass volume.
                    if r < 0.72:
                        delay = self.rng.randint(*CLOCK_TURNOVER_THROW_S)
                    else:
                        delay = self.rng.randint(*CLOCK_TURNOVER_OPEN_S)

                    # Possession genuinely changes hands: pending creator/gift
                    # die here, before the loose-ball contest is even decided.
                    self._last_passer_idx = None
                    self._error_chance_pending = False
                    winner_id = self._resolve_turnover(team_id, ex, ey)
                    winner = self._pick_player(
                        winner_id, {"DEF": 4.0, "MID": 3.0, "FWD": 1.0}
                    )
                    self._emit(
                        event_type=EventType.POSSESSION_CHANGE,
                        team_id=winner_id,
                        player=winner,
                        outcome=Outcome.COMPLETE,
                        x=ex,
                        y=ey,
                    )
                    self._shock_control(winner_id, 0.02 + 0.06 * (lx / 100.0))
                    self.clock_s += delay
                    self._possessor = winner_id
                    if winner_id != team_id:
                        # Only flip the frame if possession actually changed.
                        # A loose ball recovered by the SAME side must not
                        # teleport it to the other end.
                        self._lx, self._ly = 100.0 - lx, 100.0 - ly
                continue

            # An unforced error: the give-away hands the ball straight over in a
            # dangerous area, bypassing the normal loose-ball contest. The PASS
            # event above already carried error_true in its latent channel; the
            # observable trace is the failed pass deep in their own third and the
            # immediate turnover that follows.
            if is_error:
                self._error_chance_pending = True
                self._last_passer_idx = None
                winner_id = away_id if team_id == home_id else home_id
                winner = self._pick_player(
                    winner_id, {"DEF": 4.0, "MID": 3.0, "FWD": 1.0}
                )
                self._emit(
                    event_type=EventType.POSSESSION_CHANGE,
                    team_id=winner_id,
                    player=winner,
                    outcome=Outcome.COMPLETE,
                    x=ex, y=ey,
                )
                self._shock_control(winner_id, 0.03)
                self.clock_s += self.rng.randint(1, 3)
                self._set_phase(winner_id)
                self._possessor = winner_id
                self._lx, self._ly = 100.0 - tx, 100.0 - ty
                continue

            # --- pass failed: the opposition wins it ---------------------------- #
            # The challenger gets a foot in FIRST; whether they end up with the
            # ball is a separate question, decided by _resolve_turnover. Keeping
            # "won the tackle" and "has possession" apart is what lets a loose
            # ball exist at all.
            self._last_passer_idx = None
            self._error_chance_pending = False
            challenger_id = away_id if team_id == home_id else home_id
            challenger = self._pick_player(
                challenger_id, {"DEF": 4.0, "MID": 3.5, "FWD": 1.0}
            )
            turnover_type = (
                EventType.INTERCEPTION
                if distance_m < 30.0
                else EventType.TACKLE
            )
            self._emit(
                event_type=turnover_type,
                team_id=challenger_id,
                player=challenger,
                outcome=Outcome.WON,
                x=ex, y=ey,
                pressure_at_location=round(self._pressure(team_id, lx), 3),
            )

            winner_id = self._resolve_turnover(team_id, ex, ey)
            winner = self._pick_player(
                winner_id, {"DEF": 4.0, "MID": 3.5, "FWD": 1.0}
            )
            self._emit(
                event_type=EventType.POSSESSION_CHANGE,
                team_id=winner_id,
                player=winner,
                outcome=Outcome.COMPLETE,
                x=ex, y=ey,
            )
            # Control reacts to turnovers, especially high ones.
            self._shock_control(winner_id, 0.02 + 0.06 * (lx / 100.0))
            self.clock_s += self.rng.randint(1, 3)

            self._possessor = winner_id
            if winner_id != team_id:
                # Only flip the frame if possession changed hands. If the passer
                # won the loose ball back, the ball stays where it is.
                self._lx, self._ly = 100.0 - tx, 100.0 - ty  # new team's own frame

        return added_s or 0

    def _resolve_turnover(self, last_team: str, abs_x: float, abs_y: float) -> str:
        """Decide who ends up with the ball after a turnover.

        Sometimes possession passes cleanly to the other side; sometimes the ball
        is *loose* and both teams fight for it. The chance of a contest rises with
        latent chaos, which is the causal link that makes contested time
        recoverable from the event stream rather than a decorative counter.

        Returns the team that ends up in control, and records time spent
        contested along the way.
        """
        home_id, away_id = self.home.team_id, self.away.team_id
        other = away_id if last_team == home_id else home_id

        if self.rng.random() >= P_CONTEST_BASE + P_CONTEST_CHAOS * self.chaos:
            self._set_phase(other)  # clean turnover
            return other

        # A loose ball. Time spent here is contested, not controlled.
        self._set_phase(None)
        self.contests += 1
        for _ in range(self.rng.randint(1, 2)):
            scrambler_id = self.rng.choice([last_team, other])
            scrambler = self._pick_player(
                scrambler_id, {"MID": 4.0, "DEF": 3.0, "FWD": 2.0}
            )
            self._emit(
                event_type=EventType.LOOSE_BALL,
                team_id=scrambler_id,
                player=scrambler,
                outcome=Outcome.WON if self.rng.random() < 0.5 else Outcome.LOST,
                x=abs_x,
                y=abs_y,
            )
            self.clock_s += self.rng.randint(*CLOCK_LOOSE_BALL_S)

        # The 50/50 tilts toward whoever was in control of the match.
        edge = self.control if last_team == home_id else -self.control
        p_last_recovers = _clamp(0.5 + 0.30 * edge, 0.15, 0.85)
        winner_id = last_team if self.rng.random() < p_last_recovers else other
        self._set_phase(winner_id)
        return winner_id

    # ------------------------------------------------------------------ #
    # Output channels -- note the deliberate asymmetry
    # ------------------------------------------------------------------ #

    def observable_stream(self) -> Iterator[MatchEvent]:
        """The event stream an analyst could actually see. No latent state.

        This is the function the rest of the system calls. Note there is no
        parameter to include the truth -- if you want the truth you must call
        `truth_series()` by name, which makes leakage a deliberate, visible act
        rather than an accident.
        """
        for snap in self.snapshots:
            yield snap.event

    def truth_series(self) -> list[LatentState]:
        """Ground truth, for training and scoring only. Never for inference."""
        return [snap.latent for snap in self.snapshots]


# --------------------------------------------------------------------------- #
# A tiny demo league
# --------------------------------------------------------------------------- #

DEMO_HOME = TeamProfile("HOM", "Northbridge United", attack=1.05, defence=1.10, tempo=1.0, press=0.55)
DEMO_AWAY = TeamProfile("AWY", "Riverside Athletic", attack=0.95, defence=0.92, tempo=0.95, press=0.62)


def _summary(sim: MatchSimulator) -> str:
    h, a = sim.home.team_id, sim.away.team_id
    passes = [e for e in sim.observable_stream() if e.type == EventType.PASS]
    complete = [p for p in passes if p.outcome == Outcome.COMPLETE]
    h_passes = [p for p in passes if p.team == h]
    a_passes = [p for p in passes if p.team == a]

    total_controlled = sim.controlled_s[h] + sim.controlled_s[a]
    possession = {
        h: sim.controlled_s[h] / total_controlled if total_controlled else 0.5,
        a: sim.controlled_s[a] / total_controlled if total_controlled else 0.5,
    }
    elapsed = total_controlled + sim.contested_s
    contested_pct = 100.0 * sim.contested_s / elapsed if elapsed else 0.0

    def pc(side_passes):
        done = [p for p in side_passes if p.outcome == Outcome.COMPLETE]
        return 100.0 * len(done) / max(1, len(side_passes))

    def mmss(sec: int) -> str:
        return f"{sec // 60}:{sec % 60:02d}"

    pass_types = Counter(
        t.pass_type_true for t in sim.truth_series() if t.pass_type_true is not None
    )
    pass_type_line = "  ".join(
        f"{pt.value} {pass_types.get(pt, 0)}" for pt in PassType
    )

    lines = [
        "",
        "=" * 62,
        f"  {sim.home.name} {sim.score[h]} - {sim.score[a]} {sim.away.name}",
        "=" * 62,
        f"  total events        {len(sim.snapshots)}",
        f"  passes              {len(passes)}  (complete {len(complete)})",
        f"  overall completion  {100.0 * len(complete) / max(1, len(passes)):.1f}%",
        f"  possession (time)   {sim.home.name} {100 * possession[h]:.1f}%  "
        f"{sim.away.name} {100 * possession[a]:.1f}%   (contested {contested_pct:.1f}%)",
        f"  shots               {sim.home.name} {sim.shots[h]}  {sim.away.name} {sim.shots[a]}",
        f"  on target           {sim.home.name} {sim.shots_on_target[h]}  "
        f"{sim.away.name} {sim.shots_on_target[a]}",
        f"  in box              {sim.home.name} {sim.shots_in_box[h]}  "
        f"{sim.away.name} {sim.shots_in_box[a]}",
        f"  woodwork            {sum(sim.woodwork.values())}  "
        f"(H {sim.woodwork[h]} / A {sim.woodwork[a]})",
        f"  saves               H {sim.saves[h]}  / A {sim.saves[a]}",
        f"  blocks              H {sim.blocks[h]}  / A {sim.blocks[a]}",
        f"  total xG            {sim.home.name} {sim.shot_quality[h]:.2f}  "
        f"{sim.away.name} {sim.shot_quality[a]:.2f}",
        f"  completion (H/A)    {pc(h_passes):.1f}%  /  {pc(a_passes):.1f}%",
        f"  fouls               {sum(sim.fouls.values())}  "
        f"(H {sim.fouls[h]} / A {sim.fouls[a]})",
        f"  cards               yellows {sum(sim.yellows.values())}  "
        f"reds {sum(sim.reds.values())}",
        f"  corners             {sim.corners[h]} - {sim.corners[a]}",
        f"  offsides            {sim.offsides[h]} - {sim.offsides[a]}",
        f"  substitutions       {sim.substitutions[h]} - {sim.substitutions[a]}",
        f"  loose-ball contests {sim.contests}",
        f"  pass types          {pass_type_line}",
        f"  big chances         {sum(1 for t in sim.truth_series() if t.big_chance_true)}  "
        f"(created-marked {sum(1 for t in sim.truth_series() if t.chance_created_true)})",
        f"  errors              {sum(1 for t in sim.truth_series() if t.error_true)}  "
        f"(led to a shot {sum(1 for t in sim.truth_series() if t.from_error_true)})",
        f"  added time          1H +{mmss(sim.added_time_s[1])}   "
        f"2H +{mmss(sim.added_time_s[2])}",
    ]
    return "\n".join(lines)


def main() -> None:
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id="M0001", seed=42)
    sim.simulate(live=True)

    print(_summary(sim))

    print("\n--- sample of the observable stream (what the agents will see) ---")
    interesting = [
        e for e in sim.observable_stream()
        if e.type in (EventType.SHOT, EventType.SAVE, EventType.BLOCK,
                      EventType.PRESSURE, EventType.KICKOFF,
                      EventType.LOOSE_BALL, EventType.CARD)
    ][:8]
    for e in interesting:
        extra = ""
        if e.shot_quality is not None:
            extra = f" xG={e.shot_quality:.3f} speed={e.shot_speed_kmh}km/h"
        if e.pressure_at_location is not None:
            extra += f" pressure={e.pressure_at_location:.2f}"
        clock = f"{e.minute}" + (f"+{e.stoppage_minute}" if e.stoppage_minute else "")
        print(f"  {clock:>4}' {e.type.value:<16} {e.team} {e.outcome.value:<10}{extra}")

    # Occasionally something matters precisely *because* it happened late: show
    # the first shot fired in added time, and the display clock proves it.
    in_stoppage = [
        e for e in sim.observable_stream()
        if e.type == EventType.SHOT and e.stoppage_minute > 0
    ]
    if in_stoppage:
        e = in_stoppage[0]
        print(
            f"  -- a shot in added time: {e.minute}+{e.stoppage_minute}' "
            f"xG={e.shot_quality:.3f} {e.outcome.value} --"
        )

    print("\n--- a hard pass, to show the difficulty rating ---")
    hardest = max(
        (e for e in sim.observable_stream() if e.pass_difficulty is not None),
        key=lambda e: e.pass_difficulty,
    )
    print(
        f"  {hardest.minute}' {hardest.player_id} "
        f"distance={hardest.pass_distance:.1f}m "
        f"difficulty={hardest.pass_difficulty:.1f}/100 "
        f"pressure={hardest.pressure_at_location:.2f} "
        f"-> {hardest.outcome.value}"
    )

    print("\n--- why the clock ran long (added time, explained) ---")
    for period, breakdown in sorted(sim.stoppage_breakdown.items()):
        if breakdown is None:
            continue
        parts = ", ".join(
            f"{cause.value} {sec // 60}:{sec % 60:02d}"
            for cause, sec in sorted(breakdown.by_cause.items(), key=lambda kv: kv[0].value)
        )
        print(
            f"  half {period}: +{breakdown.total_s // 60}:{breakdown.total_s % 60:02d}  "
            f"({parts})"
        )

    print("\n--- ground truth vs what is observable ---")
    truth = sim.truth_series()
    for idx in (0, len(truth) // 3, (2 * len(truth)) // 3):
        t = truth[idx]
        print(
            f"  {t.clock_s // 60:>3}' latent: control={t.control_true:+.2f} "
            f"chaos={t.chaos_true:.2f} dominant={t.dominant_team} phase={t.phase.value}"
        )
    print(
        "\n  Note: control/chaos/pass-type appear ONLY here. They are absent from\n"
        "  every MatchEvent above -- that separation is what makes the evaluation\n"
        "  honest, and what makes the Phase 1 classifiers scoreable.\n"
    )


if __name__ == "__main__":
    main()
