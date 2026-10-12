"""Interpret stage: turn the observable event stream into `MatchState`.

WHY THIS FILE EXISTS
--------------------
`simulator.py` produces *events*. Everything that makes the demo worth watching
-- the overlay, the narrated insight, the full-time summary screen -- needs
*meaning*: who is on top, how much of the game the ball is loose, where the play
is happening. Deriving that meaning from observables is this module's only job.
It is the "Interpret" box in the architecture, and it is the layer every later
consumer (explainer, narrator, webapp) sits on top of.

THE ONE RULE
------------
This stage may read observable `MatchEvent`s and nothing else. It must never
touch `LatentState`. If it did, every number it produces would be reading the
answer off the input, and the project's central claim -- that the hidden
structure of a match can be *recovered* from what a broadcast camera can see --
would be vacuous. The discipline is enforced structurally: this module does not
import the simulator at all, so it has no way to reach the truth.

WHERE POSSESSION COMES FROM
---------------------------
Possession is a *time* decomposition, not a pass count. Every second between two
events is attributed to home, away, or a contested (loose-ball) phase, using only
observable cues: whose pass or shot was last seen, and whether the ball has just
been turned over. The three buckets sum to the elapsed clock *by construction*;
`test_metrics.py` asserts it. A decomposition that silently drops time would look
authoritative and be quietly wrong, which is worse than no decomposition.

BASELINES, NOT ANSWERS
----------------------
`control` and `chaos` are the real prizes -- the latent quantities the whole
project exists to recover. Phase 1 ships deliberately transparent *baselines*
(rolling possession share; the recent share of unstructured events). They are
not claimed to be accurate. Phase 2 learns better estimators and, because the
simulator knows the truth, scores them against `control_true` / `chaos_true`.
Shipping the weak baseline first is the point: you cannot improve what you have
not measured.

Run a spot check:
    ./.venv/bin/python metrics.py
"""

from __future__ import annotations

from collections import deque

from schemas import (
    CardType,
    EventType,
    MatchEvent,
    MatchState,
    MatchSummary,
    Outcome,
    StoppageBreakdown,
)

# How far back a "rolling" statistic looks. Ten minutes is roughly one phase of
# a match -- long enough to be stable, short enough that a swing in momentum
# shows up while the commentator is still talking about it.
WINDOW_S = 600.0

# Attacking-local x beyond which play counts as "the final third". Matches the
# simulator's convention, but is defined *here* rather than imported: the whole
# point of this module is that it knows nothing about the simulator.
FINAL_THIRD_X = 68.0

# Shot outcomes after which possession demonstrably leaves the shooter's side
# (goal kick, keeper's ball, or a dead-ball restart for the defence).
_SHOT_OVER = {
    Outcome.GOAL,
    Outcome.SAVED,
    Outcome.WOODWORK,
    Outcome.OFF_TARGET,
}

# Events that do not belong to a controlled phase: the ball is loose, being won
# back, or has been given away. Used only by the chaos baseline.
_UNSTRUCTURED = {EventType.LOOSE_BALL, EventType.OFFSIDE, EventType.POSSESSION_CHANGE}


def _share(mapping: dict[str, float], key: str, total: float) -> float:
    """A team's share of a total, defaulting to an even split when empty."""
    return mapping.get(key, 0.0) / total if total else 0.5


class MatchMetrics:
    """Incrementally interprets one match from its observable events.

    Feed it events with `update()` as they arrive (in clock order) and ask it for
    a `MatchState` at any point or a `MatchSummary` at the end. State is derived
    from accumulators, so it is cheap to poll -- which is what lets the server
    push a fresh snapshot without re-scanning the whole match.
    """

    def __init__(
        self,
        match_id: str,
        home_id: str,
        away_id: str,
        *,
        home_name: str | None = None,
        away_name: str | None = None,
        window_s: float = WINDOW_S,
    ) -> None:
        self.match_id = match_id
        self.home_id = home_id
        self.away_id = away_id
        self.home_name = home_name or home_id
        self.away_name = away_name or away_id
        self.window_s = window_s

        self.clock_s = 0
        self.period = 1

        # --- possession as a time decomposition ---------------------------- #
        # `_controller` is the team we currently believe has the ball;
        # `_contested` means neither side does. Every event closes the span that
        # just elapsed and opens the next one.
        self._controller: str = home_id
        self._contested = False
        self.controlled_s: dict[str, float] = {home_id: 0.0, away_id: 0.0}
        self.contested_s = 0.0

        # --- observable aggregates ----------------------------------------- #
        self.score = {home_id: 0, away_id: 0}
        self.shots = {home_id: 0, away_id: 0}
        self.shots_on_target = {home_id: 0, away_id: 0}
        self.shots_in_box = {home_id: 0, away_id: 0}
        self.woodwork = {home_id: 0, away_id: 0}
        self.saves = {home_id: 0, away_id: 0}
        self.blocks = {home_id: 0, away_id: 0}
        self.xg: dict[str, float] = {home_id: 0.0, away_id: 0.0}

        self.passes = {home_id: 0, away_id: 0}
        self.passes_complete = {home_id: 0, away_id: 0}
        self.pass_difficulty_sum: dict[str, float] = {home_id: 0.0, away_id: 0.0}
        self.final_third = {home_id: 0, away_id: 0}

        self.fouls = {home_id: 0, away_id: 0}
        self.yellows = {home_id: 0, away_id: 0}
        self.reds = {home_id: 0, away_id: 0}
        self.corners = {home_id: 0, away_id: 0}
        self.offsides = {home_id: 0, away_id: 0}
        self.substitutions = {home_id: 0, away_id: 0}
        self.stoppage: dict[int, StoppageBreakdown] = {}

        # --- rolling window ------------------------------------------------- #
        self._window: deque[MatchEvent] = deque()
        # (end_clock_s, duration_s, owner-or-None) for the time decomposition.
        # Kept for two windows: the older half is what the trend is measured
        # against.
        self._spans: deque[tuple[float, float, str | None]] = deque()

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #

    def update(self, event: MatchEvent) -> None:
        """Fold one observable event into the accumulators."""
        # 1. Close the phase span that has just elapsed, then open the next one.
        dt = event.clock_s - self.clock_s
        if dt > 0:
            owner = None if self._contested else self._controller
            if owner is None:
                self.contested_s += dt
            else:
                self.controlled_s[owner] += dt
            self._spans.append((float(event.clock_s), float(dt), owner))
        self.clock_s = event.clock_s
        self.period = event.period

        # 2. Count the event itself.
        self._accumulate(event)

        # 3. Decide who controls the ball for the *next* interval.
        self._transition(event)

        # 4. Keep the rolling window bounded.
        self._window.append(event)
        cutoff = self.clock_s - self.window_s
        while self._window and self._window[0].clock_s < cutoff:
            self._window.popleft()
        while self._spans and self._spans[0][0] < self.clock_s - 2 * self.window_s:
            self._spans.popleft()

    def extend(self, events: list[MatchEvent]) -> None:
        for event in events:
            self.update(event)

    # ------------------------------------------------------------------ #
    # Per-event folds
    # ------------------------------------------------------------------ #

    def _accumulate(self, event: MatchEvent) -> None:
        team = event.team
        kind = event.type

        if kind == EventType.SHOT:
            self.shots[team] += 1
            if event.shot_quality is not None:
                self.xg[team] += event.shot_quality
            if event.in_box:
                self.shots_in_box[team] += 1
            if event.outcome in (Outcome.GOAL, Outcome.SAVED):
                self.shots_on_target[team] += 1
            elif event.outcome == Outcome.WOODWORK:
                self.woodwork[team] += 1
            if event.outcome == Outcome.GOAL:
                self.score[team] += 1
        elif kind == EventType.SAVE:
            self.saves[team] += 1
        elif kind == EventType.BLOCK:
            self.blocks[team] += 1
        elif kind == EventType.PASS:
            self.passes[team] += 1
            if event.outcome == Outcome.COMPLETE:
                self.passes_complete[team] += 1
            if event.pass_difficulty is not None:
                self.pass_difficulty_sum[team] += event.pass_difficulty
            if self._local_x(team, event.x) > FINAL_THIRD_X:
                self.final_third[team] += 1
        elif kind == EventType.FOUL:
            self.fouls[team] += 1
        elif kind == EventType.CARD:
            if event.card_type == CardType.YELLOW:
                self.yellows[team] += 1
            elif event.card_type == CardType.RED:
                self.reds[team] += 1
        elif kind == EventType.CORNER:
            self.corners[team] += 1
        elif kind == EventType.OFFSIDE:
            self.offsides[team] += 1
        elif kind == EventType.SUBSTITUTION:
            self.substitutions[team] += 1
        elif kind in (EventType.HALF_TIME, EventType.FULL_TIME):
            if event.stoppage is not None:
                self.stoppage[event.period] = event.stoppage

    def _transition(self, event: MatchEvent) -> None:
        """Re-point the controller for the interval that follows this event."""
        kind = event.type

        if kind == EventType.KICKOFF:
            self._controller, self._contested = event.team, False
        elif kind == EventType.PASS:
            if event.outcome == Outcome.COMPLETE:
                self._controller, self._contested = event.team, False
            else:
                # A failed pass leaves the ball live until someone settles it.
                self._contested = True
        elif kind == EventType.POSSESSION_CHANGE:
            self._controller, self._contested = event.team, False
        elif kind == EventType.CORNER:
            # A corner is restarted by the attacking side, so it retains.
            self._controller, self._contested = event.team, False
        elif kind == EventType.OFFSIDE:
            self._controller, self._contested = self._other(event.team), False
        elif kind == EventType.SHOT:
            if event.outcome in _SHOT_OVER:
                self._controller, self._contested = self._other(event.team), False
            elif event.outcome == Outcome.BLOCKED:
                # A block may be retained (corner) or lost -- undecidable without
                # look-ahead, so mark it contested and let the next event settle.
                self._contested = True
        elif kind in (EventType.TACKLE, EventType.INTERCEPTION, EventType.LOOSE_BALL):
            self._contested = True

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def state(self) -> MatchState:
        """The rolling snapshot of meaning at the current clock."""
        h, a = self.home_id, self.away_id
        controlled = self.controlled_s[h] + self.controlled_s[a]
        elapsed = controlled + self.contested_s

        win = self._window_control()
        win_total = win[h] + win[a]
        margin = (win[h] - win[a]) / win_total if win_total else 0.0
        prev = self._previous_control()
        prev_total = prev[h] + prev[a]
        prev_margin = (prev[h] - prev[a]) / prev_total if prev_total else 0.0
        trend = (margin - prev_margin) / (self.window_s / 60.0)

        control = {h: _share(win, h, win_total), a: _share(win, a, win_total)}

        tick_denom = self.final_third[h] + self.final_third[a]
        territory = {h: _share(self.final_third, h, tick_denom),
                     a: _share(self.final_third, a, tick_denom)}

        return MatchState(
            match_id=self.match_id,
            clock_s=self.clock_s,
            period=self.period,
            score=dict(self.score),
            possession={h: _share(self.controlled_s, h, controlled),
                        a: _share(self.controlled_s, a, controlled)},
            contested_possession=(self.contested_s / elapsed if elapsed else 0.0),
            control=control,
            control_trend=trend,
            chaos=self._chaos(),
            tempo=self._tempo(),
            territory=territory,
            pass_completion={
                h: self._ratio(self.passes_complete[h], self.passes[h]),
                a: self._ratio(self.passes_complete[a], self.passes[a]),
            },
            pass_difficulty_mean={
                h: self._ratio(self.pass_difficulty_sum[h], self.passes[h]),
                a: self._ratio(self.pass_difficulty_sum[a], self.passes[a]),
            },
            shot_quality_total=dict(self.xg),
            shots=dict(self.shots),
        )

    def summary(self) -> MatchSummary:
        """The full-time aggregate -- the summary screen, in one object."""
        state = self.state()
        return MatchSummary(
            match_id=self.match_id,
            home=self.home_id,
            away=self.away_id,
            home_name=self.home_name,
            away_name=self.away_name,
            score=dict(self.score),
            possession=state.possession,
            contested_pct=100.0 * state.contested_possession,
            shots=dict(self.shots),
            shots_on_target=dict(self.shots_on_target),
            shots_in_box=dict(self.shots_in_box),
            woodwork=dict(self.woodwork),
            saves=dict(self.saves),
            blocks=dict(self.blocks),
            xg=dict(self.xg),
            passes=dict(self.passes),
            pass_completion={
                self.home_id: self._ratio(self.passes_complete[self.home_id],
                                          self.passes[self.home_id]),
                self.away_id: self._ratio(self.passes_complete[self.away_id],
                                          self.passes[self.away_id]),
            },
            pass_difficulty_mean={
                self.home_id: self._ratio(self.pass_difficulty_sum[self.home_id],
                                          self.passes[self.home_id]),
                self.away_id: self._ratio(self.pass_difficulty_sum[self.away_id],
                                          self.passes[self.away_id]),
            },
            corners=dict(self.corners),
            offsides=dict(self.offsides),
            fouls=dict(self.fouls),
            yellows=dict(self.yellows),
            reds=dict(self.reds),
            substitutions=dict(self.substitutions),
            territory=state.territory,
            added_time_s={p: b.total_s for p, b in self.stoppage.items()},
            stoppage=dict(self.stoppage),
        )

    # ------------------------------------------------------------------ #
    # Rolling statistics
    # ------------------------------------------------------------------ #

    def _window_control(self) -> dict[str, float]:
        """Controlled time per team inside the last window."""
        out = {self.home_id: 0.0, self.away_id: 0.0}
        start = self.clock_s - self.window_s
        for end, duration, owner in self._spans:
            if owner is not None and end >= start:
                out[owner] += duration
        return out

    def _previous_control(self) -> dict[str, float]:
        """Controlled time in the window *before* the last one, for the trend."""
        out = {self.home_id: 0.0, self.away_id: 0.0}
        start = self.clock_s - 2 * self.window_s
        end_cut = self.clock_s - self.window_s
        for end, duration, owner in self._spans:
            if owner is not None and start <= end < end_cut:
                out[owner] += duration
        return out

    def _chaos(self) -> float:
        """Baseline chaos: the recent share of unstructured events."""
        if not self._window:
            return 0.0
        loose = 0
        for e in self._window:
            if e.type in _UNSTRUCTURED:
                loose += 1
            elif e.type == EventType.PASS and e.outcome != Outcome.COMPLETE:
                loose += 1
        return loose / len(self._window)

    def _tempo(self) -> float:
        """Events per minute over the window actually covered so far."""
        covered_s = min(self.clock_s, self.window_s)
        if covered_s <= 0:
            return 0.0
        return len(self._window) / (covered_s / 60.0)

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #

    def _other(self, team: str) -> str:
        return self.away_id if team == self.home_id else self.home_id

    def _local_x(self, team: str, x: float) -> float:
        """Absolute x -> attacking-local x (home attacks x=100, away x=0)."""
        return x if team == self.home_id else 100.0 - x

    @staticmethod
    def _ratio(numerator: float, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0


def summarise(
    events: list[MatchEvent],
    match_id: str,
    home_id: str,
    away_id: str,
    **kwargs: str,
) -> MatchSummary:
    """Convenience: interpret a whole match at once and return its summary."""
    metrics = MatchMetrics(match_id, home_id, away_id, **kwargs)
    metrics.extend(events)
    return metrics.summary()


if __name__ == "__main__":
    # Spot check: interpret one match and show the full-time summary the webapp
    # will render, computed purely from the observable stream.
    import json

    from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id="M0001", seed=42)
    sim.simulate(live=True)

    metrics = MatchMetrics(
        "M0001", DEMO_HOME.team_id, DEMO_AWAY.team_id,
        home_name=DEMO_HOME.name, away_name=DEMO_AWAY.name,
    )
    metrics.extend(list(sim.observable_stream()))

    state = metrics.state()
    print(f"{metrics.home_name} {state.score[DEMO_HOME.team_id]}"
          f" - {state.score[DEMO_AWAY.team_id]} {metrics.away_name}")
    print(f"possession (rolling window): "
          f"{100 * state.control[DEMO_HOME.team_id]:.1f}% / "
          f"{100 * state.control[DEMO_AWAY.team_id]:.1f}%")
    print(f"chaos baseline: {state.chaos:.2f}   tempo: {state.tempo:.1f} ev/min")
    print()
    print(json.dumps(metrics.summary().model_dump(), indent=2))
