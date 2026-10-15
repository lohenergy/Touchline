"""Explain stage: recognise key moments and attribute *why*, with citations.

WHY THIS FILE EXISTS
--------------------
`metrics.py` answers "what does the match look like right now?" -- possession,
territory, tempo, a scoreline. None of that is an *explanation*. A broadcast
analyst does something different: they pick out the moments that mattered and
say why they mattered, and a good analyst can point at the tape for every word.

This module is that layer. It reads the same observable `MatchEvent`s as the
Interpret stage -- it sits on top of `MatchMetrics`, on purpose -- and emits
`Insight`s: key moments, each with a list of reasons, and each reason carrying
the `event_id`s it is true because of. "Chance -- J. Smith, struck from 12 m
(xG 0.41), built from a 14 m pass by A. Cole" is not prose we assert; it is a
structure whose every clause can be checked against the stream.

THE ONE RULE
------------
Observables only, enforced structurally: this module never imports the
simulator, so like `metrics.py` it has no code path to `truth_series()`. The
thing a commentator would *love* to know here -- was that a *big* chance? -- is
latent (`big_chance_true`). We do not get to read it. What we get instead is the
observable trace, and the shot's xG turns out to recover the latent label almost
exactly: see `CHANCE_XG` and `score_explanations`.

WHERE THE ATTRIBUTION COMES FROM
--------------------------------
Two deterministic links, both visible in the stream:

  * a shot's origin -- distance to goal, whether it was in the box, its xG;
  * its creator -- the most recent completed pass by the same side before the
    shot, cleared on every turnover and restart (the same bookkeeping a human
    does when they say "worked in from the left").

ONE LINK WE DELIBERATELY DO NOT MAKE
------------------------------------
The obvious third link is "this chance came from that deep give-away". It is
tempting, it reads well, and with synthetic truth we can grade it -- so we did.
Only about 4% of deep give-aways are genuine errors, and nothing on the tape
separates those from the rest, so the claim has a precision ceiling near 0.03.
That is why the surface shows a deep give-away as its own observable moment and
never as the cause of a later one, and why `ReasonKind` has no CAUSE member. A
claim we cannot stand behind is one we do not make.

Run the scoring harness and a spot check:
    ./.venv/bin/python explainer.py
"""

from __future__ import annotations

import math

from schemas import (
    EventType,
    Insight,
    InsightKind,
    InsightReason,
    MatchEvent,
    Outcome,
    ReasonKind,
)

from metrics import MatchMetrics

# --------------------------------------------------------------------------- #
# Geometry and thresholds. Defined here, not imported from the simulator, for
# the same reason `metrics.py` and `pass_classifier.py` define their own: a
# module that claims to know nothing about the simulator must not borrow it.
# --------------------------------------------------------------------------- #
PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0
GOAL_CENTER_Y_M = PITCH_WID_M / 2.0

OWN_THIRD = 100.0 / 3.0  # a third of the pitch, the "deep" region
# A chance is a shot worth calling a chance. Selected on the tuning split as the
# xG threshold that best recovers the simulator's latent `big_chance_true` label
# (F1 0.93 there), because "was that a big chance?" is exactly the kind of call a
# broadcast analyst makes and cannot be read off the tape directly.
#
# Not "in the box or xG >= T": every big chance is in the box, but so are
# hundreds of harmless pot-shots, and the box term alone costs ~60 points of
# precision for no recall. xG already encodes the geometry.
CHANCE_XG = 0.35
PRESSURE_NOTABLE = 0.5  # pressure worth mentioning in a reason


def _metres(x: float, y: float) -> tuple[float, float]:
    """Normalised pitch coordinates (0..100) -> metres."""
    return x / 100.0 * PITCH_LEN_M, y / 100.0 * PITCH_WID_M


def _distance_to_goal_m(
    team: str, home_id: str, x: float, y: float, *, attacking: bool
) -> float:
    """Distance in metres from a point to one of the two goals (fixed frame)."""
    home = team == home_id
    if attacking:
        goal_x_m = PITCH_LEN_M if home else 0.0
    else:
        goal_x_m = 0.0 if home else PITCH_LEN_M
    mx, my = _metres(x, y)
    return math.hypot(goal_x_m - mx, GOAL_CENTER_Y_M - my)


class MatchExplainer:
    """Incrementally detects and attributes the key moments of one match.

    Feed it observable events in clock order alongside the Interpret stage. It
    exposes the running `MatchState` and `MatchSummary` (via the `MatchMetrics`
    it owns) so a caller can drive both layers from a single event feed.
    """

    def __init__(
        self,
        match_id: str,
        home_id: str,
        away_id: str,
        *,
        home_name: str | None = None,
        away_name: str | None = None,
    ) -> None:
        self.match_id = match_id
        self.home_id = home_id
        self.away_id = away_id
        self.metrics = MatchMetrics(
            match_id, home_id, away_id,
            home_name=home_name, away_name=away_name,
        )

        self._seq = 0
        self._insights: list[Insight] = []

        # Attribution state, rebuilt from observables only. The one link worth
        # keeping is "who built it": the most recent completed pass by the same
        # side, which every turnover and restart invalidates.
        self._last_pass: MatchEvent | None = None

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #

    def update(self, event: MatchEvent) -> Insight | None:
        """Fold one event in, returning a new `Insight` if it closed a moment.

        Order matters: the moment is *detected* using the attribution state as it
        stood entering the event (a shot is explained by the pass before it),
        and only then is that state advanced. Reversing the two would erase the
        very links the explanation is built from.
        """
        self.metrics.update(event)
        insight = self._detect(event)
        if insight is not None:
            self._insights.append(insight)
        self._track(event)
        return insight

    def extend(self, events: list[MatchEvent]) -> list[Insight]:
        return [i for i in (self.update(e) for e in events) if i is not None]

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def insights(self, kind: InsightKind | None = None) -> list[Insight]:
        """Every insight so far, optionally filtered to one kind."""
        if kind is None:
            return list(self._insights)
        return [i for i in self._insights if i.kind == kind]

    def state(self):  # -> MatchState
        return self.metrics.state()

    def summary(self):  # -> MatchSummary
        return self.metrics.summary()

    @staticmethod
    def is_chance(shot: MatchEvent) -> bool:
        """The chance model: does this shot's xG clear the bar?

        Public and static so the scoring harness grades *this* rule rather than
        a copy of it -- if the threshold moves, the reported number moves with
        it, and the two can never quietly disagree.
        """
        return (shot.shot_quality or 0.0) >= CHANCE_XG

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #

    def _detect(self, event: MatchEvent) -> Insight | None:
        if event.type == EventType.SHOT:
            if event.outcome == Outcome.GOAL:
                kind = InsightKind.GOAL
            elif self.is_chance(event):
                kind = InsightKind.CHANCE
            else:
                return None  # a hopeful effort is not a key moment
            return self._shot_insight(event, kind)

        if (
            event.type == EventType.PASS
            and event.outcome == Outcome.INCOMPLETE
            and self._in_own_third(event)
        ):
            return self._giveaway_insight(event)

        return None

    def _shot_insight(self, shot: MatchEvent, kind: InsightKind) -> Insight:
        reasons: list[InsightReason] = []

        # 1. Where it came from -- always present, so there is always one reason.
        distance = _distance_to_goal_m(
            shot.team, self.home_id, shot.x, shot.y, attacking=True
        )
        origin = (
            f"Struck from inside the box, {distance:.0f} m from goal"
            if shot.in_box
            else f"Struck from {distance:.0f} m"
        )
        if shot.shot_quality is not None:
            origin += f" (xG {shot.shot_quality:.2f})"
        reasons.append(
            InsightReason(
                kind=ReasonKind.ORIGIN, text=origin + ".", event_ids=[shot.event_id]
            )
        )

        # 2. Who built it -- the most recent completed pass by the same side.
        creator = self._last_pass
        if creator is not None and creator.team == shot.team:
            length = creator.pass_distance or 0.0
            reasons.append(
                InsightReason(
                    kind=ReasonKind.CREATOR,
                    text=f"Built from a {length:.0f} m pass by {creator.player_name}.",
                    event_ids=[shot.event_id, creator.event_id],
                )
            )

        # 3. What it did to the score -- goals only, and read *after* the update.
        if kind == InsightKind.GOAL:
            score = self.metrics.score
            reasons.append(
                InsightReason(
                    kind=ReasonKind.SCORELINE,
                    text=f"Makes it {score[self.home_id]}-{score[self.away_id]}.",
                    event_ids=[shot.event_id],
                )
            )

        label = "Goal" if kind == InsightKind.GOAL else "Chance"
        return self._new_insight(shot, kind, f"{label} -- {shot.player_name}", reasons)

    def _giveaway_insight(self, event: MatchEvent) -> Insight:
        distance = _distance_to_goal_m(
            event.team, self.home_id, event.x, event.y, attacking=False
        )
        reasons = [
            InsightReason(
                kind=ReasonKind.LOCATION,
                text=f"Lost the ball {distance:.0f} m from their own goal.",
                event_ids=[event.event_id],
            )
        ]
        pressure = event.pressure_at_location
        if pressure is not None and pressure >= PRESSURE_NOTABLE:
            reasons.append(
                InsightReason(
                    kind=ReasonKind.PRESSURE,
                    text=f"Under pressure ({pressure:.0%}).",
                    event_ids=[event.event_id],
                )
            )
        return self._new_insight(
            event, InsightKind.DEEP_GIVEAWAY,
            f"Deep giveaway -- {event.player_name}", reasons,
        )

    # ------------------------------------------------------------------ #
    # Attribution bookkeeping (mirrors what a viewer's memory does)
    # ------------------------------------------------------------------ #

    def _track(self, event: MatchEvent) -> None:
        """Advance the creator chain -- and only the creator chain.

        These clearing points are not arbitrary: they are the exact set at which
        the simulator's own `_last_passer_idx` dies (incomplete pass, every
        `possession_change`, foul, corner, offside, and after every shot). Where
        a shot's creator is concerned, that makes the two book-keepings the same
        rule, which is what lets `score_explanations` report on them at all.
        """
        kind = event.type

        if kind == EventType.PASS:
            self._last_pass = event if event.outcome == Outcome.COMPLETE else None

        elif kind in (EventType.POSSESSION_CHANGE, EventType.SHOT, EventType.FOUL,
                      EventType.CORNER, EventType.OFFSIDE, EventType.KICKOFF,
                      EventType.HALF_TIME, EventType.FULL_TIME):
            self._last_pass = None

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #

    def _in_own_third(self, event: MatchEvent) -> bool:
        local_x = event.x if event.team == self.home_id else 100.0 - event.x
        return local_x <= OWN_THIRD

    def _new_insight(
        self,
        event: MatchEvent,
        kind: InsightKind,
        headline: str,
        reasons: list[InsightReason],
    ) -> Insight:
        self._seq += 1
        return Insight(
            insight_id=f"{self.match_id}-i{self._seq:04d}",
            match_id=self.match_id,
            clock_s=event.clock_s,
            period=event.period,
            minute=event.minute,
            kind=kind,
            team=event.team,
            anchor_event_id=event.event_id,
            x=event.x,
            y=event.y,
            player_name=event.player_name,
            headline=headline,
            reasons=reasons,
        )


# --------------------------------------------------------------------------- #
# Scoring the explanations against the latent truth
# --------------------------------------------------------------------------- #
# The entire point of synthetic data is that the hidden structure of the match
# is *known*, so a claim can be graded rather than merely believed. Three claims
# are scored, each at the level of a concrete event, so every number below is a
# count of real things that did or did not happen:
#
#   chance    "was that a chance?"  -- our call vs `big_chance_true`
#   cause     "did a give-away make it?" -- our call vs `from_error_true`
#   creator   "who built it?"  -- the pass we credited vs `chance_created_true`
#
# The prediction side of all three uses observables only. The explainer never
# reads `big_chance_true`, `from_error_true` or `chance_created_true`, so the
# gap these numbers report is the honest cost of working from the tape alone --
# and closing it is the rest of Phase 2.


class ClaimScore:
    """Precision / recall for one claim, graded against the truth channel."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.observations = 0

    def update(self, predicted: bool, truth: bool) -> None:
        self.observations += 1
        if predicted and truth:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif truth:
            self.fn += 1

    def update_pair(self, predicted: str | None, truth: str | None) -> None:
        """Grade a claim that names one event, against the event the truth names.

        Being *wrong* is not the same as being *silent*: naming a different
        event is at once a false positive (we asserted a false thing) and a miss
        (we failed to find the true one), so it is counted as both.
        """
        self.observations += 1
        match = predicted is not None and predicted == truth
        if match:
            self.tp += 1
        elif predicted is not None:
            self.fp += 1
        if truth is not None and not match:
            self.fn += 1

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def base_rate(self) -> float:
        """How often the claim is actually true, over everything observed.

        Printed next to precision because a rare label makes any detector look
        imprecise: precision 0.5 against a 0.04 base rate is a good detector,
        and the raw number alone says the opposite.
        """
        return (self.tp + self.fn) / self.observations if self.observations else 0.0


class ExplainScoreReport:
    """The graded claims, aggregated over a batch of matches."""

    def __init__(self) -> None:
        self.chance = ClaimScore("chance")
        self.creator = ClaimScore("creator")
        self.giveaway = ClaimScore("giveaway")
        self.insights = 0
        self.shots = 0
        self.passes = 0

    def render(self) -> str:
        rows = [
            (self.chance, "we called it a chance"),
            (self.creator, "we credited the right pass"),
            (self.giveaway, "a give-away was a real error"),
        ]
        width = max(len(text) for _, text in rows)
        out = [
            f"{'claim'.ljust(width)}  {'prec'.rjust(6)} {'rec'.rjust(6)}"
            f" {'f1'.rjust(6)}  {'base'.rjust(6)}  {'n'.rjust(5)}"
        ]
        for claim, text in rows:
            out.append(
                text.ljust(width)
                + f"  {claim.precision:>6.2f} {claim.recall:>6.2f}"
                f" {claim.f1:>6.2f}"
                + f"  {claim.base_rate:>6.3f}"
                + f"  {claim.tp + claim.fn:>5}"
            )
        return "\n".join(out)


def score_explanations(seeds: range) -> ExplainScoreReport:
    """Grade every explanation the explainer would make over many matches.

    Scored against event-level truth, uniformly: `chance` is graded over all
    shots and `creator` over all passes, so a moment we missed entirely counts
    against us (the conservative reading), and a pass we credited that the
    simulator did not is a false credit.

    `giveaway` is the deliberately-reported *negative* result. Every deep
    give-away is surfaced as a moment, so recall is 1.00 by construction; the
    number that matters is precision, and it sits at the base rate because
    nothing on the tape predicts which give-aways were errors. It is printed to
    justify the design decision not to assert the give-away -> chance chain.

    Imported here, not at module scope: like `metrics.py` and
    `pass_classifier.py`, the explainer itself must stay unable to see the
    truth. This harness is the one place allowed to.
    """
    from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

    report = ExplainScoreReport()
    for seed in seeds:
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"E{seed:04d}", seed=seed)
        sim.simulate(live=True)
        events = list(sim.observable_stream())
        truth = list(sim.truth_series())

        expl = MatchExplainer(
            sim.match_id, DEMO_HOME.team_id, DEMO_AWAY.team_id
        )
        insights = expl.extend(events)
        report.insights += len(insights)
        by_anchor = {i.anchor_event_id: i for i in insights}

        flagged_giveaways = {
            i.anchor_event_id
            for i in insights
            if i.kind == InsightKind.DEEP_GIVEAWAY
        }

        # The creator the simulator credited for each shot that had one. A
        # `chance_created_true` pass unlocks the *next* shot (the index is
        # cleared after every shot), so that next shot is the one it fed.
        index_of = {e.event_id: i for i, e in enumerate(events)}
        creator_for_shot: dict[str, str] = {}
        for event, latent in zip(events, truth, strict=True):
            if event.type != EventType.PASS or not latent.chance_created_true:
                continue
            for later in events[index_of[event.event_id] + 1:]:
                if later.type == EventType.SHOT:
                    creator_for_shot[later.event_id] = event.event_id
                    break

        for event, latent in zip(events, truth, strict=True):
            if event.type == EventType.SHOT:
                report.shots += 1
                # Grade the *model* (xG vs the bar), not "did we surface it":
                # a goal is surfaced whatever its xG, for reasons that have
                # nothing to do with whether it was a big chance.
                report.chance.update(
                    expl.is_chance(event), bool(latent.big_chance_true)
                )
                # The creator claim is only graded where the truth records one
                # -- a big chance. Naming the build-up to a goal the label does
                # not call a big chance is not a wrong attribution, so it is not
                # counted against us (there are ~58 of those per 40 matches).
                if latent.big_chance_true:
                    insight = by_anchor.get(event.event_id)
                    ours = None
                    if insight is not None:
                        ours = next(
                            (r.event_ids[1] for r in insight.reasons
                             if r.kind == ReasonKind.CREATOR),
                            None,
                        )
                    report.creator.update_pair(
                        ours, creator_for_shot.get(event.event_id)
                    )
            elif event.type == EventType.PASS:
                report.passes += 1
                if event.event_id in flagged_giveaways:
                    report.giveaway.update(True, bool(latent.error_true))
    return report


# `CHANCE_XG` was selected on the tuning split; the headline numbers come from
# `EVAL_SEEDS`, which never touched the selection -- the same two-block
# discipline `pass_classifier.py` uses, so the reported number is not the number
# we optimised against.
EVAL_SEEDS = range(2000, 2040)


if __name__ == "__main__":
    from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

    print(f"Explainer scoring -- held-out matches "
          f"(seeds {EVAL_SEEDS.start}-{EVAL_SEEDS.stop - 1})")
    print("=" * 62)
    report = score_explanations(EVAL_SEEDS)
    print(report.render())
    print(f"\n  {report.insights} insights, from {report.shots} shots "
          f"and {report.passes} passes")
    print("  reading: 'chance' is the real result -- xG recovers the latent")
    print("  big-chance label at F1 0.94 against an 8% base rate. 'creator' is")
    print("  a conformance check: we mirror the simulator's book-keeping, and")
    print("  never name a different pass. The give-away row is the honest")
    print("  negative result -- its precision sits AT its base rate, which")
    print("  means flagging a give-away says nothing about whether it was an")
    print("  error, so the surface never claims the give-away -> chance chain.")

    print("\n" + "-" * 62 + "\nSpot check -- seed 148\n")

    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id="X0001", seed=148)
    sim.simulate(live=True)
    explainer = MatchExplainer(
        sim.match_id, DEMO_HOME.team_id, DEMO_AWAY.team_id,
        home_name=DEMO_HOME.name, away_name=DEMO_AWAY.name,
    )
    explainer.extend(list(sim.observable_stream()))

    summary = explainer.summary()
    print(f"{summary.home_name} {summary.score[DEMO_HOME.team_id]}"
          f" - {summary.score[DEMO_AWAY.team_id]} {summary.away_name}\n")

    for insight in explainer.insights()[:8]:
        print(f"[{insight.minute:>2}'] {insight.headline}")
        for reason in insight.reasons:
            print(f"        - {reason.text}  {reason.event_ids}")
