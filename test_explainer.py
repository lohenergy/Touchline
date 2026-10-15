"""Tests for the Explain stage (`explainer.py`).

WHY THESE TESTS EXIST
---------------------
The value of an explanation is that you can check it. These tests check the
things that would make the Explain stage's claims worthless if they broke:

  1. Every citation is real and in the past. Each `reason` points at actual
     events from the observable stream, and no reason may cite an event that
     happens *after* the moment it explains. A claim that cites the future, or
     cites an event that never happened, is not evidence -- it is decoration.
  2. No latent truth leaks. `Insight` must not name a latent quantity, and the
     module must not import the simulator. The analyst's adjectives (big chance,
     error) are *inferred* from observables and scored later, never read off the
     answer.
  3. The two claims we do make hold up. The chance model recovers the latent
     big-chance label, and the creator we credit is the pass a human would;
     conversely, the give-away flag is pinned as carrying no signal, so that if
     someone later "fixes" it into looking predictive they get a failing test
     and have to argue for it.

Run directly:      ./.venv/bin/python test_explainer.py
Or under pytest:   ./.venv/bin/python -m pytest test_explainer.py
"""

from __future__ import annotations

import ast
import pathlib

from schemas import (
    EventType,
    Insight,
    InsightKind,
    MatchEvent,
    Outcome,
    ReasonKind,
)
from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

import explainer as explainer_module
from explainer import MatchExplainer, score_explanations

H, A = DEMO_HOME.team_id, DEMO_AWAY.team_id


def explain(seed: int) -> tuple[MatchSimulator, MatchExplainer, dict[str, MatchEvent]]:
    """One match, its explainer run to full time, and an event_id -> event map."""
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"X{seed:04d}", seed=seed)
    sim.simulate(live=True)
    events = list(sim.observable_stream())
    expl = MatchExplainer(sim.match_id, H, A)
    expl.extend(events)
    by_id = {e.event_id: e for e in events}
    return sim, expl, by_id


# --------------------------------------------------------------------------- #
# 1. Leak-proofing
# --------------------------------------------------------------------------- #
def test_module_does_not_import_the_simulator() -> None:
    """No module-level import of the simulator -- truth stays unreachable."""
    tree = ast.parse(pathlib.Path(explainer_module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        roots = {n.split(".")[0] for n in names}
        assert "simulator" not in roots, "module-level simulator import leaks truth"


def test_insight_contract_carries_no_latent_truth() -> None:
    """`Insight` may not name any latent field."""
    forbidden = {
        "control_true", "chaos_true", "pass_type_true", "big_chance_true",
        "chance_created_true", "error_true", "from_error_true", "big_chances",
        "errors", "phase", "dominant_team",
    }
    leaked = forbidden & set(Insight.model_fields)
    assert not leaked, f"latent truth leaked into Insight: {leaked}"


# --------------------------------------------------------------------------- #
# 2. Citations are real and in the past
# --------------------------------------------------------------------------- #
def test_every_citation_is_a_real_event_from_the_past() -> None:
    """A reason may only cite events that exist and do not postdate the moment.

    This is the audit-trail test. It walks every citation in a batch of matches
    and checks it against the observable stream itself -- the same bytes the
    viewer saw. A citation to the future would mean the explanation "knew" an
    outcome before it happened; a citation to a missing id would mean it was
    invented.
    """
    for seed in range(15):
        _, expl, by_id = explain(seed)
        for insight in expl.insights():
            for reason in insight.reasons:
                for event_id in reason.event_ids:
                    assert event_id in by_id, (
                        f"seed {seed}: reason cites unknown event {event_id}"
                    )
                    assert by_id[event_id].clock_s <= insight.clock_s, (
                        f"seed {seed}: reason cites the future "
                        f"({event_id} at {by_id[event_id].clock_s}s > "
                        f"{insight.clock_s}s)"
                    )


def test_evidence_ids_are_the_union_of_reason_citations() -> None:
    """The flat `evidence_event_ids` view agrees with the per-reason citations."""
    _, expl, _ = explain(148)
    for insight in expl.insights():
        union: list[str] = []
        for reason in insight.reasons:
            for event_id in reason.event_ids:
                if event_id not in union:
                    union.append(event_id)
        assert insight.evidence_event_ids == union


def test_anchor_is_the_first_citation() -> None:
    """`anchor_event_id` is the event the moment *is* -- always the first cite.

    The overlay lines an insight up with the event it explains through this id,
    so it has to be a real event and has to be the one the moment is anchored
    on, not merely one of several it cites.
    """
    for seed in range(15):
        _, expl, by_id = explain(seed)
        for insight in expl.insights():
            assert insight.anchor_event_id == insight.evidence_event_ids[0]
            assert insight.anchor_event_id in by_id


def test_creator_reason_names_a_completed_pass_by_the_same_side() -> None:
    """'Built from ... pass' must cite a same-side completed pass before it."""
    for seed in range(15):
        _, expl, by_id = explain(seed)
        for insight in expl.insights():
            for reason in insight.reasons:
                if reason.kind != ReasonKind.CREATOR:
                    continue
                assert len(reason.event_ids) == 2
                assert reason.event_ids[0] == insight.anchor_event_id
                pass_event = by_id[reason.event_ids[1]]
                assert pass_event.type == EventType.PASS
                assert pass_event.outcome == Outcome.COMPLETE
                assert pass_event.team == insight.team
                assert pass_event.clock_s <= insight.clock_s


# --------------------------------------------------------------------------- #
# 3. The moments are actually found and agree with the tape
# --------------------------------------------------------------------------- #
def test_goals_match_the_scoreline() -> None:
    """One GOAL insight per goal, and no goal goes unexplained."""
    for seed in range(15):
        sim, expl, by_id = explain(seed)
        goals = sum(sim.score.values())
        goal_insights = expl.insights(InsightKind.GOAL)
        assert len(goal_insights) == goals, (seed, len(goal_insights), goals)

        shot_goals = {
            event_id for event_id, e in by_id.items()
            if e.type == EventType.SHOT and e.outcome == Outcome.GOAL
        }
        cited = {i.evidence_event_ids[0] for i in goal_insights}
        assert cited == shot_goals, seed


def test_all_three_kinds_are_produced() -> None:
    """Across a handful of matches, all three kinds show up."""
    seen: set[InsightKind] = set()
    for seed in range(30):
        _, expl, _ = explain(seed)
        seen.update(i.kind for i in expl.insights())
    assert seen == set(InsightKind), f"missing kinds: {set(InsightKind) - seen}"


# --------------------------------------------------------------------------- #
# 4. The claims we make actually hold up
# --------------------------------------------------------------------------- #
def test_chance_model_recovers_the_latent_big_chance_label() -> None:
    """The scored headline: xG recovers `big_chance_true`, well above base rate."""
    report = score_explanations(range(2000, 2010))
    assert report.chance.f1 > 0.80, report.chance.f1
    assert report.chance.precision > report.chance.base_rate * 5, (
        "the chance model is barely beating the base rate"
    )


def test_giveaway_flag_carries_no_signal() -> None:
    """Pinned negative result: the give-away flag does not predict `error_true`.

    If this starts failing because precision has risen well clear of the base
    rate, someone has found real signal in the tape -- and then the design
    decision not to surface the give-away -> chance chain deserves revisiting on
    purpose, not by accident.
    """
    report = score_explanations(range(2000, 2010))
    assert report.giveaway.recall == 1.0, "we should surface every deep give-away"
    assert report.giveaway.precision < 2 * report.giveaway.base_rate, (
        report.giveaway.precision, report.giveaway.base_rate
    )


# --------------------------------------------------------------------------- #
# 5. Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_explains_itself_identically() -> None:
    _, first, _ = explain(217)
    _, second, _ = explain(217)
    assert [i.model_dump() for i in first.insights()] == [
        i.model_dump() for i in second.insights()
    ]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} explainer tests passed")
    raise SystemExit(1 if failures else 0)
