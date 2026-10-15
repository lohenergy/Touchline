"""Tests for the Render stage (`overlay.py`).

WHY THESE TESTS EXIST
---------------------
The Render stage is where a bug could put *graphics* in front of *meaning* -- a
card at the wrong minute, a moment drawn twice, a summary that never arrives.
None of those are football errors, so the Explain tests would not catch them.
These check the one contract a graphics layer depends on:

  * every insight is rendered exactly once, at its own clock, in clock order;
  * the summary is last and holds until dismissed;
  * the payload carries no latent truth, and the module cannot reach any.

Run directly:      ./.venv/bin/python test_overlay.py
Or under pytest:   ./.venv/bin/python -m pytest test_overlay.py
"""

from __future__ import annotations

import ast
import pathlib

from explainer import MatchExplainer
from overlay import TTL_S, build_overlay
from schemas import OverlayKind, OverlayPayload, ReasonKind
from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

import overlay as overlay_module

H, A = DEMO_HOME.team_id, DEMO_AWAY.team_id


def build(seed: int) -> tuple[list, list]:
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"X{seed:04d}", seed=seed)
    sim.simulate(live=True)
    expl = MatchExplainer(sim.match_id, H, A)
    expl.extend(list(sim.observable_stream()))
    return expl.insights(), build_overlay(expl.insights(), expl.summary())


def test_module_does_not_import_the_simulator() -> None:
    tree = ast.parse(pathlib.Path(overlay_module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert "simulator" not in {n.split(".")[0] for n in names}


def test_payload_contract_carries_no_latent_truth() -> None:
    forbidden = {
        "control_true", "chaos_true", "pass_type_true", "big_chance_true",
        "chance_created_true", "error_true", "from_error_true",
    }
    assert not (forbidden & set(OverlayPayload.model_fields))


def test_every_insight_is_rendered_exactly_once() -> None:
    """No dropped moments, no duplicates -- one card per insight."""
    insights, payloads = build(148)
    moments = [p for p in payloads if p.kind == OverlayKind.KEY_MOMENT]
    assert len(moments) == len(insights)
    assert {p.insight.insight_id for p in moments} == {
        i.insight_id for i in insights
    }


def test_payloads_are_in_clock_order_and_anchored_at_their_moment() -> None:
    _, payloads = build(148)
    clocks = [p.clock_s for p in payloads]
    assert clocks == sorted(clocks)
    for payload in payloads:
        if payload.kind == OverlayKind.KEY_MOMENT:
            assert payload.clock_s == payload.insight.clock_s
            assert payload.ttl_s == TTL_S[payload.insight.kind]


def test_summary_is_last_and_holds_until_dismissed() -> None:
    _, payloads = build(148)
    assert payloads[-1].kind == OverlayKind.MATCH_SUMMARY
    assert payloads[-1].ttl_s == 0
    assert payloads[-1].summary is not None
    assert sum(1 for p in payloads if p.kind == OverlayKind.MATCH_SUMMARY) == 1


def test_payloads_cite_only_real_events() -> None:
    """Whatever a card renders, it can be traced back to the tape."""
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id="X0148", seed=148)
    sim.simulate(live=True)
    known = {e.event_id for e in sim.observable_stream()}
    _, payloads = build(148)
    for payload in payloads:
        if payload.kind == OverlayKind.KEY_MOMENT:
            for event_id in payload.insight.evidence_event_ids:
                assert event_id in known, event_id
            assert any(
                r.kind in (ReasonKind.ORIGIN, ReasonKind.LOCATION)
                for r in payload.insight.reasons
            ), "every moment must keep its grounding reason"


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
    print(f"\n{len(tests) - failures}/{len(tests)} overlay tests passed")
    raise SystemExit(1 if failures else 0)
