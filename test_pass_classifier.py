"""Tests for the geometric pass classifier (`pass_classifier.py`).

WHY THESE TESTS EXIST
---------------------
This is the first place in the project where a model is *scored against ground
truth*, so the tests guard the two things that make that score meaningful:

  1. The classifier cannot see the answer. Its inputs are an observable
     `MatchEvent` and the home team id -- nothing else -- and the module does not
     import the simulator at module scope. If a truth channel ever leaks in, the
     measured accuracy becomes a lie, so this is asserted structurally.
  2. The score is real. A held-out block of matches must be classified well
     above the majority-class baseline, and every class must actually be
     recovered (a classifier that only ever predicts the majority class would
     pass a naive accuracy check but fail the per-class one).

A handful of hand-built passes pin the specific geometric rules -- especially
that "long" means *forward travel*, not euclidean distance.

Run directly:      ./.venv/bin/python test_pass_classifier.py
Or under pytest:   ./.venv/bin/python -m pytest test_pass_classifier.py
"""

from __future__ import annotations

import ast
import inspect
import pathlib

from schemas import EventType, MatchEvent, Outcome, PassType
from simulator import DEMO_AWAY, DEMO_HOME

from pass_classifier import classify_pass, pass_distance_m

H, A = DEMO_HOME.team_id, DEMO_AWAY.team_id
EVAL_SEEDS = range(2000, 2010)  # a small held-out block; disjoint from tuning


def make_pass(team: str, x: float, y: float, end_x: float, end_y: float) -> MatchEvent:
    """A minimal, valid observable pass event -- no latent fields exist on it."""
    return MatchEvent(
        event_id="e", match_id="t", clock_s=0, minute=0, period=1,
        type=EventType.PASS, team=team, player_id="p", player_name="P",
        player_number=9, outcome=Outcome.COMPLETE,
        x=x, y=y, end_x=end_x, end_y=end_y,
    )


# --------------------------------------------------------------------------- #
# 1. Leak-proofing: the classifier is not allowed to reach the truth
# --------------------------------------------------------------------------- #
def test_module_does_not_import_the_simulator() -> None:
    """No module-level import of the simulator.

    `score_seeds` imports it *inside* the function, on purpose: the scoring
    harness is the one place allowed to look at the answer, and putting the
    import there keeps it out of everything else. A module-level import would
    make truth reachable from `classify_pass` and void the whole exercise.
    """
    tree = ast.parse(pathlib.Path(classify_pass.__globals__["__file__"]).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        roots = {n.split(".")[0] for n in names}
        assert "simulator" not in roots, "module-level simulator import leaks truth"


def test_classify_takes_only_observables() -> None:
    """The classifier's inputs are an event and a team id -- no truth channel."""
    params = list(inspect.signature(classify_pass).parameters)
    assert params == ["event", "home_id"], params


# --------------------------------------------------------------------------- #
# 2. Every pass resolves to a real label
# --------------------------------------------------------------------------- #
def test_every_pass_gets_a_valid_label() -> None:
    sim_seed = 3
    from simulator import MatchSimulator

    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id="L", seed=sim_seed)
    sim.simulate(live=True)
    n = 0
    for event, truth in zip(sim.observable_stream(), sim.truth_series(), strict=True):
        if truth.pass_type_true is None:
            continue
        assert classify_pass(event, H) in tuple(PassType)
        n += 1
    assert n > 100, "expected a full match of passes"


# --------------------------------------------------------------------------- #
# 3. The score is real: beat the baseline, and recover every class
# --------------------------------------------------------------------------- #
def test_beats_majority_baseline_on_held_out_seeds() -> None:
    from pass_classifier import score_seeds

    report = score_seeds(EVAL_SEEDS)
    counts = {c: report._row(c) for c in tuple(PassType)}
    baseline = max(counts.values()) / report.total

    assert report.accuracy() > 0.85, f"accuracy {report.accuracy():.3f} too low"
    assert report.accuracy() > baseline + 0.15, (
        f"accuracy {report.accuracy():.3f} barely beats baseline {baseline:.3f}"
    )

    # A majority-only classifier would score the baseline; require real per-class
    # recovery so that cannot pass. LONG is exactly recoverable; the genuinely
    # ambiguous classes must still clear a floor.
    assert report.f1(PassType.LONG) > 0.95
    assert report.f1(PassType.THROUGH_BALL) > 0.70
    assert report.f1(PassType.CROSS) > 0.60
    assert report.f1(PassType.SHORT) > 0.85


# --------------------------------------------------------------------------- #
# 4. "Long" is forward travel, not distance on the ground
# --------------------------------------------------------------------------- #
def test_long_is_forward_travel_not_euclidean_distance() -> None:
    # A ~34 m diagonal that only advances 20 units: big on the grass, but not a
    # penetrating long ball. The old distance rule mislabelled this as LONG.
    dx = pass_distance_m(30.0, 50.0, 50.0, 90.0)
    assert dx > 22.0
    wide_diagonal = make_pass(H, 30.0, 50.0, 50.0, 90.0)
    assert classify_pass(wide_diagonal, H) is not PassType.LONG

    # Same start, driven straight down the pitch: a long ball.
    driven = make_pass(H, 30.0, 50.0, 55.0, 50.0)
    assert classify_pass(driven, H) is PassType.LONG


# --------------------------------------------------------------------------- #
# 5. The frame conversion is symmetric across the two teams
# --------------------------------------------------------------------------- #
def test_both_teams_are_read_in_their_own_attacking_frame() -> None:
    """The same attacking-local geometry must classify identically for either team.

    Home attacks +x, away attacks -x, so an away pass is the mirror image of the
    equivalent home pass. Reading each in its own frame is the only reason one
    rule can classify both -- if the flip were wrong, away passes would be read
    backwards and every label would be corrupted.
    """
    geometries = [
        (20.0, 10.0, 40.0, 12.0),   # short, from wide
        (55.0, 50.0, 70.0, 52.0),   # through ball into the final third
        (75.0, 10.0, 88.0, 45.0),   # cross from the left
        (30.0, 50.0, 55.0, 50.0),   # long ball
    ]
    for lx, ly, tx, ty in geometries:
        home = classify_pass(make_pass(H, lx, ly, tx, ty), H)
        # away's pass is the point reflection through the pitch centre
        away = classify_pass(make_pass(A, 100 - lx, 100 - ly, 100 - tx, 100 - ty), H)
        assert home is away, (lx, ly, tx, ty, home, away)


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
    print(f"\n{len(tests) - failures}/{len(tests)} classifier tests passed")
    raise SystemExit(1 if failures else 0)
