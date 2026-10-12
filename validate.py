"""Calibration harness: does the synthetic data look like real football?

WHY THIS FILE EXISTS
--------------------
The first run of the simulator produced a 6-7 scoreline with 73 shots. It was
"working" in the sense that it ran, and completely wrong in the sense that
mattered.

The temptation at that point is to adjust a constant, re-run, eyeball it, and
adjust again. That is how people spend three days tuning a model by vibes and
end up overfitting their own intuition. The professional response is to build a
measuring instrument first, and tune against it.

This harness generates many matches, aggregates the results, and compares them
against reference ranges for real football. Once it exists, every "is this
better?" question becomes a number instead of an opinion.

ON THE REFERENCE NUMBERS
------------------------
The ranges below are commonly cited aggregate figures for professional league
football. Before submission they must be sourced and cited properly -- an
unsourced number in a validation report is worse than no validation report,
because it looks rigorous while being unfalsifiable. Treat them as targets to
verify, not as truth handed down.

A NOTE ON WHAT THIS DOES AND DOES NOT PROVE
-------------------------------------------
Matching aggregate *marginals* (shots per match, completion %) is necessary but
not sufficient. Data can match every average and still be structurally absurd --
for example, if every shot were taken from exactly the same position. Sequence
structure, correlation and game-state effects are validated separately in
`validate_sequence.py` later. This file is the first of several layers.
"""

from __future__ import annotations

import statistics
import sys
from collections.abc import Callable

from schemas import EventType, Outcome
from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

# name -> (low, high, what it is)
REFERENCE: dict[str, tuple[float, float, str]] = {
    "goals_per_match": (2.4, 3.2, "both teams combined"),
    "shots_per_match": (20.0, 30.0, "both teams combined"),
    "shots_in_box_pct": (55.0, 75.0, "share of shots from inside the penalty area"),
    "woodwork_per_match": (0.5, 2.0, "both teams combined"),
    "saves_per_match": (4.0, 9.0, "both teams combined"),
    "blocks_per_match": (4.0, 12.0, "both teams combined"),
    "big_chances_per_match": (2.0, 6.0, "both teams combined (latent truth)"),
    "errors_per_match": (0.5, 4.0, "unforced giveaways, both teams (latent truth)"),
    "shots_on_target_per_match": (7.0, 11.0, "both teams combined"),
    "xg_per_match": (2.2, 3.4, "should track goals closely"),
    "xg_per_shot": (0.08, 0.14, "shot quality as a rate"),
    "passes_per_match": (800.0, 1200.0, "both teams combined"),
    "pass_completion_pct": (78.0, 88.0, "all passes"),
    "home_possession_pct": (42.0, 58.0, "home share of CONTROLLED time (time-based)"),
    "contested_possession_pct": (3.0, 18.0, "share of time the ball has no clear owner"),
    "loose_balls_per_match": (10.0, 90.0, "contested phases, both teams combined"),
    "added_time_first_half_min": (0.5, 7.0, "first-half stoppage, minutes"),
    "added_time_second_half_min": (1.0, 10.0, "second-half stoppage, minutes"),
    "added_time_total_min": (2.0, 16.0, "both halves, minutes"),
    "corners_per_match": (8.0, 12.0, "both teams combined"),
    "offsides_per_match": (1.5, 4.5, "both teams combined"),
    "fouls_per_match": (18.0, 28.0, "both teams combined"),
    "yellow_cards_per_match": (2.0, 6.0, "both teams combined"),
    "red_cards_per_match": (0.0, 0.5, "both teams combined, low by design"),
    "substitutions_per_match": (4.0, 10.0, "both teams combined"),
}


def measure_one(seed: int) -> dict[str, float]:
    """Run a single match and reduce it to headline numbers."""
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"M{seed:04d}", seed=seed)
    sim.simulate(live=True)

    events = list(sim.observable_stream())
    passes = [e for e in events if e.type == EventType.PASS]
    complete = [p for p in passes if p.outcome == Outcome.COMPLETE]

    h, a = DEMO_HOME.team_id, DEMO_AWAY.team_id
    home_passes = [p for p in passes if p.team == h]

    total_shots = sum(sim.shots.values())
    total_goals = sum(sim.score.values())
    total_xg = sum(sim.shot_quality.values())

    controlled = sim.controlled_s[h] + sim.controlled_s[a]
    elapsed = controlled + sim.contested_s
    loose_balls = [e for e in events if e.type == EventType.LOOSE_BALL]

    return {
        "goals_per_match": float(total_goals),
        "shots_per_match": float(total_shots),
        "shots_in_box_pct": 100.0
        * sum(1 for e in events if e.type == EventType.SHOT and e.in_box)
        / max(1, total_shots),
        "woodwork_per_match": float(sum(sim.woodwork.values())),
        "saves_per_match": float(sum(sim.saves.values())),
        "blocks_per_match": float(sum(sim.blocks.values())),
        "shots_on_target_per_match": float(sum(sim.shots_on_target.values())),
        "xg_per_match": total_xg,
        "xg_per_shot": total_xg / max(1, total_shots),
        "passes_per_match": float(len(passes)),
        "pass_completion_pct": 100.0 * len(complete) / max(1, len(passes)),
        "home_possession_pct": 100.0 * sim.controlled_s[h] / max(1.0, controlled),
        "contested_possession_pct": 100.0 * sim.contested_s / max(1.0, elapsed),
        "loose_balls_per_match": float(len(loose_balls)),
        "added_time_first_half_min": sim.added_time_s[1] / 60.0,
        "added_time_second_half_min": sim.added_time_s[2] / 60.0,
        "added_time_total_min": (sim.added_time_s[1] + sim.added_time_s[2]) / 60.0,
        "corners_per_match": float(sum(sim.corners.values())),
        "offsides_per_match": float(sum(sim.offsides.values())),
        "fouls_per_match": float(sum(sim.fouls.values())),
        "yellow_cards_per_match": float(sum(sim.yellows.values())),
        "red_cards_per_match": float(sum(sim.reds.values())),
        "substitutions_per_match": float(sum(sim.substitutions.values())),
        "big_chances_per_match": float(
            sum(1 for t in sim.truth_series() if t.big_chance_true)
        ),
        "errors_per_match": float(
            sum(1 for t in sim.truth_series() if t.error_true)
        ),
    }


def run(n_matches: int = 60) -> int:
    """Aggregate n_matches and report against the reference ranges.

    Returns a process exit code: 0 if every metric is in range, 1 otherwise.
    Exit codes matter -- this is what lets the harness run in CI, so a change
    that breaks realism fails the build instead of quietly shipping.
    """
    samples: dict[str, list[float]] = {k: [] for k in REFERENCE}
    for seed in range(n_matches):
        for k, v in measure_one(seed).items():
            samples[k].append(v)

    print(f"\nCalibration report over {n_matches} synthetic matches")
    print("=" * 78)
    print(f"{'metric':<28}{'mean':>9}{'sd':>8}{'target':>16}   {'':<6}")
    print("-" * 78)

    failures = 0
    for name, (lo, hi, note) in REFERENCE.items():
        vals = samples[name]
        mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals)
        ok = lo <= mean <= hi
        if not ok:
            failures += 1
        flag = "ok" if ok else "OUT OF RANGE"
        print(f"{name:<28}{mean:>9.2f}{sd:>8.2f}{f'{lo:.1f}-{hi:.1f}':>16}   {flag}")

    print("-" * 78)
    for name, (lo, hi, note) in REFERENCE.items():
        print(f"  {name:<28} {note}")

    if failures:
        print(f"\n{failures} metric(s) out of range -- tune simulator constants.\n")
    else:
        print("\nAll headline metrics within reference range.\n")
    return 1 if failures else 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    raise SystemExit(run(n))
