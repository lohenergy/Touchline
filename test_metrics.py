"""Invariant tests for the Interpret stage (`metrics.py`).

WHY THESE TESTS EXIST
---------------------
`test_events.py` proves the simulator's stream is *internally* consistent.
These tests prove the opposite direction: that `metrics.py` can rebuild the
match from the observable stream alone, and agree with the simulator's own
counters -- which the simulator computed from its full internal state, not from
the stream.

That is a real check, not a tautology. If `metrics.py` is missing an event type,
double-counting a shot, or mis-attributing possession time, the aggregate will
drift away from the simulator's counters and these tests will catch it.

The possession test is the important one. It asserts the property the design
rests on: the time decomposition accounts for *every second* of the clock, with
nothing silently dropped.

Run directly:      ./.venv/bin/python test_metrics.py
Or under pytest:   ./.venv/bin/python -m pytest test_metrics.py
"""

from __future__ import annotations

from schemas import MatchState, MatchSummary, Outcome
from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

from metrics import MatchMetrics

N_MATCHES = 20
H, A = DEMO_HOME.team_id, DEMO_AWAY.team_id


def interpret(seed: int) -> tuple[MatchSimulator, MatchMetrics]:
    """One match, plus the metrics rebuilt purely from its observable events."""
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
    sim.simulate(live=True)
    metrics = MatchMetrics(sim.match_id, H, A)
    metrics.extend(list(sim.observable_stream()))
    return sim, metrics


# --------------------------------------------------------------------------- #
# 1. Reconstruction: the summary must equal the simulator's own counters
# --------------------------------------------------------------------------- #
def test_objective_stats_match_simulator_counters() -> None:
    """Every counted stat rebuilt from events equals the simulator's counter.

    The simulator increments these counters as it *decides* events; `metrics.py`
    only ever sees the events themselves. Agreement means the observable stream
    carries enough information to reconstruct the match's objective record.
    """
    for seed in range(N_MATCHES):
        sim, metrics = interpret(seed)
        s = metrics.summary()

        pairs = {
            "score": (s.score, sim.score),
            "shots": (s.shots, sim.shots),
            "shots_on_target": (s.shots_on_target, sim.shots_on_target),
            "shots_in_box": (s.shots_in_box, sim.shots_in_box),
            "woodwork": (s.woodwork, sim.woodwork),
            "saves": (s.saves, sim.saves),
            "blocks": (s.blocks, sim.blocks),
            "corners": (s.corners, sim.corners),
            "offsides": (s.offsides, sim.offsides),
            "fouls": (s.fouls, sim.fouls),
            "yellows": (s.yellows, sim.yellows),
            "reds": (s.reds, sim.reds),
            "substitutions": (s.substitutions, sim.substitutions),
        }
        for name, (rebuilt, truth) in pairs.items():
            assert rebuilt == truth, (
                f"seed {seed}: {name} {rebuilt} != simulator {truth}"
            )

        # xG is summed from rounded event values, so allow a hair of drift.
        assert abs(s.xg[H] - sim.shot_quality[H]) < 0.01, (seed, "home xG")
        assert abs(s.xg[A] - sim.shot_quality[A]) < 0.01, (seed, "away xG")


# --------------------------------------------------------------------------- #
# 2. The possession-time invariant (the design's foundation)
# --------------------------------------------------------------------------- #
def test_possession_time_sums_to_the_clock() -> None:
    """Controlled + contested time equals the elapsed clock, exactly.

    Every second between events lands in home, away, or contested. If this ever
    fails, a possession percentage is being reported against a denominator that
    does not exist -- the kind of error that looks fine until someone adds it up.
    """
    for seed in range(N_MATCHES):
        _, metrics = interpret(seed)
        accounted = metrics.controlled_s[H] + metrics.controlled_s[A] + metrics.contested_s
        assert abs(accounted - metrics.clock_s) < 1e-6, (
            f"seed {seed}: accounted {accounted}s != clock {metrics.clock_s}s"
        )


def test_possession_shares_are_normalised() -> None:
    """Shares that are presented as shares must sum to 1."""
    for seed in range(N_MATCHES):
        _, metrics = interpret(seed)
        state = metrics.state()
        assert abs(sum(state.possession.values()) - 1.0) < 1e-9
        assert abs(sum(state.control.values()) - 1.0) < 1e-9
        assert abs(sum(state.territory.values()) - 1.0) < 1e-9
        assert 0.0 <= state.contested_possession <= 1.0
        assert 0.0 <= state.chaos <= 1.0


# --------------------------------------------------------------------------- #
# 3. Pass aggregates are rebuilt independently
# --------------------------------------------------------------------------- #
def test_pass_completion_matches_the_stream() -> None:
    """Pass counts and completion are recomputed from the raw events."""
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        events = list(sim.observable_stream())

        metrics = MatchMetrics(sim.match_id, H, A)
        metrics.extend(events)
        s = metrics.summary()

        for team in (H, A):
            attempted = [e for e in events
                         if e.type.value == "pass" and e.team == team]
            complete = [e for e in attempted if e.outcome == Outcome.COMPLETE]
            assert s.passes[team] == len(attempted)
            expected = len(complete) / len(attempted) if attempted else 0.0
            assert abs(s.pass_completion[team] - expected) < 1e-9


# --------------------------------------------------------------------------- #
# 4. The leak-proofing invariant, restated for the interpreted layer
# --------------------------------------------------------------------------- #
def test_interpreted_contracts_carry_no_latent_truth() -> None:
    """Neither MatchState nor MatchSummary may name a latent field.

    The whole evaluation rests on the interpreted layer being *recoverable*, and
    recoverable only from observables. If a latent field ever appears here, the
    interpretation has been handed the answer.
    """
    forbidden = {
        "control_true", "chaos_true", "dominant_team", "phase",
        "pass_type_true", "big_chance_true", "chance_created_true",
        "error_true", "from_error_true", "big_chances", "errors",
    }
    for model in (MatchState, MatchSummary):
        leaked = forbidden & set(model.model_fields)
        assert not leaked, f"latent truth leaked into {model.__name__}: {leaked}"


# --------------------------------------------------------------------------- #
# 5. Added time is carried through from the markers
# --------------------------------------------------------------------------- #
def test_summary_reports_added_time_from_the_markers() -> None:
    """The full-time summary's added time equals the simulator's ledger."""
    for seed in range(N_MATCHES):
        sim, metrics = interpret(seed)
        s = metrics.summary()
        for period in (1, 2):
            assert s.added_time_s[period] == sim.stoppage_breakdown[period].total_s
            breakdown = s.stoppage[period]
            assert sum(breakdown.by_cause.values()) == breakdown.total_s


# --------------------------------------------------------------------------- #
# 6. Rolling state is stable enough to trust while the match is live
# --------------------------------------------------------------------------- #
def test_state_is_available_at_every_step() -> None:
    """Polling state() mid-stream must never raise and never leave its bounds."""
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id="Tstep", seed=7)
    sim.simulate(live=True)
    metrics = MatchMetrics("Tstep", H, A)

    for i, event in enumerate(sim.observable_stream()):
        metrics.update(event)
        if i % 97 == 0:
            state = metrics.state()
            assert 0.0 <= state.chaos <= 1.0
            assert state.tempo >= 0.0
            assert abs(sum(state.possession.values()) - 1.0) < 1e-9


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
    print(f"\n{len(tests) - failures}/{len(tests)} metric tests passed")
    raise SystemExit(1 if failures else 0)
