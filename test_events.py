"""Invariant tests for the event stream.

WHY THESE TESTS EXIST
---------------------
`validate.py` answers "do the aggregate numbers look like football?" That is a
*statistical* check. It cannot catch a bug that produces statistically normal
numbers while getting the semantics exactly backwards -- which is precisely what
the original foul handling did: it produced a plausible number of fouls per match
while attributing every one of them to the wrong team and handing over possession
the wrong way.

These tests check *invariants*: properties that must hold for every event, in
every match, regardless of the averages. Statistics and invariants are
complementary and you want both. A model can have the right mean and the wrong
logic.

Run directly:      ./.venv/bin/python test_events.py
Or under pytest:   ./.venv/bin/python -m pytest test_events.py
"""

from __future__ import annotations

from collections import Counter

from schemas import CardType, EventType, MatchEvent
from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

# A pass or shot tells us who currently has the ball.
POSSESSION_MARKERS = {EventType.PASS, EventType.SHOT}

N_MATCHES = 40


def collect(seed: int) -> list[MatchEvent]:
    """One match, reduced to its observable events (no latent state)."""
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
    sim.simulate(live=True)
    return list(sim.observable_stream())


def all_matches() -> list[list[MatchEvent]]:
    return [collect(seed) for seed in range(N_MATCHES)]


# --------------------------------------------------------------------------- #
# 1. The vocabulary contract
# --------------------------------------------------------------------------- #
def test_every_declared_event_type_is_emitted() -> None:
    """Every member of EventType must actually appear in the stream.

    A declared-but-never-emitted type is a silent contract violation: a
    downstream consumer writes code for it, and that code never runs -- until it
    does, in a way nobody tested. (Both SUBSTITUTION and CARD were dead in an
    early draft.)
    """
    seen: Counter[EventType] = Counter()
    for events in all_matches():
        for e in events:
            seen[e.type] += 1

    missing = [t.value for t in EventType if seen[t] == 0]
    assert not missing, f"event types declared but never emitted: {missing}"


def test_card_type_only_on_cards() -> None:
    """card_type is set on CARD events and nowhere else."""
    for events in all_matches():
        for e in events:
            if e.type == EventType.CARD:
                assert e.card_type in (CardType.YELLOW, CardType.RED), e
            else:
                assert e.card_type is None, e


# --------------------------------------------------------------------------- #
# 2. Football semantics (the bug this file was written to trap)
# --------------------------------------------------------------------------- #
def _possession_before(events: list[MatchEvent], i: int) -> str | None:
    for j in range(i - 1, -1, -1):
        if events[j].type in POSSESSION_MARKERS:
            return events[j].team
    return None


def _possession_after(events: list[MatchEvent], i: int) -> str | None:
    for j in range(i + 1, len(events)):
        if events[j].type in POSSESSION_MARKERS:
            return events[j].team
    return None


def test_foul_is_committed_by_the_defending_team() -> None:
    """The team that fouls is the team WITHOUT the ball."""
    examined = 0
    for events in all_matches():
        for i, e in enumerate(events):
            if e.type != EventType.FOUL:
                continue
            poss = _possession_before(events, i)
            if poss is None:
                continue
            examined += 1
            assert poss != e.team, (
                f"foul attributed to the team in possession at event {e.event_id}"
            )
    assert examined > 100, "not enough fouls sampled to be meaningful"


def test_possession_is_retained_after_a_foul() -> None:
    """The fouled side keeps the ball via the free kick.

    This is the half of the bug that is easiest to miss: it is not enough to
    name the right team, the possession must not change either. Fouls that
    change possession inflate the turnover count and quietly corrupt every
    downstream possession metric.
    """
    retained = 0
    checked = 0
    for events in all_matches():
        for i, e in enumerate(events):
            if e.type != EventType.FOUL:
                continue
            poss = _possession_before(events, i)
            nxt = _possession_after(events, i)
            if poss is None or nxt is None:
                continue
            checked += 1
            if nxt == poss:
                retained += 1

    assert checked > 100, "not enough fouls sampled to be meaningful"
    # A handful of edge cases (a foul in the final second of a half) are fine.
    assert retained / checked > 0.98, f"possession retained only {retained}/{checked}"


def test_possession_is_retained_after_a_corner() -> None:
    """A corner is restarted by the attacking side, so it does not turn over."""
    retained = 0
    checked = 0
    for events in all_matches():
        for i, e in enumerate(events):
            if e.type != EventType.CORNER:
                continue
            nxt = _possession_after(events, i)
            if nxt is None:
                continue
            checked += 1
            if nxt == e.team:
                retained += 1
    assert checked > 20, "not enough corners sampled"
    assert retained / checked > 0.9, f"corner possession retained {retained}/{checked}"


# --------------------------------------------------------------------------- #
# 3. The leak-proofing invariant
# --------------------------------------------------------------------------- #
def test_observable_events_carry_no_latent_state() -> None:
    """Latent truth must never ride along inside an observable event.

    This is the invariant the whole evaluation rests on. If control_true or
    chaos_true ever appear on a MatchEvent, the "prediction" task becomes
    meaningless -- the model would be reading the answer off the input. The
    same holds for pass_type_true: it is the simulator's *intent*, which a
    real analyst can only guess at from the trajectory.
    """
    forbidden = {
        "control_true",
        "chaos_true",
        "control",
        "chaos",
        "dominant_team",
        "pass_type_true",
    }
    forbidden &= set(MatchEvent.model_fields)
    assert not forbidden, f"latent fields leaked into MatchEvent: {forbidden}"


def test_substitution_replaces_a_different_player() -> None:
    """A sub brings one player on and takes a different one off."""
    seen = 0
    for events in all_matches():
        for e in events:
            if e.type != EventType.SUBSTITUTION:
                continue
            seen += 1
            assert e.player_out_id is not None, e
            assert e.player_out_id != e.player_id, e
    assert seen > 20, "not enough substitutions sampled"


# --------------------------------------------------------------------------- #
# 4. Ground truth stays latent (pass intent)
# --------------------------------------------------------------------------- #
def test_pass_type_true_only_on_pass_snapshots() -> None:
    """Pass intent is ground truth: present on PASS latents, nowhere else."""
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        for event, latent in sim.snapshots:
            if event.type == EventType.PASS:
                assert latent.pass_type_true is not None, event
            else:
                assert latent.pass_type_true is None, event


# --------------------------------------------------------------------------- #
# 5. The clock and added time
# --------------------------------------------------------------------------- #
def test_display_clock_respects_half_caps() -> None:
    """minute is a *display* minute: capped at 45 / 90, never above.

    If minute ever exceeds 45 in the first half or 90 in the second, some
    downstream broadcast renderer will draw "98'" on screen -- a cosmetic bug
    that also means the simulator's time model is lying about how football
    clocks work. And 45+X may only appear once the clock has actually left
    regulation.
    """
    for events in all_matches():
        for e in events:
            if e.period == 1:
                assert 0 <= e.minute <= 45, e
                if e.stoppage_minute > 0:
                    assert e.minute == 45, e
            else:
                assert 45 <= e.minute <= 90, e
                if e.stoppage_minute > 0:
                    assert e.minute == 90, e


def test_stoppage_breakdown_sums_are_self_consistent() -> None:
    """The added-time number must equal the sum of its stated causes.

    The whole point of the ledger is that the figure and its explanation are
    the same object; if they disagree, the explanation is decoration.
    """
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        for period in (1, 2):
            breakdown = sim.stoppage_breakdown[period]
            assert breakdown is not None, f"half {period} missing stoppage"
            assert breakdown.total_s > 0, f"half {period}: zero added time"
            assert sum(breakdown.by_cause.values()) == breakdown.total_s, breakdown
            assert breakdown.total_s <= 600, breakdown  # clamp upper bound


def test_there_is_stoppage_in_both_halves() -> None:
    """Both halves must run past 45:00 / 90:00, because every match stops."""
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        assert sim.added_time_s[1] > 0
        assert sim.added_time_s[2] > 0


# --------------------------------------------------------------------------- #
# 6. Possession is a time decomposition
# --------------------------------------------------------------------------- #
def test_possession_time_sums_to_elapsed_clock() -> None:
    """Every second of the match belongs to home, away, or the contested phase.

    This decomposition is useless -- worse, deceptive -- if it silently drops
    time. The three buckets must account for the whole clock, every match.
    """
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        total = sim.controlled_s[sim.home.team_id] + sim.controlled_s[sim.away.team_id]
        accounted = total + sim.contested_s
        assert abs(accounted - sim.clock_s) < 1.0, (
            f"possession time {accounted:.1f}s != match clock {sim.clock_s}s"
        )


def test_loose_ball_count_matches_contest_counter() -> None:
    """Each contest emits between one and two LOOSE_BALL events.

    The observable count brackets the ground-truth counter exactly -- the
    observable version of the contested phase recovers the latent one.
    """
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        loose = sum(1 for e in sim.observable_stream() if e.type == EventType.LOOSE_BALL)
        if sim.contests:
            assert sim.contests <= loose <= 2 * sim.contests, (
                f"contests={sim.contests} loose_balls={loose}"
            )


if __name__ == "__main__":
    # Plain-runner so the file needs no test dependency. pytest also finds these.
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
    print(f"\n{len(tests) - failures}/{len(tests)} invariant tests passed")
    raise SystemExit(1 if failures else 0)
