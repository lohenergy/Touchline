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

from schemas import CardType, EventType, MatchEvent, Outcome
from simulator import (
    BIG_CHANCE_NOISE,
    BIG_CHANCE_QUALITY,
    DEFENSIVE_THIRD_X,
    DEMO_AWAY,
    DEMO_HOME,
    MatchSimulator,
    in_box,
)

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
        "big_chance_true",
        "chance_created_true",
        "error_true",
        "from_error_true",
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


# --------------------------------------------------------------------------- #
# 7. The objective trio (Phase 0.6)
# --------------------------------------------------------------------------- #
def test_every_saved_shot_has_a_save_event() -> None:
    """A keeper save is an observable act by a named keeper, paired 1:1."""
    for events in all_matches():
        saved = [
            e for e in events
            if e.type == EventType.SHOT and e.outcome == Outcome.SAVED
        ]
        saves = [e for e in events if e.type == EventType.SAVE]
        assert len(saved) == len(saves), (
            f"{len(saved)} saved shots but {len(saves)} SAVE events"
        )
        prev_shot: MatchEvent | None = None
        for e in events:
            if e.type == EventType.SHOT and e.outcome == Outcome.SAVED:
                prev_shot = e
            elif e.type == EventType.SAVE:
                assert prev_shot is not None, "SAVE event with no saved shot before it"
                assert e.team != prev_shot.team, f"save credited to the shooter: {e}"


def test_every_blocked_shot_has_a_block_event() -> None:
    """A shot block is an observable act by a named defender, paired 1:1."""
    for events in all_matches():
        blocked = [
            e for e in events
            if e.type == EventType.SHOT and e.outcome == Outcome.BLOCKED
        ]
        blocks = [e for e in events if e.type == EventType.BLOCK]
        assert len(blocked) == len(blocks), (
            f"{len(blocked)} blocked shots but {len(blocks)} BLOCK events"
        )
        prev_shot: MatchEvent | None = None
        for e in events:
            if e.type == EventType.SHOT and e.outcome == Outcome.BLOCKED:
                prev_shot = e
            elif e.type == EventType.BLOCK:
                assert prev_shot is not None, "BLOCK event with no blocked shot before it"
                assert e.team != prev_shot.team, f"block credited to the shooter: {e}"


def test_woodwork_is_a_separate_outcome() -> None:
    """A strike on the frame is neither a goal nor a shot on target."""
    strikes = 0
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        events = list(sim.observable_stream())
        ww = [
            e for e in events
            if e.type == EventType.SHOT and e.outcome == Outcome.WOODWORK
        ]
        strikes += len(ww)
        assert sum(sim.woodwork.values()) == len(ww), (
            "woodwork counter disagrees with the event stream"
        )
        on_target = sum(sim.shots_on_target.values())
        assert on_target + len(ww) <= sum(sim.shots.values()), (
            "woodwork must not be counted as a shot on target"
        )
    assert strikes > 0, "no woodwork strikes observed across 40 matches"


def test_in_box_is_geometrically_consistent() -> None:
    """The observed in_box flag must match the shot's origin geometry.

    Recompute the check from the *absolute* coordinates using the same fixed
    frame the simulator uses (home attacks toward x=100, away toward x=0), so
    the test does not just re-read a field the simulator already set.
    """

    def local(team_id: str, x: float, y: float) -> tuple[float, float]:
        if team_id == DEMO_HOME.team_id:
            return x, y
        return 100.0 - x, 100.0 - y  # away's attacking-local frame is mirrored

    for events in all_matches():
        for e in events:
            if e.type != EventType.SHOT:
                continue
            lx, ly = local(e.team, e.x, e.y)
            assert e.in_box == in_box(lx, ly), (
                f"in_box={e.in_box} inconsistent with origin ({e.x}, {e.y})"
            )


# --------------------------------------------------------------------------- #
# 8. Ground truth stays latent: big chances and errors (Phase 0.6)
# --------------------------------------------------------------------------- #
def test_big_chance_true_only_on_shots_with_bounded_consistency() -> None:
    """The big-chance tier is latent, on shots only, and bounded by observable xG.

    The tier is decided on latent quality with uniform noise, so the bounds are
    *exact* and this is a hard invariant, not a statistical one: a TRUE tier
    requires observable shot_quality at least one noise-width below the
    threshold, and a FALSE one at most one noise-width above it.
    """
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        for event, latent in sim.snapshots:
            if event.type == EventType.SHOT:
                assert latent.big_chance_true is not None, event
                assert event.shot_quality is not None, event
                if latent.big_chance_true:
                    assert event.shot_quality >= BIG_CHANCE_QUALITY - BIG_CHANCE_NOISE, (
                        f"tiered big chance at xG {event.shot_quality:.3f}"
                    )
                else:
                    assert event.shot_quality <= BIG_CHANCE_QUALITY + BIG_CHANCE_NOISE, (
                        f"tiered non-chance at xG {event.shot_quality:.3f}"
                    )
            else:
                assert latent.big_chance_true is None, event


def test_creator_pass_always_yields_a_big_chance() -> None:
    """chance_created_true marks the pass that manufactured a big chance.

    A marked pass must be followed directly (only pressure events may intervene)
    by a big-chance shot from the same team. Some big chances come from
    set-piece restarts with no creator pass, so the creator count can be lower
    than the big-chance count -- never higher.
    """
    creators = 0
    big_chances = 0
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        snaps = sim.snapshots
        for i, (event, latent) in enumerate(snaps):
            if event.type == EventType.SHOT and latent.big_chance_true:
                big_chances += 1
            if not (event.type == EventType.PASS and latent.chance_created_true):
                continue
            creators += 1
            j = i + 1
            while j < len(snaps):
                ev, lat = snaps[j]
                if ev.type == EventType.PRESSURE:
                    j += 1
                    continue
                assert ev.type == EventType.SHOT, (
                    f"creator pass not followed directly by a shot (got {ev.type.value})"
                )
                assert ev.team == event.team, "creator pass credited the wrong team"
                assert lat.big_chance_true is True, (
                    f"creator pass did not yield a big chance ({ev.event_id})"
                )
                break
    assert creators <= big_chances, f"{creators} creators for {big_chances} big chances"
    assert big_chances > 0, "no big chances observed across 40 matches"


def test_error_flags_carried_only_on_the_right_events() -> None:
    """error_true marks failed defensive-third passes; from_error marks shots.

    The observable side of an error is an INCOMPLETE pass by the team in
    possession, deep in that team's own third. The gift's consequence -- a
    shot -- carries from_error_true. The error itself is never observable.
    """
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        for event, latent in sim.snapshots:
            if event.type == EventType.PASS and latent.error_true:
                assert event.outcome == Outcome.INCOMPLETE, event
                if event.team == DEMO_HOME.team_id:
                    assert event.x < DEFENSIVE_THIRD_X, event
                else:
                    assert event.x > 100.0 - DEFENSIVE_THIRD_X, event
            elif event.type == EventType.SHOT:
                assert latent.from_error_true is not None, event
            else:
                assert latent.error_true is None, event
                assert latent.from_error_true is None, event


def test_every_from_error_shot_follows_an_error() -> None:
    """A shot can only be error-gifted after an opposing deep giveaway.

    Walk back from each error-gifted shot: before any possession change away
    from the shot's team, a defensive-third error pass by the other team must
    appear -- closing the chain error -> giveaway -> chance.
    """
    error_shots = 0
    for seed in range(N_MATCHES):
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"T{seed:04d}", seed=seed)
        sim.simulate(live=True)
        snaps = sim.snapshots
        for i, (event, latent) in enumerate(snaps):
            if event.type != EventType.SHOT or not latent.from_error_true:
                continue
            error_shots += 1
            found_error = False
            j = i - 1
            while j >= 0:
                ev, lat = snaps[j]
                if ev.type == EventType.PASS and lat.error_true and ev.team != event.team:
                    found_error = True
                    break
                if ev.type in (EventType.OFFSIDE, EventType.POSSESSION_CHANGE):
                    if ev.team != event.team:
                        break  # possession had flipped away before the error
                j -= 1
            assert found_error, (
                f"error-gifted shot {event.event_id} has no preceding giveaway"
            )
    assert error_shots >= 1, "no error-gifted shots across 40 matches"


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
