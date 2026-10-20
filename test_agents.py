"""Tests for the agent layer (`agents.py`) and the way `server.py` drives it.

WHY THESE TESTS EXIST
---------------------
Every other suite checks that Touchline's *football* is right. These check that
its *team* is real -- because a team of named roles that is only true in prose
is worse than no team at all: it looks like orchestration and behaves like a
pile of calls.

Three claims are worth protecting, and each has tests below:

  1. The contracts bind. An agent cannot write a key it did not declare, and a
     view cannot contain a key its holder did not ask for. If that ever stops
     being mechanically true, "the desk is structurally blind to latent truth"
     becomes a comment again.
  2. The board has nowhere to put the truth. No declared key is a latent one --
     the blackboard cannot leak what it was never allowed to hold.
  3. Failure is survivable and visible. A renderer that throws costs one
     payload and stays in the trace as a failure; a dead model costs the
     commentary, not the match. Neither is allowed to vanish quietly.

Run directly:      ./.venv/bin/python test_agents.py
Or under pytest:   ./.venv/bin/python -m pytest test_agents.py
"""

from __future__ import annotations

import asyncio
import json

import server
from agents import (
    DESK,
    EXPLAIN,
    GENERATE,
    INTERPRET,
    RENDER,
    ROSTER,
    VERIFY,
    AgentKind,
    AgentStatus,
    Blackboard,
    ContractError,
    Trace,
)
from narrator import LocalNarrator
from verifier import LocalVerifier

# The keys that may never reach the analyst surface, and never the board.
LATENT_KEYS = frozenset(
    {
        "control_true",
        "chaos_true",
        "pass_type_true",
        "big_chance_true",
        "chance_created_true",
        "error_true",
        "from_error_true",
        "dominant_team",
    }
)

SEED = 81


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse(chunk: str) -> dict[str, str] | None:
    """One yielded chunk is one SSE frame: an `event:` line and a `data:` line."""
    name, data = None, None
    for line in chunk.split("\n"):
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            data = line[6:]
    return {"event": name, "data": data} if name is not None else None


def collect(seed: int = SEED) -> list[dict[str, str]]:
    """Run one whole match through the real stream and return its frames.

    The narrator singleton is pinned to the deterministic desk for the duration:
    these tests must be identical in a hotel room, in CI, and with a dead
    network -- so the one stage that can reach out to a model is not allowed to
    decide for itself here.
    """

    async def go() -> list[dict[str, str]]:
        out = []
        async for chunk in server._stream_match(seed):
            frame = parse(chunk)
            if frame is not None:
                out.append(frame)
        return out

    original_desk = server._NARRATOR
    original_critic = server._VERIFIER
    server._NARRATOR = LocalNarrator()
    server._VERIFIER = LocalVerifier()
    try:
        return asyncio.run(go())
    finally:
        server._NARRATOR = original_desk
        server._VERIFIER = original_critic


def terminal_trace(frames: list[dict[str, str]]) -> list[dict]:
    """The agent trace as it appears on the wire, in the `done` frame."""
    done = [f for f in frames if f["event"] == "done"]
    assert len(done) == 1, "every match must terminate with exactly one `done`"
    return json.loads(done[0]["data"])["trace"]


def expect_contract_error(fn, *args) -> None:
    try:
        fn(*args)
    except ContractError:
        return
    raise AssertionError(f"{fn.__name__}{args} should have refused the contract")


def filled_board() -> Blackboard:
    """A board every agent has already committed its declared products to."""
    board = Blackboard()
    for spec in ROSTER:
        for key in spec.writes:
            board.put(spec, key, f"{spec.name}->{key}")
    return board


# --------------------------------------------------------------------------- #
# 1. The contracts bind
# --------------------------------------------------------------------------- #
def test_the_roster_names_the_pipeline_in_handoff_order() -> None:
    assert [s.name for s in ROSTER] == [
        "generate",
        "interpret",
        "explain",
        "render",
        "desk",
        "verify",
    ]


def test_every_declared_read_has_a_producer() -> None:
    """An agent may only ask for something a previous agent in hand."""
    produced = {key for spec in ROSTER for key in spec.writes}
    for spec in ROSTER:
        for key in spec.reads:
            assert key in produced, f"{spec.name} reads {key!r}, which nobody writes"


def test_everything_any_agent_may_reach_is_declared_here() -> None:
    """The whole vocabulary of the blackboard, in one line.

    Deliberately a fixed set: adding a key means editing this test, and that is
    the point -- shared state should grow under review, not by accident.
    """
    touched = {key for spec in ROSTER for key in spec.reads + spec.writes}
    assert touched == {
        "match", "state", "summary", "insight", "overlay", "draft", "narrative", "verdict",
    }


def test_no_declared_key_is_a_latent_one() -> None:
    """The board cannot leak the truth it was never allowed to hold."""
    touched = {key for spec in ROSTER for key in spec.reads + spec.writes}
    assert not (touched & set(LATENT_KEYS))


def test_an_agent_cannot_write_a_key_it_did_not_declare() -> None:
    board = Blackboard()
    for spec in ROSTER:
        for key in ("big_chance_true", "narrative", "something_new"):
            if key in spec.writes:
                continue
            expect_contract_error(board.put, spec, key, "value")


def test_a_view_hands_out_only_what_its_holder_declared_it_reads() -> None:
    board = filled_board()
    for spec in ROSTER:
        view = board.view(spec)
        assert set(view) <= set(spec.reads), f"{spec.name} was shown extra state"
        assert set(view) == set(spec.reads), f"{spec.name} was shown its own input"


def test_the_desk_is_handed_nothing_but_a_summary_and_its_citations() -> None:
    """The one view that matters: the language model's.

    `state` and `overlay` exist on this board when the desk speaks, and are not
    in it -- nor is `narrative`, `match` aside. There is no key called anything
    like the truth to reach for even if it tried.
    """
    view = filled_board().view(DESK)
    assert set(view) == {"match", "summary", "insight"}
    assert "state" not in view
    assert "overlay" not in view
    # The critic is handed the draft and never the finished narrative: it reads
    # what the desk wrote, not a copy passed alongside it.
    assert set(filled_board().view(VERIFY)) == {"draft", "summary", "insight"}


def test_a_declared_read_that_has_not_been_produced_yet_is_absent_not_faked() -> None:
    """A goalless, chanceless match still has to reach the desk.

    `insight` is a history key, so zero insights must read as "empty", not as a
    contract violation -- otherwise a 0-0 would fail to get commentary at all.
    """
    board = Blackboard()
    board.put(GENERATE, "match", {"match_id": "M"})
    assert set(board.view(DESK)) == {"match"}
    assert board.view(DESK).get("insight", []) == []


def test_history_is_never_overwritten_by_the_next_product() -> None:
    """The desk at full time must be able to see that half-time happened."""
    board = Blackboard()
    board.put(INTERPRET, "summary", "half time")
    board.put(INTERPRET, "summary", "full time")
    assert board.history("summary") == ["half time", "full time"]
    assert board.latest("summary") == "full time"
    assert board.latest("never_put", default="fallback") == "fallback"


def test_the_interpreter_runs_inside_the_explainer() -> None:
    """The nesting is part of the contract, not folklore in a comment."""
    assert INTERPRET.inside == "explain"
    assert all(s.inside is None for s in ROSTER if s is not INTERPRET)


# --------------------------------------------------------------------------- #
# 2. The trace tells the truth about the run
# --------------------------------------------------------------------------- #
def test_runs_come_back_in_handoff_order_regardless_of_when_they_finished() -> None:
    trace = Trace()
    trace.tick(DESK)
    trace.tick(GENERATE)
    trace.tick(RENDER)
    assert [r.agent for r in trace.runs()] == ["generate", "render", "desk"]


def test_the_same_note_is_recorded_once_not_twice() -> None:
    trace = Trace()
    with trace.session(DESK) as run:
        run.note("provider=azure model=gpt-4.1-mini")
        run.note("provider=azure model=gpt-4.1-mini")
    assert run.detail == "provider=azure model=gpt-4.1-mini"


def test_different_notes_accumulate_in_order() -> None:
    trace = Trace()
    with trace.session(DESK) as run:
        run.note("ht")
        run.note("ft")
    assert run.detail == "ht · ft"


def test_failed_beats_degraded_beats_ok() -> None:
    degraded = Trace()
    with degraded.session(DESK) as run:
        run.degrade("fell back")
        run.degrade("fell back")
    assert run.status == AgentStatus.DEGRADED
    assert run.error is None

    failed = Trace()
    try:
        with failed.session(DESK):
            raise RuntimeError("the model is gone")
    except RuntimeError:
        pass  # a session reports a failure; it does not swallow it
    assert failed.runs()[0].status == AgentStatus.FAILED
    assert "RuntimeError" in (failed.runs()[0].error or "")


def test_a_soft_failure_returns_none_and_keeps_the_error() -> None:
    trace = Trace()

    def explodes() -> str:
        raise ValueError("no")

    assert trace.run(RENDER, explodes, soft=True) is None
    run = trace.runs()[0]
    assert run.agent == "render"
    assert run.status == AgentStatus.FAILED
    assert run.calls == 0
    assert run.error == "ValueError: no"


def test_a_hard_failure_is_recorded_and_reraised_not_swallowed() -> None:
    trace = Trace()

    def explodes() -> str:
        raise ValueError("no")

    try:
        trace.run(EXPLAIN, explodes)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-soft failure must reach the caller")
    assert trace.runs()[0].status == AgentStatus.FAILED


def test_a_successful_soft_call_is_counted_as_a_call() -> None:
    trace = Trace()
    assert trace.run(RENDER, lambda: "payload", soft=True) == "payload"
    run = trace.runs()[0]
    assert run.status == AgentStatus.OK
    assert run.calls == 1


# --------------------------------------------------------------------------- #
# 3. A real match, on the wire
# --------------------------------------------------------------------------- #
def test_a_real_match_reports_one_run_per_agent() -> None:
    frames = collect()
    trace = terminal_trace(frames)
    assert [r["agent"] for r in trace] == [
        "generate", "interpret", "explain", "render", "desk", "verify",
    ]
    for run in trace:
        assert run["status"] == "ok", f"{run['agent']} failed: {run['error']}"
        assert run["role"] and run["kind"] in {k.value for k in AgentKind}
        assert run["calls"] > 0, f"{run['agent']} never ran"


def test_every_row_on_the_wire_can_stand_on_its_own() -> None:
    """A renderer must not need the roster handed to it as well."""
    for run in terminal_trace(collect()):
        assert set(run) == {
            "agent", "role", "kind", "status",
            "calls", "ms", "inside", "detail", "error",
        }


def test_a_hosted_agent_reports_no_timing_it_does_not_have() -> None:
    """Interpret's ms lives inside Explain's; guessing a split would be worse.

    It must show `null`, not `0` -- `0` would claim it was measured and found
    instant.
    """
    trace = {r["agent"]: r for r in terminal_trace(collect())}
    assert trace["interpret"]["ms"] is None
    assert trace["explain"]["ms"] is not None and trace["explain"]["ms"] >= 0
    assert trace["explain"]["calls"] == trace["interpret"]["calls"]


def test_a_dead_renderer_costs_a_payload_not_the_match() -> None:
    """Recovery you can see: the failure lands in the trace, not in silence."""
    original = server.payload_for

    def explode(_insight):
        raise RuntimeError("renderer exploded")

    server.payload_for = explode
    try:
        frames = collect()
    finally:
        server.payload_for = original

    trace = {r["agent"]: r for r in terminal_trace(frames)}
    assert [f["event"] for f in frames if f["event"] == "done"], "the match still ended"
    assert not any(f["event"] == "overlay" for f in frames)
    assert trace["render"]["status"] == "failed"
    assert "renderer exploded" in (trace["render"]["error"] or "")
    # Everything downstream carried on untouched.
    for agent in ("generate", "interpret", "explain", "desk", "verify"):
        assert trace[agent]["status"] == "ok", f"{agent} was collateral damage"


class _DeadDesk:
    """Stands in for an unreachable model."""

    provider = "azure"
    model = "gpt-narrator"

    async def narrate(self, ctx):  # noqa: ANN001 -- the payload is not our business
        raise RuntimeError("endpoint refused the connection")


def test_a_dead_model_costs_the_commentary_not_the_match() -> None:
    original = server._narrator
    server._narrator = lambda: _DeadDesk()
    try:
        frames = collect()
    finally:
        server._narrator = original

    kinds = [f["event"] for f in frames]
    assert "narrative" not in kinds, "no half-built commentary may reach the viewer"
    assert kinds[-1] == "done"
    assert "summary" in kinds, "the numbers must arrive regardless"

    trace = {r["agent"]: r for r in terminal_trace(frames)}
    assert trace["desk"]["status"] == "failed"
    assert trace["desk"]["calls"] == 0
    assert "endpoint refused" in (trace["desk"]["error"] or "")


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
    print(f"\n{len(tests) - failures}/{len(tests)} agent tests passed")
    raise SystemExit(1 if failures else 0)
