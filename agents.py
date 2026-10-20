"""Touchline's agent layer: named roles, declared contracts, one shared board.

WHY THIS FILE EXISTS
--------------------
The stages upstream of the language model already behaved like a team: one
component simulates, one interprets, one explains, one renders, and one talks in
prose. What they did not do was *say so*. Nothing recorded who ran, on what
input, with what output, or what happened when one of them broke -- so a reader
could not tell a pipeline from a pile of calls.

This module makes the team explicit, and then holds it to it:

    AgentSpec    what an agent is called, what it is for, and -- in `reads` and
                 `writes` -- exactly which pieces of shared state it may touch.
    Blackboard   the one place agents meet. It is *not* a free-for-all: products
                 only enter it under the writing agent's contract, and a view
                 only leaves it under the reading agent's contract.
    Trace        who ran, how many times, how long, and whether it succeeded,
                 degraded or failed -- in handoff order, ready to be shown.

THE COURIER PRINCIPLE
---------------------
Agents never touch the board directly. `server.py` acts as the courier: it
commits an agent's output under that agent's declared `writes`, and it hands an
agent (or, on the agent's behalf, the function that serves it) a view holding
only its declared `reads`. Both directions are therefore checked, without ever
giving one agent a reference to the whole match.

This is what makes the sandbox worth having. The desk agent is handed a view of
`match`, `summary` and `insight` -- and nothing else exists for it to reach for.
Latent ground truth is not "hidden from" the desk; it is structurally absent from
the only object it is given.

WHAT IS NOT HERE
----------------
The gates that audit the desk's claims live in `narrator.py`, because they audit
`narrator.py`'s own output. They are part of the desk's contract (its role string
says so), not a separate agent -- claiming otherwise would be inflating the
roster.

What *is* here is the one agent that earns its own row: `verify`, the critic. It
qualifies because it is not a continuation of the desk. It does not write, it
does not see the conversation that produced the draft, and it cannot introduce a
fact -- it can only rule on claims the desk already made, and its rulings are
recorded where a reader can see them.

Run the checks:
    ./.venv/bin/python test_agents.py
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class AgentKind(StrEnum):
    """How an agent decides.

    The distinction is not decoration: only a LANGUAGE_MODEL agent is allowed to
    produce prose a human might take at face value, and only it may fail softly
    into a fallback. Everything else is a rule, and a rule that throws is a bug.
    """

    DETERMINISTIC = "deterministic"
    LANGUAGE_MODEL = "language_model"


class AgentStatus(StrEnum):
    """The outcome of an agent's work during one match.

    DEGRADED is the interesting one: the agent produced output, but not by its
    primary path -- a language model that fell back to the deterministic desk.
    The output is still good, and a viewer is still owed the fact that it came
    from somewhere else, which is why it is recorded rather than folded into OK.
    """

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


_SEVERITY = {AgentStatus.OK: 0, AgentStatus.DEGRADED: 1, AgentStatus.FAILED: 2}


class ContractError(RuntimeError):
    """An agent touched shared state its contract does not allow.

    Raised, never logged: a contract that is only *probably* enforced is a
    comment.
    """


# --------------------------------------------------------------------------- #
# The contracts
# --------------------------------------------------------------------------- #
class AgentSpec(BaseModel):
    """One agent, declared.

    `reads` and `writes` name keys on the `Blackboard`. They are deliberately
    strings rather than types -- the point is that the *set* of what an agent may
    reach for is small, written down, and checkable, not that the board is
    type-checked.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Stable id. Appears in the trace.")
    role: str = Field(
        description="One sentence, human: what this agent is for. Long enough "
        "to stand on its own in a README or an on-screen trace."
    )
    kind: AgentKind
    reads: tuple[str, ...] = Field(
        default=(), description="Board keys this agent may be handed."
    )
    writes: tuple[str, ...] = Field(
        default=(), description="Board keys this agent is permitted to produce."
    )
    inside: str | None = Field(
        default=None,
        description="The agent that runs this one internally, if any.",
    )


class AgentRun(BaseModel):
    """What actually happened to one agent during one match.

    Self-contained on purpose: `role` and `kind` travel with every row so a
    renderer can show the whole picture without being handed the roster too.
    """

    agent: str
    role: str
    kind: AgentKind
    status: AgentStatus = AgentStatus.OK
    calls: int = 0
    ms: float | None = Field(
        default=None,
        description="Wall time measured for this agent, or None where the work "
        "is not separable from the agent that hosts it.",
    )
    inside: str | None = None
    detail: str | None = None
    error: str | None = None

    _notes: set[str] = PrivateAttr(default_factory=set)

    # -- mutation, so a caller holding a run can report on itself ---------- #
    def tick(self, n: int = 1) -> None:
        """Account for `n` calls made without going through `Trace.run`."""
        self.calls += n

    def note(self, detail: str) -> None:
        """Record a fact about this run. The same fact twice is recorded once."""
        if not detail or detail in self._notes:
            return
        self._notes.add(detail)
        self.detail = f"{self.detail} · {detail}" if self.detail else detail

    def degrade(self, detail: str) -> None:
        """Mark the run as having produced output by a fallback path."""
        self.note(detail)
        if _SEVERITY[AgentStatus.DEGRADED] > _SEVERITY[self.status]:
            self.status = AgentStatus.DEGRADED

    def fail(self, exc: BaseException) -> None:
        """Mark the run as having produced nothing."""
        self.status = AgentStatus.FAILED
        self.error = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# The roster -- in handoff order
# --------------------------------------------------------------------------- #
GENERATE = AgentSpec(
    name="generate",
    role=(
        "Simulates one match from a seed and yields its observable events. "
        "It is the only agent that ever sees the ground truth, and it is not "
        "allowed to say a word of it: everything downstream is handed the "
        "observable stream and nothing more."
    ),
    kind=AgentKind.DETERMINISTIC,
    writes=("match",),
)

INTERPRET = AgentSpec(
    name="interpret",
    role=(
        "Turns observables into the running MatchState and MatchSummary -- "
        "possession, shots, expected goals, set pieces, added time. It measures "
        "what happened and cannot know why, which is exactly what makes its "
        "numbers usable as evidence by every other agent."
    ),
    kind=AgentKind.DETERMINISTIC,
    reads=("match",),
    writes=("state", "summary"),
    inside="explain",
)

EXPLAIN = AgentSpec(
    name="explain",
    role=(
        "Detects and attributes key moments, and writes each one down with the "
        "event ids that make the explanation true. It owns Interpret and advances "
        "it once per event before interpreting, so attribution always uses the "
        "match as it stood entering the event."
    ),
    kind=AgentKind.DETERMINISTIC,
    reads=("match",),
    writes=("insight",),
)

RENDER = AgentSpec(
    name="render",
    role=(
        "Turns one Insight into a timed overlay payload for the broadcast "
        "graphics: the key moment, its citations, and how long it should stay on "
        "screen. It renders the verdict it is given and does not second-guess it."
    ),
    kind=AgentKind.DETERMINISTIC,
    reads=("insight",),
    writes=("overlay",),
)

DESK = AgentSpec(
    name="desk",
    role=(
        "Three pundits reading the same cited summary. The only agent allowed "
        "to speak in prose, and the only one with a fallback: its claims pass "
        "four deterministic gates before they leave it, and when the model is "
        "gone the local desk speaks rather than the panel going silent. It writes "
        "the narrative twice over -- `draft` as it first stands, `narrative` once "
        "the critic has ruled -- because a reader is owed both."
    ),
    kind=AgentKind.LANGUAGE_MODEL,
    reads=("match", "summary", "insight"),
    writes=("draft", "narrative"),
)

VERIFY = AgentSpec(
    name="verify",
    role=(
        "The critic. Reads the desk's draft and, for each claim, the evidence "
        "behind the events that claim cites -- and rules it supported, "
        "misattributed or unsupported. It never writes commentary, never suggests "
        "a replacement and never sees the conversation that produced the draft, "
        "so a second call can never become a way of smuggling prose past the "
        "gates. Every ruling is attached to the claim it judges, and a claim ruled "
        "against stays on the desk: the disagreement is on screen for a viewer to "
        "inspect, because a silent rejection is the only one that cannot be audited."
    ),
    kind=AgentKind.LANGUAGE_MODEL,
    reads=("draft", "summary", "insight"),
    writes=("verdict",),
)

ROSTER: tuple[AgentSpec, ...] = (GENERATE, INTERPRET, EXPLAIN, RENDER, DESK, VERIFY)

_ORDER: dict[str, int] = {spec.name: i for i, spec in enumerate(ROSTER)}


# --------------------------------------------------------------------------- #
# Trace -- what happened, in handoff order
# --------------------------------------------------------------------------- #
class Trace:
    """The record of one match's runs, merged per agent and ordered by roster.

    Runs are merged rather than listed one per call: sixteen hundred rows for the
    Explainer would say nothing that "1689 calls, 41 ms" does not say better. Two
    sessions opened for the same agent therefore add into the same row, which is
    also what lets a streaming agent and a batch agent sit side by side.
    """

    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}

    def _get(self, spec: AgentSpec) -> AgentRun:
        run = self._runs.get(spec.name)
        if run is None:
            run = AgentRun(
                agent=spec.name,
                role=spec.role,
                kind=spec.kind,
                inside=spec.inside,
            )
            self._runs[spec.name] = run
        return run

    @contextmanager
    def session(self, spec: AgentSpec) -> Iterator[AgentRun]:
        """Open a block of this agent's work, timing whatever happens inside.

        An exception marks the run failed and re-raises: a session is for timing
        and reporting, never for swallowing.
        """
        run = self._get(spec)
        started = time.perf_counter()
        try:
            yield run
        except BaseException as exc:
            run.fail(exc)
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            run.ms = (run.ms or 0.0) + elapsed

    def run(
        self,
        spec: AgentSpec,
        fn: Callable[..., Any],
        *args: Any,
        soft: bool = False,
        detail: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Make one call on this agent's behalf, timed and accounted for.

        `soft=True` means "a failure here costs one payload, not the match":
        the run is marked failed, the error is kept, and None comes back so the
        caller can carry on. Only use it where None is unambiguous -- the caller
        can then tell "produced nothing" from "produced a legitimately empty
        answer". `explain` returns Insight | None on success, so it must not be
        called soft.
        """
        run = self._get(spec)
        if detail:
            run.note(detail)
        started = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
        except BaseException as exc:
            run.ms = (run.ms or 0.0) + (time.perf_counter() - started) * 1000
            run.fail(exc)
            if soft:
                return None
            raise
        run.ms = (run.ms or 0.0) + (time.perf_counter() - started) * 1000
        run.tick()
        return out

    def tick(self, spec: AgentSpec, n: int = 1) -> AgentRun:
        """Account for calls made on this agent's behalf without measuring them.

        Used for Interpret, which the Explainer hosts: its calls are real and its
        effect is real, but its wall time lives inside the Explainer's, and
        splitting them would be a guess dressed up as a number.
        """
        run = self._get(spec)
        run.tick(n)
        return run

    def fail(self, spec: AgentSpec, exc: BaseException) -> AgentRun:
        """Record a failure the caller has already decided to survive."""
        run = self._get(spec)
        run.fail(exc)
        return run

    def runs(self) -> list[AgentRun]:
        """Every run so far, in handoff order."""
        return sorted(self._runs.values(), key=lambda r: _ORDER[r.agent])

    def snapshot(self) -> list[dict[str, Any]]:
        """JSON-ready form, for the terminal frame and the on-screen trace."""
        return [run.model_dump(mode="json") for run in self.runs()]


# --------------------------------------------------------------------------- #
# Blackboard -- the one shared state
# --------------------------------------------------------------------------- #
class Blackboard:
    """Everything an agent leaves for another agent to pick up.

    Values are stored as *history*, not as "latest": the desk at full time must
    see the half-time summary's successor rather than have it overwritten, and a
    reader auditing the run can see that Interpret produced 42 states rather than
    trusting that it did.

    The board deliberately has no latent keys at all. Not "latent keys are
    forbidden here" -- there is nowhere to put them, because `put` will only
    accept a key some agent declared, and no agent declares one.
    """

    def __init__(self) -> None:
        self.trace = Trace()
        self._store: dict[str, list[Any]] = {}

    # -- the courier's write side ----------------------------------------- #
    def put(self, spec: AgentSpec, key: str, value: Any) -> None:
        """Commit a product under the producing agent's contract."""
        if key not in spec.writes:
            raise ContractError(
                f"{spec.name!r} may not write {key!r}; its contract declares "
                f"{list(spec.writes) or 'nothing'}"
            )
        self._store.setdefault(key, []).append(value)

    # -- the courier's read side ------------------------------------------ #
    def view(self, spec: AgentSpec) -> dict[str, list[Any]]:
        """Exactly what this agent declared it reads -- never one key more.

        A declared key that has not been produced yet is simply absent: a match
        with no key moments legitimately has no insights to hand the desk. What
        this will never return is a key the agent did not ask for.
        """
        return {k: self._store[k] for k in spec.reads if k in self._store}

    # -- orchestrator-facing reads ---------------------------------------- #
    def latest(self, key: str, default: Any = None) -> Any:
        """The most recent value for a key.

        For `server.py` only, and by design: the courier is the composition
        root, and constraining it would just move the same checks somewhere they
        could not be read. Agents get `view(spec)` and nothing else.
        """
        values = self._store.get(key)
        return values[-1] if values else default

    def history(self, key: str) -> list[Any]:
        """Every value ever committed under this key."""
        return list(self._store.get(key, ()))

    def keys(self) -> set[str]:
        """Which products exist on the board. Used by the contract tests."""
        return set(self._store)
