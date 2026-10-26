"""Touchline's web surface -- the walking skeleton.

WHAT THIS IS
------------
This file is the *composition root* of the web application: the one place where
the pieces of Touchline are assembled behind an HTTP server. Compare it to a
Telegram bot's `main.py`, which imported the handlers and registered them with
the framework. Same job, different framework.

    bot's main.py : import handlers  -> app.add_handler(...)  -> app.run_polling()
    server.py     : import modules   -> @app.get(...) routes   -> uvicorn server:app

The modules it imports (`simulator.py`, `metrics.py`) never import this file
back. That direction -- entrypoint imports modules, never the reverse -- is what
keeps the dependency graph a tree instead of a knot.

WHAT IT STREAMS
---------------
Eight kinds of frame ride the one long-lived connection:

    meta       -- match identity, once, at the start
    event      -- the raw observable event stream (the Ingest layer)
    overlay    -- a timed key-moment payload, whenever the Explain stage finds one
    state      -- a periodic MatchState, rebuilt from observables by metrics.py
    htsummary  -- the half-time MatchSummary, at the half_time marker
    summary    -- the full-time MatchSummary, just before the stream closes
    narrative  -- the studio desk's cited reading, at each reveal (twice: HT, FT),
                  checked by the critic before it leaves the server
    done       -- terminal marker, carrying the run's agent trace

The server streams these as fast as it can produce them. *Playback* -- pause,
speed, replay -- belongs to the client, which buffers the events and advances a
playhead at the viewer's chosen pace. That is why there is no server-side
`speed` any more: pacing moved to the one place that can see the controls.

The page is no longer computing key moments itself. It used to: a little
`keyMoment()` heuristic lived in the browser, deciding what counted as a chance.
Now `explainer.py` decides, on the server, with cited evidence, and ships the
answer as an `overlay` payload. The client renders that verdict instead of
guessing at it -- which is the whole point of moving the intelligence upstream of
the graphics.

The `narrative` frame is the one place a language model is used, and it is
deliberately last: it consumes the finished, cited `MatchSummary` and `Insight`
objects. It cannot fail the stream -- a timeout or a missing key falls back to the
deterministic narrator (`narrator.py`), because the demo must never depend on a
network. Latent ground truth never crosses this boundary: `explainer.py`, like
`metrics.py`, cannot import the simulator, and the narrator is handed only what
they expose.

WHAT IS NEW HERE (the things worth learning)
--------------------------------------------
  1. async / await   -- the stream below is an async generator. FastAPI runs it
                        on the event loop, so one worker can hold many open
                        connections.
  2. SSE             -- Server-Sent Events: one long-lived HTTP response that
                        the server keeps pushing text into. The browser's
                        EventSource API turns those frames back into events.
  3. type-driven routes -- FastAPI reads the annotations below (`seed: int`,
                        `-> StreamingResponse`) to parse and validate the request
                        and to serialise the response. The signature IS the API.
  4. server time vs client time -- the server decides *what happened*; the
                        client decides *when you see it*. Keeping those apart is
                        what makes a replay scrubbable without the server caring.

Run it:
    ./.venv/bin/python -m uvicorn server:app --reload
Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from agents import (
    DESK,
    EXPLAIN,
    GENERATE,
    INTERPRET,
    RENDER,
    VERIFY,
    AgentSpec,
    Blackboard,
    ContractError,
)
from explainer import MatchExplainer
from narrator import Narrator, build_context, narrator_from_env
from overlay import payload_for
from schemas import Reveal
from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator
from verifier import VerificationContext, Verifier, Review, apply_verdicts, verifier_from_env

# The application object. `uvicorn server:app` means "import `server`, take the
# attribute named `app`, and serve it." This single line is the composition root.
app = FastAPI(title="Touchline", version="0.1.0")


# --------------------------------------------------------------------------- #
# Server-Sent Events -- a tiny text protocol
# --------------------------------------------------------------------------- #
def _sse(event: str, data: str) -> str:
    """Frame one Server-Sent Event.

    SSE is deliberately low-tech: three optional lines and a blank line to end
    the frame.

        event: <name>      which message type this is (optional; defaults to
                           "message")
        data: <payload>    the body (newlines must be escaped or split across
                           several `data:` lines)
        <blank line>       terminates the frame

    The browser side is `EventSource`, which reconnects automatically and hands
    each named frame to `addEventListener("<name>", ...)`.
    """
    return f"event: {event}\ndata: {data}\n\n"


# How often a MatchState snapshot is pushed, in events: responsive enough that
# the live numbers move, sparse enough not to flood the stream. The client pairs
# each frame with the number of events it had received, so it can replay the
# numbers in sync with playback instead of on arrival.
STATE_EVERY_N_EVENTS = 40


# The Narrator is built once per process from the environment. It is the only
# stage that may touch the network, and it fails soft: a missing key, a dead
# endpoint or a timeout all fall back to the deterministic narrator, so the
# stream below always completes.
_NARRATOR: Narrator | None = None

# Same contract, same lazy singleton, for the critic. `LocalVerifier` is the
# offline answer: it checks what a rule can check and says so in its own name.
_VERIFIER: Verifier | None = None


def _narrator() -> Narrator:
    global _NARRATOR
    if _NARRATOR is None:
        _NARRATOR = narrator_from_env()
    return _NARRATOR


def _verifier() -> Verifier:
    global _VERIFIER
    if _VERIFIER is None:
        _VERIFIER = verifier_from_env()
    return _VERIFIER


async def _narrative_frame(
    board: Blackboard, match_id: str, reveal: Reveal, clock_s: int
) -> str | None:
    """Ask the desk for one reveal, have it checked, and frame the result.

    Two agents run here and they meet only through the board. The desk is handed a
    *view* -- its declared reads and nothing else -- so there is no latent key for
    it to reach for, even by accident. Its draft is committed under the desk's
    contract, and the critic is then handed a view of that draft plus the same
    evidence, so the review reads what the desk actually wrote rather than a copy
    passed alongside it.

    The rest is failure recovery, and the two failures are different sizes. A dead
    model costs the commentary: the desk run is marked FAILED, this returns None,
    and the client keeps the previous panel -- no half-built frame is ever sent. A
    dead *critic* costs only the review: the draft still reaches the viewer, and
    the narrative records that it went unchecked rather than implying otherwise.
    """
    view = board.view(DESK)
    summaries = view.get("summary")
    if not summaries:
        board.trace.fail(
            DESK, ContractError("asked to speak before Interpret produced a summary")
        )
        return None
    ctx = build_context(
        match_id, reveal, clock_s, summaries[-1], view.get("insight", [])
    )
    desk = _narrator()
    try:
        with board.trace.session(DESK) as desk_run:
            draft = await desk.narrate(ctx)
            desk_run.tick()
            desk_run.note(f"provider={draft.provider} model={draft.model}")
            if desk.provider == "local":
                # Not a fallback: the deterministic desk was the configured path.
                desk_run.note("deterministic desk, as configured")
            elif draft.provider != desk.provider:
                desk_run.degrade(
                    f"{desk.provider} unavailable; fell back to the local desk"
                )
            elif not draft.validated:
                desk_run.degrade("one or more personas backfilled by the local desk")
    except Exception as exc:  # noqa: BLE001 -- last line of defence around the LLM
        print(f"[server] narrator failed ({type(exc).__name__}: {exc}); "
              "omitting the narrative frame")
        return None
    board.put(DESK, "draft", draft)

    review = await _review(board)
    narrative = apply_verdicts(board.latest("draft"), review)
    board.put(DESK, "narrative", narrative)
    return _sse("narrative", board.latest("narrative").model_dump_json())


async def _review(board: Blackboard) -> Review:
    """Run the critic under its own contract, and never let it cost the desk's work.

    Given a failure here the draft stands unreviewed, which is the honest outcome:
    the claims were not checked, and the narrative says so (`verifier="none"`).
    """
    view = board.view(VERIFY)
    drafts = view.get("draft")
    summaries = view.get("summary")
    if not drafts or not summaries:
        board.trace.fail(VERIFY, ContractError("asked to review before a draft existed"))
        return Review(provider="none", model="none")
    critic = _verifier()
    try:
        with board.trace.session(VERIFY) as critic_run:
            review = await critic.review(
                VerificationContext(
                    draft=drafts[-1],
                    summary=summaries[-1],
                    insights=view.get("insight", []),
                )
            )
            critic_run.tick()
            critic_run.note(f"provider={review.provider} model={review.model}")
            if critic.provider == "local":
                critic_run.note("attribution-only critic, as configured")
            elif review.fell_back:
                critic_run.degrade(
                    f"{critic.provider} unavailable; attribution-only critic stood in"
                )
            elif not review.complete:
                critic_run.degrade("the critic did not rule on every claim")
            # The count goes in the trace, not into the prose: a review that found
            # nothing and a review that never happened must not read the same way.
            critic_run.note(
                f"{review.disputed} of {len(review.verdicts)} claims disputed"
                if review.disputed
                else f"all {len(review.verdicts)} claims stood"
            )
    except Exception as exc:  # noqa: BLE001 -- a failed review must not cost the commentary
        board.trace.fail(VERIFY, exc)
        print(f"[server] verifier failed ({type(exc).__name__}: {exc}); "
              "the draft stands unreviewed")
        return Review(provider="none", model="none")
    board.put(VERIFY, "verdict", review)
    return review


def _product(board: Blackboard, spec: AgentSpec, fn: Callable[[], Any]) -> Any:
    """Read one of an agent's products; a broken read costs a frame, not the match.

    Deliberately not timed: Interpret's wall time lives inside the Explainer's,
    and reporting a number that only covers the moments we happened to sample
    would be worse than reporting none. `Trace.run` is for agents whose calls can
    be timed honestly.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 -- an observability product must not kill the stream
        board.trace.fail(spec, exc)
        print(f"[server] {spec.name} failed to produce a product "
              f"({type(exc).__name__}: {exc}); skipping this frame")
        return None


async def _stream_match(seed: int) -> AsyncIterator[str]:
    """Simulate one match, replay its events, and stream live meaning alongside.

    The whole match is generated first (the simulator is synchronous and
    deterministic), then streamed as fast as it can be written. The *client*
    owns the clock: it buffers these frames and advances a playhead at the
    viewer's chosen speed, so pause / speed / replay need no cooperation from
    the server. The one price is that "live" here means "a deterministic replay
    at 1x if you want it" -- see the playback engine in the page below.

    Four layers ride the one connection: the raw `event` stream, an `overlay`
    (the Explain stage's cited key moments), a periodic `state` (Interpret,
    updated incrementally), and a final `summary`. The `MatchExplainer` instance
    wraps a `MatchMetrics` and is fed exactly the events we send -- the same bytes
    the browser receives -- so it can only ever know what the viewer knows.
    """
    # The one shared state every agent meets at. Nothing latent is ever put on
    # it: `Blackboard.put` only accepts a key some agent declared as a product,
    # and no agent declares one.
    board = Blackboard()
    with board.trace.session(GENERATE) as gen:
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"LIVE-{seed}", seed=seed)
        gen.tick()
        sim.simulate(live=True)  # adds kickoff / half-time / full-time markers
        gen.tick()
    home_id, away_id = sim.home.team_id, sim.away.team_id

    identity = {
        "match_id": sim.match_id,
        "home": sim.home.name,
        "away": sim.away.name,
        "home_id": home_id,
        "away_id": away_id,
        "events": len(sim.snapshots),
    }
    board.put(GENERATE, "match", identity)

    # The Explainer is built from the identity the generator committed, not from
    # the simulator object -- that is the handoff, and it is the same data.
    explainer = MatchExplainer(
        identity["match_id"],
        identity["home_id"],
        identity["away_id"],
        home_name=identity["home"],
        away_name=identity["away"],
    )

    yield _sse("meta", json.dumps(identity))

    ht_emitted = False
    last_clock = 0
    for i, event in enumerate(sim.observable_stream()):
        # Only the OBSERVABLE stream is sent. The latent ground truth (control,
        # chaos, pass intent, big-chance/error labels) never crosses this
        # boundary -- enforced by the fact that `observable_stream()` cannot
        # yield it, and neither `metrics.py` nor `explainer.py` imports the
        # simulator to reach it.
        last_clock = event.clock_s
        insight = board.trace.run(EXPLAIN, explainer.update, event)
        board.trace.tick(INTERPRET)  # the Explainer advanced Interpret once
        yield _sse("event", event.model_dump_json())

        # A key moment, explained with citations, as a timed overlay payload. The
        # client buffers these with the events and shows each one when its own
        # playhead reaches it, so the overlay stays in step with playback.
        if insight is not None:
            # Produced, committed under Explain's contract, then handed to the
            # renderer *from the board*. The wire is a consumer too: a frame can
            # only carry a product the producing agent was contracted to make.
            board.put(EXPLAIN, "insight", insight)
            payload = board.trace.run(
                RENDER, payload_for, board.view(RENDER)["insight"][-1], soft=True
            )
            if payload is not None:
                board.put(RENDER, "overlay", payload)
                yield _sse("overlay", board.latest("overlay").model_dump_json())

        if (i + 1) % STATE_EVERY_N_EVENTS == 0:
            state = _product(board, INTERPRET, explainer.state)
            if state is not None:
                board.put(INTERPRET, "state", state)
                yield _sse("state", board.latest("state").model_dump_json())
        if not ht_emitted and event.type.value == "half_time":
            summary = _product(board, INTERPRET, explainer.summary)
            if summary is not None:
                board.put(INTERPRET, "summary", summary)
                yield _sse("htsummary", board.latest("summary").model_dump_json())
                # The desk speaks at the break, over the half-time card.
                frame = await _narrative_frame(
                    board, identity["match_id"], Reveal.HALF_TIME, event.clock_s
                )
                if frame:
                    yield frame
            ht_emitted = True

    # Full time: the aggregate the summary screen renders, then the desk's verdict.
    summary = _product(board, INTERPRET, explainer.summary)
    if summary is not None:
        board.put(INTERPRET, "summary", summary)
        yield _sse("summary", board.latest("summary").model_dump_json())
    frame = await _narrative_frame(
        board, identity["match_id"], Reveal.FULL_TIME, last_clock
    )
    if frame:
        yield frame

    # The full-time narrative is the one the done frame describes. Reading it back
    # from the board (rather than recomputing) means the trace panel can never
    # report a dispute count that belongs to a different draft than the one on
    # screen. `disputed_claims` is surfaced, not deleted: the count says how many
    # claims the critic ruled against, and those claims still arrive with their
    # rulings attached.
    # Only trust the latest narrative when the full-time frame was actually sent;
    # otherwise the board still holds the half-time one and its count would be
    # attributed to the wrong reveal.
    final_narrative = board.latest("narrative") if frame else None
    yield _sse(
        "done",
        json.dumps(
            {
                "score": {
                    "home": sim.score[home_id],
                    "away": sim.score[away_id],
                },
                # Who did what, in handoff order. The trace panel and the tests are
                # the two readers of this.
                "trace": board.trace.snapshot(),
                "disputed_claims": (
                    final_narrative.disputed_claims
                    if final_narrative is not None
                    else None
                ),
            }
        ),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe -- used by the container platform and by you, debugging."""
    return {"status": "ok", "stream": "/events"}


@app.get("/events")
async def events(
    seed: int = Query(42, description="Match seed; the same seed yields the same match."),
) -> StreamingResponse:
    """Stream one synthetic match as Server-Sent Events.

    `StreamingResponse` never buffers the whole body: FastAPI iterates the async
    generator and writes each frame as it is produced, which is exactly what a
    long-lived push stream needs.
    """
    return StreamingResponse(
        _stream_match(seed),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disables buffering in reverse proxies (nginx and friends) so frames
            # are not held back in production deployments.
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- #
# The browser surface -- deliberately tiny, one file, no build step
# --------------------------------------------------------------------------- #
# Kept inline (rather than a template engine or static directory) because the
# skeleton's value is that there is nothing between reading this file and seeing
# pixels. When the analyst console and overlay renderer arrive the pitch panels
# graduate to real templates / static assets.
#
# Four surfaces, top to bottom, exactly as designed: the pitch (an action map
# with switchable lenses), the live numbers, the event ticker -- whose key moments
# now arrive pre-explained, with the events each claim cites -- and, revealed
# at the whistle rather than browsed, the full-time summary, which is the
# metrics engine given a face.
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Touchline — live match intelligence</title>
  <style>
    :root {
      color-scheme: dark;
      --line: #141b23; --ink: #d7e0ea; --mut: #7d8b99;
      --home: #3b82f6; --away: #f59e0b;
      --poss: #facc15; --chance: #4ade80; --giveaway: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 1.1rem 1.4rem;
      font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #0b0f14; color: var(--ink);
    }
    header { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
    h1 { font-size: 1rem; margin: 0; letter-spacing: .08em; text-transform: uppercase; }
    .meta { color: var(--mut); }
    #scoreboard { margin-left: auto; display: flex; gap: .6rem; align-items: baseline; }
    #score { font-weight: 700; font-size: 1.15rem; color: #f2f5f8; }
    #clock { color: var(--mut); }
    #status { font-size: .78rem; padding: .1rem .5rem; border-radius: 1rem; }
    .up { background: #16351f; color: #6ee7a0; }
    .down { background: #3a1a1a; color: #f79a9a; }

    /* The live strip: the same numbers the pitch lenses draw. */
    #strip { display: flex; gap: 1.2rem; align-items: center; margin: .8rem 0 .5rem; flex-wrap: wrap; }
    .posbar {
      flex: 0 0 220px; height: 10px; display: flex; border-radius: 5px;
      overflow: hidden; background: #1a2430;
    }
    .posbar .h { background: var(--home); }
    .posbar .a { background: var(--away); }
    .chips { display: flex; gap: 1.1rem; color: var(--mut); font-size: .82rem; flex-wrap: wrap; }
    .chips b { color: var(--ink); font-weight: 600; }
    /* The possession split, painted faintly on the pitch. It is a *derived*
       headline (time-based possession from observable events), so it belongs on
       the analyst surface -- unlike control_true / chaos_true, which never do.
       Faint on purpose: readable at a glance, never fighting the event marks. */
    .possplit { font-weight: 800; font-size: 13px; letter-spacing: .05em; }
    .possplit.home { fill: #93c5fd; }
    .possplit.away { fill: #fcd34d; }

    /* Playback. The server streams the whole match; these controls pace it. */
    #controls { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin: .3rem 0 .7rem; }
    .btn, .spd, #seedsel, #seedin {
      font: inherit; font-size: .78rem; color: var(--ink); background: #0f151d;
      border: 1px solid #1d2836; border-radius: 6px; padding: .22rem .6rem; cursor: pointer;
    }
    .btn:hover, .spd:hover { border-color: #2b3a4a; }
    .speeds { display: flex; gap: .3rem; }
    .spd { border-radius: 999px; color: var(--mut); }
    .spd.active { color: #0b0f14; background: #9fd0ff; border-color: #9fd0ff; font-weight: 600; }
    #seedin { width: 7rem; cursor: text; }
    .grow { flex: 1 1 auto; }
    /* The preview's way out. While "Show me" replays a cited moment the reveal
       card is closed, so this button stays on screen for the whole preview as an
       explicit route back. It is what makes the preview a detour rather than a
       dead end. */
    #backtodesk { color: #0b0f14; background: #9fd0ff; border-color: #9fd0ff; font-weight: 600; }
    #backtodesk:hover { border-color: #9fd0ff; opacity: .88; }

    /* The pitch: an honest action map drawn to scale (105m x 68m). Every mark
       is one real event, and nothing is invented between them. */
    #pitch { margin: .4rem 0 .9rem; }
    .lensbar { display: flex; gap: .4rem; margin-bottom: .5rem; flex-wrap: wrap; }
    .lens {
      font: inherit; font-size: .78rem; color: var(--mut); cursor: pointer;
      background: #0f151d; border: 1px solid #1d2836; border-radius: 999px;
      padding: .18rem .7rem;
    }
    .lens:hover { color: var(--ink); border-color: #2b3a4a; }
    .lens.active { color: #0b0f14; background: #9fd0ff; border-color: #9fd0ff; font-weight: 600; }
    #pitchsvg { display: block; width: 100%; max-height: 56vh; border-radius: 8px; }
    .pitchhint { color: #56636f; font-size: .74rem; margin-top: .35rem; }

    /* The legend: a judge should be able to decode every colour and width. */
    .legend { display: flex; gap: 1rem; flex-wrap: wrap; color: var(--mut);
              font-size: .74rem; margin-top: .45rem; }
    .lg { display: inline-flex; align-items: center; gap: .35rem; }
    .sw { width: .7rem; height: .7rem; border-radius: 2px; display: inline-block; }
    .sw.poss { background: var(--poss); }
    .sw.chance { background: var(--chance); }
    .sw.giveaway { background: var(--giveaway); }
    .sw.ball { background: #f1f5f9; border-radius: 50%; }

    ol { list-style: none; margin: 0; padding: 0; }
    li {
      display: grid; grid-template-columns: 4rem 9rem 4rem 1fr auto;
      gap: .75rem; padding: .2rem .4rem; border-bottom: 1px solid var(--line);
      white-space: nowrap; border-left: 3px solid transparent;
    }
    li.goal { background: #12261a; border-left-color: var(--chance); }
    li.chance { border-left-color: var(--chance); }
    li.giveaway { border-left-color: var(--giveaway); }
    li.poss { border-left-color: var(--poss); }
    .t { color: #8ea2b5; }
    .type { color: #9fd0ff; }
    .team { color: #c9d4df; }
    .out { color: var(--mut); }
    .extra { color: #b7a3ff; }
    .empty { color: #56636f; padding: 1rem .4rem; border-left: 0; }
    /* The cited "why" under a key moment. Every clause here is checkable against
       an event on the pitch -- that is the difference between an explanation and
       a caption. */
    li.keyrow { grid-template-columns: 1fr; gap: 0; }
    li.keyrow .row { display: grid; grid-template-columns: 4rem 9rem 4rem 1fr auto;
                     gap: .75rem; }
    .why { color: #8ea2b5; font-size: .76rem; padding: .15rem 0 .1rem 4.75rem;
           white-space: normal; }
    .why b { color: #cbd5e1; font-weight: 600; }

    /* The full-time reveal: a broadcast-app stat screen, kept off the live view
       until the whistle so it never competes with the action. */
    .overlay {
      position: fixed; inset: 0; z-index: 20; padding: 1.5rem;
      background: rgba(5, 8, 12, .82); display: flex;
      align-items: center; justify-content: center;
    }
    .hidden { display: none !important; }
    .card {
      width: min(620px, 100%); max-height: 90vh; overflow: auto;
      background: #0e141c; border: 1px solid #1d2836; border-radius: 12px;
      padding: 1.2rem 1.3rem;
    }
    .ft-head { text-align: center; margin-bottom: 1rem; }
    .ft-teams { display: flex; gap: 1rem; justify-content: center; align-items: baseline; font-size: 1.05rem; }
    .ft-score { font-size: 1.8rem; font-weight: 700; }
    .ft-sub { color: var(--mut); font-size: .78rem; margin-top: .3rem; letter-spacing: .14em; text-transform: uppercase; }
    .row { display: grid; grid-template-columns: 3.4rem 1fr 3.4rem; align-items: center; gap: .5rem; padding: .24rem 0; }
    .row .hv { text-align: right; }
    .row .av { text-align: left; }
    .row .lbl { color: var(--mut); font-size: .8rem; text-align: center; }
    /* The two-sided bar: home from the left, away filling the remainder. */
    .row .bar { grid-column: 1 / -1; height: 4px; border-radius: 2px; background: var(--away); display: flex; overflow: hidden; }
    .row .bar i { background: var(--home); }
    .ft-foot { margin-top: 1rem; padding-top: .7rem; border-top: 1px solid var(--line);
               color: var(--mut); font-size: .8rem; text-align: center; }
    .ft-dismiss { margin-top: .5rem; color: #56636f; font-size: .72rem; text-align: center; }

    /* The studio desk: three pundits reading the SAME facts in three voices.
       Every claim shows the events it is true because of, and a claim anchored on
       a real moment offers "Show me" -- which closes the card and drops the
       playhead onto that event. The desk is where the grounded language model
       speaks, so its provenance (which model, or the deterministic fallback) is
       printed rather than hidden. */
    .desk { margin-top: 1rem; padding-top: .8rem; border-top: 1px solid var(--line); }
    .desk-head { display: flex; align-items: baseline; gap: .55rem; margin-bottom: .55rem; }
    .desk-head b { font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
                   color: #cbd5e1; }
    .prov { font-size: .7rem; color: #56636f; margin-left: auto; }
    .prov.model { color: #9fd0ff; }
    .desk-typing { color: #56636f; font-size: .8rem; font-style: italic; padding: .3rem 0; }
    .pundit { border-left: 3px solid #2b3a4a; padding: .4rem .65rem .5rem; margin-bottom: .5rem;
              background: #0b1119; border-radius: 0 6px 6px 0; }
    .pundit.analyst { border-left-color: #9fd0ff; }
    .pundit.diehard { border-left-color: var(--chance); }
    .pundit.numbers { border-left-color: #b7a3ff; }
    .pundit-head { display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; }
    .pname { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
             color: var(--mut); }
    .pline { font-weight: 600; font-size: .86rem; }
    .pundit ul { list-style: none; margin: .35rem 0 0; padding: 0; }
    .pundit li { display: block; white-space: normal; border: 0; border-bottom: 0;
                 padding: .2rem 0; color: #c9d4df; font-size: .82rem; line-height: 1.35; }
    .pundit li.clickable { cursor: pointer; }
    .showme { font: inherit; font-size: .68rem; margin-left: .45rem; padding: .07rem .5rem;
              color: #0b0f14; background: #9fd0ff; border: 0; border-radius: 999px;
              cursor: pointer; opacity: 0; transition: opacity .12s; white-space: nowrap; }
    .pundit li.clickable:hover .showme, .pundit li.open .showme { opacity: 1; }
    .ev { display: none; margin: .3rem 0 .1rem; padding: .35rem .55rem;
          background: #0f151d; border: 1px solid #1d2836; border-radius: 6px; }
    .pundit li.open .ev { display: block; }
    .evrow { font-size: .72rem; color: #8ea2b5; padding: .05rem 0; }
    .evrow b { color: #9fd0ff; }
    .evteam { color: #c9d4df; }
    .judge { font-size: .68rem; color: #56636f; margin-top: .3rem; white-space: normal; }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(159, 208, 255, .55); }
      100% { box-shadow: 0 0 0 16px rgba(159, 208, 255, 0); }
    }
    .pulse { animation: pulse .85s ease-out 2; border-radius: 8px; }

    /* Orchestration trace panel */
    .orchestration {
      margin-top: 1rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0e141c;
      overflow: hidden;
    }
    .orch-head {
      padding: .5rem .7rem;
      background: #0f151d;
      border-bottom: 1px solid var(--line);
      font-size: .78rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: #cbd5e1;
    }
    .orch-body {
      padding: .5rem .7rem;
      display: flex;
      flex-direction: column;
      gap: .25rem;
    }
    .orch-row {
      display: grid;
      grid-template-columns: 5rem 8rem 6rem 1fr;
      gap: .75rem;
      align-items: baseline;
      padding: .15rem 0;
      border-bottom: 1px solid var(--line);
      font-size: .76rem;
    }
    .orch-row:last-of-type { border-bottom: none; }
    .orch-status {
      color: var(--mut);
      text-transform: uppercase;
      font-size: .7rem;
      letter-spacing: .08em;
    }
    .orch-row.ok .orch-status { color: #6ee7a0; }
    .orch-row.failed .orch-status { color: #f79a9a; }
    .orch-agent { color: #9fd0ff; }
    .orch-calls { color: var(--mut); }
    .orch-detail { color: #c9d4df; white-space: normal; }
    .orch-foot {
      margin-top: .3rem;
      padding-top: .4rem;
      border-top: 1px solid var(--line);
      color: var(--mut);
      font-size: .74rem;
    }

    .app-footer {
      margin-top: 1.2rem;
      margin-bottom: 2rem;
    }
  </style>
</head>
<body>
  <header>
    <h1>Touchline</h1>
    <span class="meta" id="matchup">connecting…</span>
    <span id="status" class="down">idle</span>
    <span id="scoreboard">
      <span id="score">0 – 0</span>
      <span id="clock">0'</span>
    </span>
  </header>

  <section id="strip" class="hidden">
    <div class="posbar" title="possession (share of controlled time)">
      <span class="h" id="pos-h"></span><span class="a" id="pos-a"></span>
    </div>
    <div class="chips" id="chips"></div>
  </section>

  <div id="controls">
    <button id="playpause" class="btn" title="Pause">❚❚</button>
    <span class="speeds" id="speeds"></span>
    <button id="backtodesk" class="btn hidden" title="Leave the replay and return to the desk">← back to the desk</button>
    <span class="grow"></span>
    <button id="replay" class="btn" title="Replay from the start">⟲</button>
    <select id="seedsel" title="Choose a match"></select>
    <input id="seedin" type="number" min="0" placeholder="seed #">
    <button id="newmatch" class="btn">New match</button>
  </div>

  <section id="pitch">
    <div class="lensbar" id="lensbar"></div>
    <svg id="pitchsvg" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet">
      <g id="pitchlines"></g>
      <g id="pitchstats"></g>
      <g id="marks"></g>
      <g id="active"></g>
      <g id="discs"></g>
    </svg>
    <div class="pitchhint" id="pitchhint"></div>
    <div class="legend">
      <span class="lg"><i class="sw poss"></i>possession change</span>
      <span class="lg"><i class="sw chance"></i>chance · xG ≥ 0.35</span>
      <span class="lg"><i class="sw giveaway"></i>deep giveaway</span>
      <span class="lg"><i class="sw ball"></i>shot · dashed line to target</span>
      <span class="lg">pass thickness: short / medium / long</span>
    </div>
  </section>

  <ol id="feed"><li class="empty">waiting for the first event…</li></ol>

  <footer class="app-footer">
    <section id="orchestration" class="orchestration">
      <div class="orch-head">Orchestration — agent trace</div>
      <div id="orch-body" class="orch-body">Waiting for match to complete…</div>
    </section>
  </footer>

  <div id="summary" class="overlay hidden">
    <div class="card">
      <div class="ft-head">
        <div class="ft-teams">
          <span id="ft-home"></span>
          <span class="ft-score" id="ft-score">0 – 0</span>
          <span id="ft-away"></span>
        </div>
        <div class="ft-sub" id="ft-sub">Full time</div>
      </div>
      <div id="ft-rows"></div>
      <div id="desk" class="desk"></div>
      <div class="ft-foot" id="ft-foot"></div>
      <div class="ft-dismiss">click outside the card to return to the pitch</div>
    </div>
  </div>

<script>
  const el = (id) => document.getElementById(id);
  const feed = el("feed"), status = el("status"), matchup = el("matchup");
  const scoreEl = el("score"), clockEl = el("clock");
  const MAX_ROWS = 250;   // keep the DOM bounded on a long stream

  const NS = "http://www.w3.org/2000/svg";
  const sx = (x) => (x / 100) * 105;
  const sy = (y) => (y / 100) * 68;
  const mk = (tag, attrs) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  const marksLayer = el("marks"), activeLayer = el("active"), discsLayer = el("discs");
  const pitchStatsLayer = el("pitchstats");
  // The two faint headline numbers painted on the pitch, one per half.
  let homePct = null, awayPct = null;

  // ---- identity, filled from the `meta` frame -------------------------- //
  let HOME = null, AWAY = null, score = {};
  // Team *names*, for the desk's evidence drawer. A claim says "Northbridge's
  // giveaways" and cites events; showing each cited event's team is what lets a
  // reader see for themselves whether the sentence and the tape agree.
  let HOME_NAME = null, AWAY_NAME = null;
  const teamName = (id) =>
    id === HOME ? (HOME_NAME || "home") : id === AWAY ? (AWAY_NAME || "away") : id;
  const TEAM_COLOR = {};
  const setColors = () => { TEAM_COLOR[HOME] = "#3b82f6"; TEAM_COLOR[AWAY] = "#f59e0b"; };

  // ---- playback state -------------------------------------------------- //
  // The server streams the whole match as fast as it can; everything below is
  // the client's own clock over that buffer. This split is the whole reason
  // pause / speed / replay work with no server cooperation.
  const events = [];         // every event received, in order
  const stateFrames = [];    // { at, data } -- `at` = events.length on arrival
  const overlayByAnchor = {}; // anchor_event_id -> the server's key-moment payload
  let playhead = 0;          // how many events have been drawn
  let playClock = 0;         // match-seconds drawn so far
  let stateCursor = 0;       // next unapplied state frame
  let playing = true;
  let speed = 30;            // match-seconds per real second
  let lastTs = null;         // null until the first animation frame
  let pauseUntil = 0;        // auto-pause deadline, ms
  let summaryData = null, htData = null, doneFlag = false, traceData = null, summaryShown = false, htSummaryShown = false, pausedForHt = false;
  let htNarrative = null, ftNarrative = null;   // the desk's reading at each reveal
  let currentReveal = null;                     // "ht" | "ft" -- which card is open
  let lastPossTeam = null;   // for detecting possession-change moments
  let es = null;
  // ---- "Show me" preview ------------------------------------------------- //
  // Pressing "Show me" on a desk claim does not teleport the pitch onto the cited
  // event; it replays the build-up to that event and then hands the viewer back
  // to the desk, card and all. The replay is therefore stateful -- it carries a
  // snapshot to restore to -- which is what stops the jump becoming a one-way
  // trip with nothing on screen and no way back.
  let preview = null;          // { targetIdx, dwellUntil, saved } | null
  let suppressCards = false;   // true mid-rebuild, so no reveal fires on the way through

  const SPEEDS = [8, 15, 30, 60, 120, 300];
  const AUTO_PAUSE_MS = 1800;   // linger on a goal / chance so the eye can land
  const PREVIEW_LEAD_S = 12;    // match-seconds of build-up replayed before the cited moment
  // The preview runs at its own rate rather than the viewer's, so a moment chosen
  // at 300x still reads as a build-up instead of a single flashing frame.
  const PREVIEW_SPEED = 10;     // match-seconds per real second while previewing
  const PREVIEW_DWELL_MS = 1700;// hold on the cited moment itself before the desk returns

  // The match a visitor lands on. It was picked to exercise the most of the
  // surface at once: both teams score, it is a real comeback (3–0 down at the
  // break to 3–3), and it contains every key-moment kind -- goals, big chances
  // and deep give-aways -- plus woodwork, a red card and a busy keeper. The
  // other curated matches sit in the picker; this one is the front door.
  const DEFAULT_SEED = 81;
  const CURATED = [
    ["Comeback · 3–3 from 3 down", 81],
    ["Thriller · 4–5", 148],
    ["Goalfest · 6–3", 217],
    ["Eight-goal draw · 4–4", 365],
    ["Goalless · 0–0", 25],
    ["Tight · 1–0", 5],
  ];

  const clock = (e) => e.minute + (e.stoppage_minute ? "+" + e.stoppage_minute : "");
  // Model prose is untrusted text: it lands in the DOM via innerHTML, so it is
  // escaped on the way in. The gates on the server decide what is *true*; this
  // only decides what is *inert*.
  const esc = (s) => String(s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const PERSONA_LABEL = { analyst: "The Analyst", diehard: "The Diehard", numbers: "The Numbers" };
  function setStatus(text, up) {
    status.textContent = text;
    status.className = up ? "up" : "down";
  }

  // ------------------------------------------------------------------ //
  // The pitch -- an honest action map.
  //
  // Coordinates are the simulator's fixed frame: x=0 is the home goal-line,
  // x=100 the away goal-line, y=0/100 the touchlines. That frame represents a
  // real 105m x 68m pitch, so we draw it in metres and keep the geometry honest.
  //
  // Two rules are load-bearing, and both are the point of the project:
  //   1. Every mark is exactly one real event. Nothing is interpolated.
  //   2. The actor disc sits at the actor's TRUE (x, y) and *jumps* between
  //      events -- no tweening, because faking motion would be inventing data.
  // ------------------------------------------------------------------ //
  const passColor = (e) => (e.outcome === "complete" ? "#4ade80" : "#f87171");
  const shotColor = (e) =>
    e.outcome === "goal" ? "#4ade80" :
    e.outcome === "saved" ? "#fbbf24" :
    e.outcome === "woodwork" ? "#facc15" :
    e.outcome === "blocked" ? "#94a3b8" : "#64748b";

  // Pass thickness encodes DISTANCE (an observable), not difficulty. Distance
  // bands in metres, so the legend can say "short / medium / long" honestly.
  const distBin = (d) => (d == null ? 0 : d < 18 ? 0 : d < 35 ? 1 : 2);
  const DIST_W = [0.22, 0.34, 0.52];

  // A small ball glyph -- a marker, never a moving ball. It sits at the strike
  // and grows with xG; the dashed line shows where the shot was aimed.
  function ballGlyph(e, r) {
    const g = mk("g", {});
    const x = sx(e.x), y = sy(e.y);
    g.appendChild(mk("circle", { cx: x, cy: y, r, fill: "#f1f5f9",
                                 stroke: shotColor(e), "stroke-width": r * 0.42 }));
    g.appendChild(mk("circle", { cx: x, cy: y, r: r * 0.3, fill: "#0b0f14" }));
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2 - Math.PI / 2;
      g.appendChild(mk("circle", { cx: x + Math.cos(a) * r * 0.66, cy: y + Math.sin(a) * r * 0.66,
                                   r: r * 0.17, fill: "#0b0f14" }));
    }
    return g;
  }

  // A lens is just a filter + style over the SAME event list. Switching is
  // client-side, so the server never has to know what the viewer is looking at.
  // `recency: true` routes marks through the bright "current possession" layer.
  const LENSES = {
    action: {
      label: "Action",
      hint: "every event · line = where it went · disc = the actor, at their true position",
      nodes(e) {
        const out = [];
        if (e.end_x != null) out.push(mk("line", {
          x1: sx(e.x), y1: sy(e.y), x2: sx(e.end_x), y2: sy(e.end_y),
          stroke: TEAM_COLOR[e.team], "stroke-width": 0.22, opacity: 0.4,
          "stroke-linecap": "round" }));
        out.push(mk("circle", { cx: sx(e.x), cy: sy(e.y),
          r: e.type === "shot" ? 0.95 : 0.65, fill: TEAM_COLOR[e.team], opacity: 0.75 }));
        return out;
      },
    },
    passing: {
      label: "Passing",
      hint: "every pass · green complete / red incomplete · thickness = short / medium / long",
      recency: true,
      nodes(e) {
        if (e.type !== "pass") return [];
        const out = [];
        if (e.end_x != null) out.push(mk("line", {
          x1: sx(e.x), y1: sy(e.y), x2: sx(e.end_x), y2: sy(e.end_y),
          stroke: passColor(e), "stroke-width": DIST_W[distBin(e.pass_distance)],
          opacity: 0.75, "stroke-linecap": "round" }));
        out.push(mk("circle", { cx: sx(e.x), cy: sy(e.y), r: 0.5, fill: passColor(e), opacity: 0.9 }));
        return out;
      },
    },
    possession: {
      label: "Possession",
      hint: "where each team's play happens (territory) · one dot per real event",
      nodes(e) {
        return [mk("circle", { cx: sx(e.x), cy: sy(e.y), r: 0.55,
          fill: TEAM_COLOR[e.team], opacity: 0.3 })];
      },
    },
    shots: {
      label: "Shots & xG",
      hint: "every shot · the ball marks the strike, the dashed line its target · colour = outcome",
      nodes(e) {
        if (e.type !== "shot") return [];
        const out = [];
        if (e.end_x != null) {
          out.push(mk("line", {
            x1: sx(e.x), y1: sy(e.y), x2: sx(e.end_x), y2: sy(e.end_y),
            stroke: "#cbd5e1", "stroke-width": 0.16, "stroke-dasharray": "0.9 0.9",
            opacity: 0.6, "stroke-linecap": "round" }));
          out.push(mk("circle", { cx: sx(e.end_x), cy: sy(e.end_y), r: 1.0,
            fill: "none", stroke: shotColor(e), "stroke-width": 0.35, opacity: 0.95 }));
        }
        out.push(ballGlyph(e, 0.8 + 2.4 * Math.sqrt(e.shot_quality || 0)));
        return out;
      },
    },
  };
  let currentLens = "action";

  // ---- key moments (decided on the server, not here) -------------------- //
  const POSS_MARKERS = new Set(["pass", "shot"]);
  // The simulator marks every turnover with its own `possession_change` event,
  // so we do not have to *infer* one -- we delimit possession sequences on that
  // event, and call it a genuine *change* only when the team actually differs
  // (a loose ball recovered by the same side ends a phase but is not a change).
  function sequenceEvent(e) {
    if (e.type === "possession_change") {
      const changed = lastPossTeam !== null && e.team !== lastPossTeam;
      lastPossTeam = e.team;
      return { newSeq: true, changed };
    }
    if (POSS_MARKERS.has(e.type)) lastPossTeam = e.team;
    return { newSeq: false, changed: false };
  }
  // There used to be a `keyMoment()` heuristic here, guessing from xG and pitch
  // position what counted as a chance. It is gone on purpose. `explainer.py`
  // decides key moments on the server now, and ships each verdict as an
  // `overlay` payload with the events it cites. This page renders that verdict;
  // it does not second-guess it. One definition, one place.

  // ---- pitch construction ---------------------------------------------- //
  function drawPitch() {
    const g = el("pitchlines");
    const base = { fill: "none", stroke: "#26333f", "stroke-width": 0.22 };
    g.appendChild(mk("rect", { x: 0, y: 0, width: 105, height: 68,
                               fill: "#0f151d", stroke: "#26333f", "stroke-width": 0.35 }));
    g.appendChild(mk("line", { x1: 52.5, y1: 0, x2: 52.5, y2: 68, ...base }));
    g.appendChild(mk("circle", { cx: 52.5, cy: 34, r: 9.15, ...base }));
    g.appendChild(mk("circle", { cx: 52.5, cy: 34, r: 0.3, fill: "#26333f" }));
    const boxY = (68 - 40.32) / 2, sixY = (68 - 18.32) / 2, goalY = (68 - 7.32) / 2;
    for (const left of [true, false]) {
      g.appendChild(mk("rect", { x: left ? 0 : 105 - 16.5, y: boxY, width: 16.5, height: 40.32, ...base }));
      g.appendChild(mk("rect", { x: left ? 0 : 105 - 5.5, y: sixY, width: 5.5, height: 18.32, ...base }));
      g.appendChild(mk("rect", { x: left ? -1.5 : 105, y: goalY, width: 1.5, height: 7.32,
                                 fill: "#0f151d", stroke: "#3d5064", "stroke-width": 0.35 }));
      g.appendChild(mk("circle", { cx: left ? 11 : 94, cy: 34, r: 0.3, fill: "#26333f" }));
    }
  }

  // The possession split, painted faintly across the pitch: home owns the left
  // half in the fixed frame (they attack x=100), so their share sits there. Built
  // once and only *updated* by state frames, so it stays in step with playback.
  function buildPitchStats() {
    homePct = mk("text", { x: 26.25, y: 34, "text-anchor": "middle",
                           "dominant-baseline": "central", "class": "possplit home", opacity: "0" });
    awayPct = mk("text", { x: 78.75, y: 34, "text-anchor": "middle",
                           "dominant-baseline": "central", "class": "possplit away", opacity: "0" });
    pitchStatsLayer.appendChild(homePct);
    pitchStatsLayer.appendChild(awayPct);
  }

  // The actor disc: one persistent marker that JUMPS to each new actor. No CSS
  // transition, so it can never glide between two real events.
  const actorCircle = mk("circle", { r: 2.0, stroke: "#0b0f14", "stroke-width": 0.3 });
  const actorText = mk("text", { "text-anchor": "middle", "dominant-baseline": "central",
                                 "font-size": 2.3, "font-weight": 700, fill: "#0b0f14" });
  const actorG = mk("g", {});
  actorG.appendChild(actorCircle);
  actorG.appendChild(actorText);
  discsLayer.appendChild(actorG);

  function moveActor(e) {
    const x = sx(e.x), y = sy(e.y);
    actorCircle.setAttribute("cx", x);
    actorCircle.setAttribute("cy", y);
    actorCircle.setAttribute("fill", TEAM_COLOR[e.team] || "#94a3b8");
    actorText.setAttribute("x", x);
    actorText.setAttribute("y", y);
    actorText.textContent = e.player_number;
  }

  // When the possession sequence ends, demote its bright marks into the base
  // layer. This is what makes the passing lens show the *current* build-up at a
  // glance -- still one mark per real event, just restyled.
  function flushActive() {
    for (const n of Array.from(activeLayer.children)) {
      const o = parseFloat(n.getAttribute("opacity"));
      n.setAttribute("opacity", (isNaN(o) ? 1 : o) * 0.4);
      marksLayer.appendChild(n);
    }
  }

  function drawMark(e) {
    const lens = LENSES[currentLens];
    const layer = lens.recency ? activeLayer : marksLayer;
    for (const n of lens.nodes(e)) layer.appendChild(n);
  }

  // Re-draw every mark up to the playhead for the current lens (lens switch,
  // replay). Ticker rows are NOT rebuilt here -- they are playback-driven.
  function renderAll() {
    marksLayer.replaceChildren();
    activeLayer.replaceChildren();
    lastPossTeam = null;
    for (let i = 0; i < playhead; i++) {
      const e = events[i];
      if (sequenceEvent(e).newSeq) flushActive();
      drawMark(e);
    }
    if (playhead > 0) moveActor(events[playhead - 1]);
    el("pitchhint").textContent = LENSES[currentLens].hint;
    // Also update the faint possession split visibility to match the lens.
    updatePossSplitVisibility();
  }

  function buildLensBar() {
    const bar = el("lensbar");
    bar.replaceChildren();
    for (const name in LENSES) {
      const b = document.createElement("button");
      b.className = "lens" + (name === currentLens ? " active" : "");
      b.textContent = LENSES[name].label;
        b.onclick = () => { currentLens = name; buildLensBar(); renderAll(); };
      bar.appendChild(b);
    }
  }

  // ---- ticker + live strip --------------------------------------------- //
  // The row shows what happened; for a key moment it also shows *why*, taken
  // straight from the server's cited reasons. No clause appears here that is not
  // traceable to an event the viewer can already see.
  function addRow(e, payload, poss) {
    const empty = feed.querySelector(".empty");
    if (empty) empty.remove();
    const li = document.createElement("li");

    const kind = payload ? payload.insight.kind : null;
    if (kind === "goal" || (e.type === "shot" && e.outcome === "goal")) li.className = "goal";
    else if (kind === "chance") li.className = "chance";
    else if (kind === "deep_giveaway") li.className = "giveaway";
    else if (poss) li.className = "poss";

    let extra = "";
    if (e.shot_quality != null) extra = "xG " + e.shot_quality.toFixed(2);
    else if (e.in_box === true) extra = "in box";

    const row =
      '<span class="t">' + clock(e) + "'</span>" +
      '<span class="type">' + e.type + "</span>" +
      '<span class="team">' + e.team + "</span>" +
      '<span class="out">' + e.player_name + " · " + e.outcome + "</span>" +
      '<span class="extra">' + extra + "</span>";

    if (payload) {
      li.className += " keyrow";
      li.innerHTML = '<div class="row">' + row + "</div>" + whyHtml(payload.insight);
    } else {
      li.innerHTML = row;
    }

    feed.prepend(li);
    // Newest first, so the latest event is always in view without scrolling.
    while (feed.children.length > MAX_ROWS) feed.removeChild(feed.lastChild);
  }

  // The cited reasons, as one line. Each `reason` from the server already carries
  // the event ids it is true because of; we show the sentences and keep the ids
  // in the tooltip, so a curious viewer can audit the claim without reading raw
  // JSON.
  function whyHtml(insight) {
    const parts = insight.reasons.map((r) =>
      '<span title="' + r.event_ids.join(", ") + '">' + r.text + "</span>"
    );
    return '<div class="why">' + parts.join(" ") + "</div>";
  }

  // ---- the studio desk -------------------------------------------------- //
  // Server-side, every claim has already been checked: its citations resolve to
  // real events and every number in it appears in the facts the desk was given.
  // The page's job is only to render that, and to let a viewer audit it -- open
  // the evidence inline, or jump the playhead onto the moment a claim cites.
  function eventById(id) {
    for (let i = 0; i < events.length; i++) if (events[i].event_id === id) return events[i];
    return null;
  }

  function evidenceHtml(ids) {
    return ids.map((id) => {
      const e = eventById(id);
      const body = e
        ? "<b>" + clock(e) + "'</b> " + '<span class="evteam">' +
          esc(teamName(e.team)) + "</span> " + esc(e.player_name) + " · " +
          esc(e.type) + " · " + esc(e.outcome)
        : esc(id) + " (not in this stream)";
      return '<div class="evrow">' + body + "</div>";
    }).join("");
  }

  function renderDesk(narrative) {
    const box = el("desk");
    if (!box) return;
    if (!narrative) {
      box.innerHTML = '<div class="desk-head"><b>The desk</b></div>' +
        '<div class="desk-typing">the desk is still typing…</div>';
      return;
    }
    const provenance = narrative.validated
      ? '<span class="prov model">live desk · ' + esc(narrative.model) +
        (narrative.dropped_claims ? " · " + narrative.dropped_claims + " claim(s) cut" : "") + "</span>"
      : '<span class="prov">deterministic desk · ' + esc(narrative.model) + "</span>";

    const parts = ['<div class="desk-head"><b>The desk</b>' + provenance + "</div>"];
    for (const take of narrative.personas) {
      parts.push('<div class="pundit ' + take.persona + '">');
      parts.push('<div class="pundit-head"><span class="pname">' +
        (PERSONA_LABEL[take.persona] || take.persona) + '</span><span class="pline">' +
        esc(take.headline) + "</span></div><ul>");
      for (const claim of take.claims) {
        const anchored = claim.event_ids.length > 0;
        parts.push('<li class="claim' + (anchored ? " clickable" : "") + '"' +
          (anchored ? ' data-anchor="' + esc(claim.event_ids[0]) + '"' : "") + ">");
        parts.push("<span>" + esc(claim.text) + "</span>");
        if (anchored) {
          parts.push('<button class="showme" type="button">Show me</button>');
          parts.push('<div class="ev">' + evidenceHtml(claim.event_ids) + "</div>");
        }
        parts.push("</li>");
      }
      parts.push("</ul></div>");
    }
    parts.push('<div class="judge">Hover a claim for the events behind it, ' +
      "or press “Show me” to jump the pitch to that moment.</div>");
    box.innerHTML = parts.join("");

    // Wire the interactions after the HTML is in the DOM. Clicking the row opens
    // the evidence drawer; the button seeks to the claim's first cited event.
    box.querySelectorAll("li.claim.clickable").forEach((li) => {
      const btn = li.querySelector(".showme");
      li.addEventListener("click", (e) => {
        if (e.target === btn) return;          // the button seeks, it does not toggle
        li.classList.toggle("open");
      });
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        seekToEvent(li.dataset.anchor);
      });
    });
  }

  function pulsePitch() {
    const node = el("pitch");
    node.classList.remove("pulse");
    void node.offsetWidth;                     // restart the animation
    node.classList.add("pulse");
    setTimeout(() => node.classList.remove("pulse"), 1900);
  }

  // ---- rebuild, preview, return ----------------------------------------- //
  // Redraw the pitch from the in-memory buffer up to (not including) index
  // `upto`. Replaying rather than teleporting is what keeps the pitch an honest
  // record: every mark, ticker row and score is redrawn exactly as it would have
  // been live, and nothing is interpolated between them. Returns the playhead it
  // landed on so the caller can carry on from there.
  function rebuildTo(upto) {
    suppressCards = true;
    marksLayer.replaceChildren();
    activeLayer.replaceChildren();
    const stop = Math.max(0, Math.min(upto, events.length));
    feed.innerHTML = stop > 0 ? "" : '<li class="empty">waiting for the first event…</li>';
    // Live numbers are cleared too, so a rebuild can never show values belonging
    // to a point further along the match than where the playhead now sits.
    el("chips").innerHTML = "";
    el("pos-h").style.width = "50%";
    el("pos-a").style.width = "50%";
    score = {}; if (HOME) { score[HOME] = 0; score[AWAY] = 0; }
    scoreEl.textContent = "0 – 0";
    playhead = 0; stateCursor = 0; lastPossTeam = null;
    for (let i = 0; i < stop; i++) {
      drawEvent(events[i]);
      playhead = i + 1;
    }
    while (stateCursor < stateFrames.length && stateFrames[stateCursor].at <= playhead) {
      paintState(stateFrames[stateCursor].data);
      stateCursor++;
    }
    suppressCards = preview !== null;   // a preview owns the screen until it ends
    return playhead;
  }

  // Rewind to just before a cited event and play its build-up through. The
  // snapshot is taken *before* the jump, so returning restores the exact session
  // the viewer came from -- at half time that means the held half-time card and
  // the paused clock, not just "somewhere earlier".
  function seekToEvent(eventId) {
    if (!eventId) return;
    const idx = events.findIndex((e) => e.event_id === eventId);
    if (idx < 0) return;
    const target = events[idx];

    preview = {
      targetIdx: idx,
      dwellUntil: 0,
      saved: {
        playhead, playClock, summaryShown, htSummaryShown, pausedForHt,
        currentReveal, pauseUntil, playing,
        revealVisible: !el("summary").classList.contains("hidden"),
        status: status.textContent,
        statusUp: status.classList.contains("up"),
      },
    };

    el("summary").classList.add("hidden");
    el("backtodesk").classList.remove("hidden");
    currentReveal = null;
    summaryShown = false;   // the preview must not trip the full-time reveal
    pausedForHt = false;    // and it owns playback while it runs

    // Start far enough back that the moment arrives with a visible build-up.
    let start = idx;
    while (start > 0 && events[start - 1].clock_s >= target.clock_s - PREVIEW_LEAD_S) start--;
    rebuildTo(start);

    playClock = events[start] ? events[start].clock_s : target.clock_s;
    playing = true; pauseUntil = 0; lastTs = null;
    updatePlayBtn();
    clockEl.textContent = start > 0 ? clock(events[start - 1]) + "'" : "0'";
    setStatus("replaying the build-up", true);
    pulsePitch();
  }

  // Hand the pitch back to exactly where the desk left off: same playhead, same
  // clock, same reveal card. The narrative is re-rendered rather than assumed
  // still on screen, because it may have arrived while the preview held the card
  // closed.
  function exitPreview() {
    if (!preview) return;
    const saved = preview.saved;
    preview = null;
    el("backtodesk").classList.add("hidden");

    rebuildTo(saved.playhead);
    playClock = saved.playClock;
    summaryShown = saved.summaryShown;
    htSummaryShown = saved.htSummaryShown;
    pausedForHt = saved.pausedForHt;
    currentReveal = saved.currentReveal;
    playing = saved.playing;
    pauseUntil = saved.pauseUntil;
    lastTs = null;
    updatePlayBtn();
    if (saved.playhead > 0) clockEl.textContent = clock(events[saved.playhead - 1]) + "'";
    setStatus(saved.status, saved.statusUp);

    if (saved.revealVisible) {
      el("summary").classList.remove("hidden");
      if (currentReveal === "ht" && htNarrative) renderDesk(htNarrative);
      else if (currentReveal === "ft" && ftNarrative) renderDesk(ftNarrative);
    }
  }

  function updatePossSplitVisibility() {
    const inPoss = currentLens === "Possession";
    const opacity = inPoss ? "0.12" : "0";
    if (homePct) homePct.setAttribute("opacity", opacity);
    if (awayPct) awayPct.setAttribute("opacity", opacity);
  }

  function paintState(s) {
    const ph = s.possession[HOME], pa = s.possession[AWAY];
    el("pos-h").style.width = ph * 100 + "%";
    el("pos-a").style.width = pa * 100 + "%";
    // Possession is two-sided, so both numbers are shown -- a lone "55%" reads
    // as if one team owns the ball, which is exactly the ambiguity to avoid.
    el("chips").innerHTML =
      "poss <b>" + (ph * 100).toFixed(0) + "% – " + (pa * 100).toFixed(0) + "%</b> · " +
      "shots <b>" + s.shots[HOME] + "–" + s.shots[AWAY] + "</b> · " +
      "xG <b>" + s.shot_quality_total[HOME].toFixed(2) + "–" +
      s.shot_quality_total[AWAY].toFixed(2) + "</b> · " +
      "tempo <b>" + s.tempo.toFixed(0) + "</b>/min · " +
      "chaos <b>" + s.chaos.toFixed(2) + "</b>";
    // Same split, faint on the pitch. Visibility follows the Possession lens.
    if (homePct) homePct.textContent = (ph * 100).toFixed(0) + "%";
    if (awayPct) awayPct.textContent = (pa * 100).toFixed(0) + "%";
    updatePossSplitVisibility();
  }

  // ---- playback loop ---------------------------------------------------- //
  function drawEvent(e) {
    const seq = sequenceEvent(e);
    if (seq.newSeq) flushActive();

    // The server's verdict on this event, if it is a key moment. This page does
    // not decide -- it renders. The payload was buffered with the events, so a
    // moment lights up when the playhead *reaches* it, not when it arrived.
    const payload = overlayByAnchor[e.event_id] || null;
    addRow(e, payload, seq.changed);
    drawMark(e);
    moveActor(e);

    if (e.type === "shot" && e.outcome === "goal") {
      score[e.team] = (score[e.team] || 0) + 1;
      scoreEl.textContent = score[HOME] + " – " + score[AWAY];
    }
    clockEl.textContent = clock(e) + "'";
    // Auto-pause on a goal, a chance, or the half-time whistle. Full time needs
    // no pause here -- reaching the last event reveals the summary. Suppressed
    // while a rebuild or a preview is in flight: the caller decides what is on
    // screen, so redrawing the pitch must never raise a card on its own.

    if (e.type === "half_time" && !htSummaryShown && !suppressCards) {
      if (htData && HOME) {
        renderHtSummary(htData);
      } else {
        // Fallback minimal
        if (HOME) {
          el("ft-home").textContent = matchup.textContent.split(" vs ")[0];
          el("ft-away").textContent = matchup.textContent.split(" vs ")[1] || "";
        }
        el("ft-score").textContent = scoreEl.textContent;
        el("ft-rows").innerHTML = "";
        el("ft-foot").textContent = "";
        el("ft-sub").textContent = "Half time";
      }
      htSummaryShown = true;
      pausedForHt = true;
      playing = false;
      updatePlayBtn();
      el("summary").classList.remove("hidden");
      pauseUntil = Number.POSITIVE_INFINITY;
      return true;
    }
    return (e.type === "shot" && e.outcome === "goal") ||
      (payload !== null && payload.insight.kind === "chance");
  }

  function advance(ts) {
    while (playhead < events.length && events[playhead].clock_s <= playClock) {
      const e = events[playhead];
      const autoPause = drawEvent(e);
      playhead++;
      if (autoPause) { pauseUntil = ts + AUTO_PAUSE_MS; break; }
    }
    while (stateCursor < stateFrames.length && stateFrames[stateCursor].at <= playhead) {
      paintState(stateFrames[stateCursor].data);
      stateCursor++;
    }
  }

  function maybeReveal() {
    if (preview) return;   // a preview is a detour: it does not raise the full-time card
    if (summaryShown || !summaryData || !doneFlag || playhead < events.length) return;
    renderSummary(summaryData);
    summaryShown = true;
    playing = false;
    updatePlayBtn();
    setStatus("full time", false);
    el("summary").classList.remove("hidden");
  }

  function frame(ts) {
    if (lastTs === null) lastTs = ts;
    // Clamp dt so a backgrounded tab (which stops rAF) does not fast-forward the
    // whole match in one jump when it wakes up.
    const dt = Math.min((ts - lastTs) / 1000, 0.1);
    lastTs = ts;
    if (playing && ts >= pauseUntil) {
      playClock += dt * (preview ? PREVIEW_SPEED : speed);
      advance(ts);
    }
    stepPreview(ts);
    maybeReveal();
    requestAnimationFrame(frame);
  }

  // traceData declared above with other state variables
  function renderOrchestration() {
    const body = document.getElementById("orch-body");
    if (!body) return;
    if (!traceData || !traceData.trace) {
      body.textContent = "Waiting for match to complete…";
      return;
    }
    body.innerHTML = "";
    for (const r of traceData.trace) {
      const row = document.createElement("div");
      row.className = "orch-row " + (r.status || "");
      const status = document.createElement("span");
      status.className = "orch-status";
      status.textContent = r.status;
      const agent = document.createElement("span");
      agent.className = "orch-agent";
      agent.textContent = r.agent;
      const calls = document.createElement("span");
      calls.className = "orch-calls";
      calls.textContent = (r.calls != null ? r.calls + " calls" : "");
      const detail = document.createElement("span");
      detail.className = "orch-detail";
      detail.textContent = r.detail || "";
      row.appendChild(status);
      row.appendChild(agent);
      row.appendChild(calls);
      row.appendChild(detail);
      body.appendChild(row);
    }
    if (traceData.disputed_claims != null) {
      const d = document.createElement("div");
      d.className = "orch-foot";
      d.textContent = "Verifier: " + traceData.disputed_claims + " disputed claim(s)";
      body.appendChild(d);
    }
  }
  // holds long enough to be read, and then hands control back -- and the "back to
  // the desk" button covers the case where the viewer pauses first. There is no
  // state a viewer can reach from a claim that leaves them with no next move.
  function stepPreview(ts) {
    if (!preview) return;
    if (preview.dwellUntil === 0 && playhead > preview.targetIdx) {
      preview.dwellUntil = ts + PREVIEW_DWELL_MS;
      playing = false;
      updatePlayBtn();
      setStatus("showing the cited moment", true);
    } else if (preview.dwellUntil && ts >= preview.dwellUntil) {
      exitPreview();
    }
  }

  // ---- controls --------------------------------------------------------- //
  function updatePlayBtn() {
    el("playpause").textContent = playing ? "❚❚" : "▶";
    el("playpause").title = playing ? "Pause" : "Play";
  }
  el("playpause").onclick = () => { playing = !playing; updatePlayBtn(); };

  function buildSpeeds() {
    const box = el("speeds");
    box.replaceChildren();
    for (const s of SPEEDS) {
      const b = document.createElement("button");
      b.className = "spd" + (s === speed ? " active" : "");
      b.textContent = s + "×";
      b.onclick = () => { speed = s; buildSpeeds(); };
      box.appendChild(b);
    }
  }

  function buildSeeds() {
    const sel = el("seedsel");
    sel.replaceChildren();
    const opt = (v, t) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = t; sel.appendChild(o);
    };
    opt("random", "Random match");
    for (const [name, s] of CURATED) opt(String(s), name + " · seed " + s);
    sel.value = String(DEFAULT_SEED);   // show the match actually on screen
  }

  // ---- the preview's exits ------------------------------------------------ //
  el("backtodesk").onclick = (e) => { e.stopPropagation(); exitPreview(); };
  // Escape always resolves to something known: leave a preview if one is running,
  // otherwise dismiss the reveal. Whatever the viewer has clicked into, there is a
  // keyboard route back and no state can strand them.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (preview) exitPreview();
    else if (!el("summary").classList.contains("hidden")) el("summary").click();
  });

  // Replay rewinds this match's playback, not the match itself, so the agent
  // trace already received still describes what is on screen. Clearing it left
  // the panel reading "waiting for match to complete…" forever, because the
  // stream has already closed and cannot send the trace again.
  el("replay").onclick = () => resetPlayback();
  // The full-time card is a reveal, not a trap: click the backdrop to go back to
  // the pitch. Replay or a new match will reveal it again.
  el("summary").onclick = (ev) => {
    if (ev.target === el("summary")) {
      el("summary").classList.add("hidden");
      currentReveal = null;
      if (pausedForHt) {
        pausedForHt = false;
        playing = true;
        updatePlayBtn();
        pauseUntil = 0;
      }
    }
  };
  el("newmatch").onclick = () => {
    const typed = el("seedin").value.trim();
    const chosen = el("seedsel").value;
    let s = typed !== "" ? parseInt(typed, 10)
          : chosen === "random" ? Math.floor(Math.random() * 100000)
          : parseInt(chosen, 10);
    if (!Number.isFinite(s) || s < 0) s = Math.floor(Math.random() * 100000);
    connect(s);
  };

  // ---- (re)connect + reset --------------------------------------------- //
  function resetPlayback() {
    playhead = 0; playClock = 0; stateCursor = 0; lastTs = null; pauseUntil = 0;
    lastPossTeam = null; summaryShown = false; playing = true;
    updatePlayBtn();
    marksLayer.replaceChildren();
    activeLayer.replaceChildren();
    feed.innerHTML = '<li class="empty">waiting for the first event…</li>';
    score = {}; if (HOME) { score[HOME] = 0; score[AWAY] = 0; }
    scoreEl.textContent = "0 – 0";
    clockEl.textContent = "0'";
    // Clear the live numbers too, so a replay never shows the previous run's
    // possession / shots until the playhead reaches the first state frame.
    el("pos-h").style.width = "50%";
    el("pos-a").style.width = "50%";
    el("chips").innerHTML = "";
    if (homePct) homePct.textContent = "";
    if (awayPct) awayPct.textContent = "";
    htSummaryShown = false; pausedForHt = false;
    currentReveal = null;
    preview = null; suppressCards = false;
    el("summary").classList.add("hidden");
    el("backtodesk").classList.add("hidden");
  }

  function connect(seed) {
    if (es) es.close();
    events.length = 0;
    stateFrames.length = 0;
    for (const k in overlayByAnchor) delete overlayByAnchor[k];
    summaryData = null; htData = null; doneFlag = false; traceData = null; summaryShown = false; htSummaryShown = false; pausedForHt = false;
    // A new match invalidates the buffered halves. Replay must *keep* them --
    // the EventSource is already closed by then, so there is nothing to re-fetch
    // them from, and clearing them left the half-time card empty on replay.
    htData = null;
    htNarrative = null; ftNarrative = null; currentReveal = null;
    HOME = null; AWAY = null;
    matchup.textContent = "connecting…";
    setStatus("connecting…", false);
    el("strip").classList.add("hidden");
    resetPlayback();
    // A new match has no trace yet: clear the panel now rather than leaving the
    // previous match's agents on screen until this run's `done` frame arrives.
    renderOrchestration();

    es = new EventSource("/events?seed=" + seed);
    es.addEventListener("meta", (msg) => {
      const m = JSON.parse(msg.data);
      HOME = m.home_id; AWAY = m.away_id;
      HOME_NAME = m.home; AWAY_NAME = m.away;
      score = {}; score[HOME] = 0; score[AWAY] = 0;
      setColors();
      matchup.textContent = m.home + " vs " + m.away + " · seed " + seed;
      el("strip").classList.remove("hidden");
      setStatus("streaming " + m.events + " events", true);
    });
    es.addEventListener("event", (msg) => { events.push(JSON.parse(msg.data)); });
    es.addEventListener("overlay", (msg) => {
      // The Explain stage's verdict, buffered by the event it is anchored on.
      // Like the state frames, it is applied when the playhead reaches that
      // point, not when it arrives.
      const p = JSON.parse(msg.data);
      overlayByAnchor[p.insight.anchor_event_id] = p;
    });
    es.addEventListener("state", (msg) => {
      // Pair the frame with how many events had arrived, so playback can apply
      // it at the right point in the timeline rather than on arrival.
      stateFrames.push({ at: events.length, data: JSON.parse(msg.data) });
    });
    es.addEventListener("htsummary", (msg) => { htData = JSON.parse(msg.data); });
    es.addEventListener("summary", (msg) => { summaryData = JSON.parse(msg.data); });
    es.addEventListener("narrative", (msg) => {
      // The desk's reading. It may arrive after the card that shows it (the
      // server generates it while the client is paused at the break), so store it
      // and re-render if the matching card is already on screen.
      const n = JSON.parse(msg.data);
      if (n.reveal === "ht") {
        htNarrative = n;
        if (currentReveal === "ht") renderDesk(n);
      } else {
        ftNarrative = n;
        if (currentReveal === "ft") renderDesk(n);
      }
    });
    es.addEventListener("done", (msg) => {
      try { traceData = JSON.parse(msg.data); renderOrchestration(); } catch(e) {}
      doneFlag = true;
      if (es) es.close();
    });
    es.onerror = () => setStatus("reconnecting…", false);
  }

  // ---- the full-time reveal -------------------------------------------- //
  function renderHtSummary(s) {
    el("ft-home").textContent = s.home_name;
    el("ft-away").textContent = s.away_name;
    el("ft-score").textContent = s.score[HOME] + " – " + s.score[AWAY];
    const rows = el("ft-rows");
    rows.innerHTML = "";
    const pct = (v) => (v * 100).toFixed(0) + "%";
    const num = (v) => String(v);
    const dec = (v) => v.toFixed(2);
    const row = (label, hv, av, fmt) => {
      const total = hv + av;
      const homeShare = total > 0 ? (hv / total * 100) : 50;
      const d = document.createElement("div");
      d.className = "row";
      d.innerHTML =
        '<span class="hv">' + fmt(hv) + "</span>" +
        '<span class="lbl">' + label + "</span>" +
        '<span class="av">' + fmt(av) + "</span>" +
        '<div class="bar"><i style="width:' + homeShare + '%"></i></div>';
      rows.appendChild(d);
    };
    const H = HOME, A = AWAY;
    row("Possession", s.possession[H], s.possession[A], pct);
    row("Shots", s.shots[H], s.shots[A], num);
    row("On target", s.shots_on_target[H], s.shots_on_target[A], num);
    row("In the box", s.shots_in_box[H], s.shots_in_box[A], num);
    row("Woodwork", s.woodwork[H], s.woodwork[A], num);
    row("Saves", s.saves[H], s.saves[A], num);
    row("Blocks", s.blocks[H], s.blocks[A], num);
    row("Expected goals", s.xg[H], s.xg[A], dec);
    row("Passes", s.passes[H], s.passes[A], num);
    row("Pass accuracy", s.pass_completion[H], s.pass_completion[A], pct);
    row("Corners", s.corners[H], s.corners[A], num);
    row("Offsides", s.offsides[H], s.offsides[A], num);
    row("Fouls", s.fouls[H], s.fouls[A], num);
    row("Yellow cards", s.yellows[H], s.yellows[A], num);
    row("Red cards", s.reds[H], s.reds[A], num);
    row("Substitutions", s.substitutions[H], s.substitutions[A], num);
    const at = s.added_time_s || {};
    const mmss = (x) => Math.floor(x / 60) + ":" + String(x % 60).padStart(2, "0");
    let foot = "contested ball " + s.contested_pct.toFixed(1) + "%";
    if (at["1"] != null) foot += " · added time " + mmss(at["1"]) + " + " + mmss(at["2"] || 0);
    el("ft-foot").textContent = foot;
    el("ft-sub").textContent = "Half time";
    currentReveal = "ht";
    renderDesk(htNarrative);
  }

  function renderSummary(s) {
    el("ft-home").textContent = s.home_name;
    el("ft-away").textContent = s.away_name;
    el("ft-score").textContent = s.score[HOME] + " – " + s.score[AWAY];

    const rows = el("ft-rows");
    rows.innerHTML = "";
    const pct = (v) => (v * 100).toFixed(0) + "%";
    const num = (v) => String(v);
    const dec = (v) => v.toFixed(2);
    const row = (label, hv, av, fmt) => {
      const total = hv + av;
      const homeShare = total > 0 ? (hv / total * 100) : 50;
      const d = document.createElement("div");
      d.className = "row";
      d.innerHTML =
        '<span class="hv">' + fmt(hv) + "</span>" +
        '<span class="lbl">' + label + "</span>" +
        '<span class="av">' + fmt(av) + "</span>" +
        '<div class="bar"><i style="width:' + homeShare + '%"></i></div>';
      rows.appendChild(d);
    };

    const H = HOME, A = AWAY;
    row("Possession", s.possession[H], s.possession[A], pct);
    row("Shots", s.shots[H], s.shots[A], num);
    row("On target", s.shots_on_target[H], s.shots_on_target[A], num);
    row("In the box", s.shots_in_box[H], s.shots_in_box[A], num);
    row("Woodwork", s.woodwork[H], s.woodwork[A], num);
    row("Saves", s.saves[H], s.saves[A], num);
    row("Blocks", s.blocks[H], s.blocks[A], num);
    row("Expected goals", s.xg[H], s.xg[A], dec);
    row("Passes", s.passes[H], s.passes[A], num);
    row("Pass accuracy", s.pass_completion[H], s.pass_completion[A], pct);
    row("Corners", s.corners[H], s.corners[A], num);
    row("Offsides", s.offsides[H], s.offsides[A], num);
    row("Fouls", s.fouls[H], s.fouls[A], num);
    row("Yellow cards", s.yellows[H], s.yellows[A], num);
    row("Red cards", s.reds[H], s.reds[A], num);
    row("Substitutions", s.substitutions[H], s.substitutions[A], num);

    const at = s.added_time_s || {};
    const mmss = (x) => Math.floor(x / 60) + ":" + String(x % 60).padStart(2, "0");
    let foot = "contested ball " + s.contested_pct.toFixed(1) + "%";
    if (at["1"] != null) foot += " · added time " + mmss(at["1"]) + " + " + mmss(at["2"] || 0);
    el("ft-foot").textContent = foot;
    el("ft-sub").textContent = "Full time";
    currentReveal = "ft";
    renderDesk(ftNarrative);
  }

  // ---- boot ------------------------------------------------------------- //
  drawPitch();
  buildPitchStats();
  buildLensBar();
  buildSpeeds();
  buildSeeds();
  updatePlayBtn();
  connect(DEFAULT_SEED);        // the comeback leads; the seed picker offers more
  requestAnimationFrame(frame);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the single-page analysis console."""
    return INDEX_HTML
