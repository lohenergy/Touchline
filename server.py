"""Touchline's web surface -- the walking skeleton.

WHAT THIS IS
------------
This file is the *composition root* of the web application: the one place where
the pieces of Touchline are assembled behind an HTTP server. Compare it to a
Telegram bot's `main.py`, which imported the handlers and registered them with
the framework. Same job, different framework.

    bot's main.py : import handlers  -> app.add_handler(...)  -> app.run_polling()
    server.py     : import modules   -> @app.get(...) routes   -> uvicorn server:app

The modules it imports (`simulator.py`, later `metrics.py`) never import this
file back. That direction -- entrypoint imports modules, never the reverse -- is
what keeps the dependency graph a tree instead of a knot.

WHY A WALKING SKELETON
----------------------
Right now it streams the RAW observable event stream only. There is no metrics
engine, no explainer, no narration, no graphics: those arrive in Phase 1. The
point of building this *first* is to prove the whole pipeline end to end --
simulator -> server -> browser -- so that every later piece lands inside a
surface that is already known to work. Standing up the hardest integration
(everything that is NOT machine learning) while there is nothing to break.

WHAT IS NEW HERE (the three things worth learning)
--------------------------------------------------
  1. async / await   -- the stream below is an async generator. FastAPI runs it
                        on the event loop, so one worker can hold many open
                        connections.
  2. SSE             -- Server-Sent Events: one long-lived HTTP response that
                        the server keeps pushing text into. The browser's
                        EventSource API turns those frames back into events.
  3. type-driven routes -- FastAPI reads the annotations below (`seed: int`,
                        `-> StreamingResponse`) to parse and validate the request
                        and to serialise the response. The signature IS the API.

Run it:
    ./.venv/bin/python -m uvicorn server:app --reload
Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

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


async def _stream_match(seed: int, speed: float) -> AsyncIterator[str]:
    """Simulate one match and replay its observable events in match-time order.

    The whole match is generated first (the simulator is synchronous and
    deterministic), then replayed. Pacing is driven by the *match clock*, not by
    a fixed per-event delay: a 90-minute game at `speed=60` plays back in about
    90 seconds, quiet passages wait and goal-mouth scrambles fire in a burst --
    exactly as they would on a broadcast feed.
    """
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"LIVE-{seed}", seed=seed)
    sim.simulate(live=True)  # adds kickoff / half-time / full-time markers

    yield _sse(
        "meta",
        json.dumps(
            {
                "match_id": sim.match_id,
                "home": sim.home.name,
                "away": sim.away.name,
                "events": len(sim.snapshots),
                "speed": speed,
            }
        ),
    )

    last_clock_s = 0
    for event in sim.observable_stream():
        # Only the OBSERVABLE stream is sent. The latent ground truth
        # (control, chaos, pass intent, big-chance/error labels) never crosses
        # this boundary -- that is the whole design rule, enforced by the fact
        # that `observable_stream()` simply cannot yield it.
        delta_s = max(0.0, event.clock_s - last_clock_s)
        last_clock_s = event.clock_s
        await asyncio.sleep(min(delta_s / speed, 0.5))
        yield _sse("event", event.model_dump_json())

    yield _sse(
        "done",
        json.dumps(
            {
                "score": {
                    "home": sim.score[sim.home.team_id],
                    "away": sim.score[sim.away.team_id],
                }
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
    speed: float = Query(60.0, gt=0, description="Replay speed; 60 ~ a 90s match."),
) -> StreamingResponse:
    """Stream one synthetic match as Server-Sent Events.

    `StreamingResponse` never buffers the whole body: FastAPI iterates the async
    generator and writes each frame as it is produced, which is exactly what a
    long-lived push stream needs.
    """
    return StreamingResponse(
        _stream_match(seed, speed),
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
# pixels. When the analyst console and overlay renderer arrive in Phase 1 they
# graduate to real templates / static assets.
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Touchline — live event stream</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 1.25rem 1.5rem;
      font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #0b0f14; color: #d7e0ea;
    }
    header { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
    h1 { font-size: 1rem; margin: 0; letter-spacing: .08em; text-transform: uppercase; }
    .meta { color: #7d8b99; }
    .score { color: #f2f5f8; font-weight: 700; }
    #status { margin-left: auto; font-size: .8rem; padding: .1rem .5rem; border-radius: 1rem; }
    .up { background: #16351f; color: #6ee7a0; }
    .down { background: #3a1a1a; color: #f79a9a; }
    ol { list-style: none; margin: 1rem 0 0; padding: 0; }
    li {
      display: grid; grid-template-columns: 4rem 9rem 4rem 1fr auto;
      gap: .75rem; padding: .2rem .4rem; border-bottom: 1px solid #141b23;
      white-space: nowrap;
    }
    li.goal { background: #12261a; }
    .t { color: #8ea2b5; }
    .type { color: #9fd0ff; }
    .team { color: #c9d4df; }
    .out { color: #7d8b99; }
    .extra { color: #b7a3ff; }
    .empty { color: #56636f; padding: 1rem .4rem; }
  </style>
</head>
<body>
  <header>
    <h1>Touchline</h1>
    <span class="meta" id="matchup">connecting…</span>
    <span class="score" id="score"></span>
    <span id="status" class="down">idle</span>
  </header>
  <p class="meta" style="margin:.6rem 0 0">
    Walking skeleton — raw observable events only. No metrics, no narration yet.
  </p>
  <ol id="feed"><li class="empty">waiting for the first event…</li></ol>

<script>
  const feed = document.getElementById("feed");
  const status = document.getElementById("status");
  const matchup = document.getElementById("matchup");
  const score = document.getElementById("score");
  const MAX_ROWS = 250;   // keep the DOM bounded on a long stream

  // EventSource is the browser half of SSE. It reconnects on its own if the
  // connection drops -- which is why SSE suits a live feed better than polling.
  const url = "/events?seed=42&speed=60";
  const es = new EventSource(url);

  function setStatus(text, up) {
    status.textContent = text;
    status.className = up ? "up" : "down";
  }

  es.addEventListener("meta", (msg) => {
    const m = JSON.parse(msg.data);
    matchup.textContent = m.home + " vs " + m.away + "  (seed " + m.match_id + ")";
    setStatus("streaming " + m.events + " events", true);
  });

  es.addEventListener("event", (msg) => {
    const e = JSON.parse(msg.data);
    const first = feed.querySelector(".empty");
    if (first) first.remove();

    const li = document.createElement("li");
    if (e.type === "shot" && e.outcome === "goal") li.className = "goal";

    const clock = e.minute + (e.stoppage_minute ? "+" + e.stoppage_minute : "");
    let extra = "";
    if (e.shot_quality != null) extra = "xG " + e.shot_quality.toFixed(2);
    else if (e.in_box === true) extra = "in box";

    li.innerHTML =
      '<span class="t">' + clock + "'</span>" +
      '<span class="type">' + e.type + "</span>" +
      '<span class="team">' + e.team + "</span>" +
      '<span class="out">' + e.player_name + " · " + e.outcome + "</span>" +
      '<span class="extra">' + extra + "</span>";

    feed.appendChild(li);
    while (feed.children.length > MAX_ROWS) feed.removeChild(feed.firstChild);
    feed.scrollTop = feed.scrollHeight;
  });

  es.addEventListener("done", (msg) => {
    const d = JSON.parse(msg.data);
    score.textContent = d.score.home + " – " + d.score.away;
    setStatus("full time", false);
    es.close();
  });

  es.onerror = () => setStatus("reconnecting…", false);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the single-page event ticker."""
    return INDEX_HTML
