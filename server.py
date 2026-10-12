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

WHAT IT STREAMS
---------------
Five kinds of frame ride the one long-lived connection:

    meta     -- match identity, once, at the start
    event    -- the raw observable event stream (the Ingest layer)
    state    -- a periodic MatchState, rebuilt from observables by metrics.py
    summary  -- the full-time MatchSummary, just before the stream closes
    done     -- terminal marker

Still absent: the explainer, the narrator, the overlay graphics. Those finish
Phase 1. The discipline already holds, though: every frame is built from
observables, and latent ground truth never crosses the boundary -- the event
stream physically cannot carry it, and metrics.py never imports the simulator
to reach it.

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

from metrics import MatchMetrics
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


# How often a MatchState snapshot is pushed, in events: responsive enough that
# the live numbers move, sparse enough not to flood the stream. The browser only
# ever renders the latest one.
STATE_EVERY_N_EVENTS = 40


async def _stream_match(seed: int, speed: float) -> AsyncIterator[str]:
    """Simulate one match, replay its events, and stream live meaning alongside.

    The whole match is generated first (the simulator is synchronous and
    deterministic), then replayed. Pacing is driven by the *match clock*, not by
    a fixed per-event delay: a 90-minute game at `speed=60` plays back in about
    90 seconds, quiet passages wait and goal-mouth scrambles fire in a burst --
    exactly as they would on a broadcast feed.

    Three layers ride the one connection: the raw `event` stream, a periodic
    `state` (the Interpret stage, updated incrementally), and a final `summary`.
    The `MatchMetrics` instance is fed exactly the events we send -- the same
    bytes the browser receives -- so it can only ever know what the viewer knows.
    """
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"LIVE-{seed}", seed=seed)
    sim.simulate(live=True)  # adds kickoff / half-time / full-time markers
    home_id, away_id = sim.home.team_id, sim.away.team_id

    metrics = MatchMetrics(
        sim.match_id,
        home_id,
        away_id,
        home_name=sim.home.name,
        away_name=sim.away.name,
    )

    yield _sse(
        "meta",
        json.dumps(
            {
                "match_id": sim.match_id,
                "home": sim.home.name,
                "away": sim.away.name,
                "home_id": home_id,
                "away_id": away_id,
                "events": len(sim.snapshots),
                "speed": speed,
            }
        ),
    )

    last_clock_s = 0
    for i, event in enumerate(sim.observable_stream()):
        # Only the OBSERVABLE stream is sent. The latent ground truth (control,
        # chaos, pass intent, big-chance/error labels) never crosses this
        # boundary -- enforced by the fact that `observable_stream()` cannot
        # yield it, and `metrics.py` never imports the simulator to reach it.
        metrics.update(event)

        delta_s = max(0.0, event.clock_s - last_clock_s)
        last_clock_s = event.clock_s
        await asyncio.sleep(min(delta_s / speed, 0.5))
        yield _sse("event", event.model_dump_json())

        if (i + 1) % STATE_EVERY_N_EVENTS == 0:
            yield _sse("state", metrics.state().model_dump_json())

    # Full time: the aggregate the summary screen renders.
    yield _sse("summary", metrics.summary().model_dump_json())
    yield _sse(
        "done",
        json.dumps(
            {
                "score": {
                    "home": sim.score[home_id],
                    "away": sim.score[away_id],
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
# pixels. When the analyst console and overlay renderer arrive the pitch panels
# graduate to real templates / static assets.
#
# Four surfaces, top to bottom, exactly as designed: the pitch (an action map
# with switchable lenses), the live numbers, the event ticker, and -- revealed
# at the whistle rather than browsed -- the full-time summary, which is the
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

    /* The live strip: the same numbers the pitch lenses will draw in Step 4. */
    #strip { display: flex; gap: 1.2rem; align-items: center; margin: .8rem 0 .5rem; flex-wrap: wrap; }
    .posbar {
      flex: 0 0 220px; height: 10px; display: flex; border-radius: 5px;
      overflow: hidden; background: #1a2430;
    }
    .posbar .h { background: var(--home); }
    .posbar .a { background: var(--away); }
    .chips { display: flex; gap: 1.1rem; color: var(--mut); font-size: .82rem; flex-wrap: wrap; }
    .chips b { color: var(--ink); font-weight: 600; }

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
    #pitchsvg { display: block; width: 100%; max-height: 58vh; border-radius: 8px; }
    .pitchhint { color: #56636f; font-size: .74rem; margin-top: .35rem; }
    /* The actor disc fades -- and ONLY fades. It never glides between events:
       animating position would be inventing motion the data does not contain. */
    .disc { animation: discfade 1.5s ease-out forwards; }
    @keyframes discfade { 0%, 55% { opacity: 1; } 100% { opacity: 0; } }

    ol { list-style: none; margin: 0; padding: 0; }
    li {
      display: grid; grid-template-columns: 4rem 9rem 4rem 1fr auto;
      gap: .75rem; padding: .2rem .4rem; border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }
    li.goal { background: #12261a; }
    .t { color: #8ea2b5; }
    .type { color: #9fd0ff; }
    .team { color: #c9d4df; }
    .out { color: var(--mut); }
    .extra { color: #b7a3ff; }
    .empty { color: #56636f; padding: 1rem .4rem; }

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
    <div class="posbar" title="possession (rolling window)">
      <span class="h" id="pos-h"></span><span class="a" id="pos-a"></span>
    </div>
    <div class="chips" id="chips"></div>
  </section>

  <section id="pitch">
    <div class="lensbar" id="lensbar"></div>
    <svg id="pitchsvg" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet">
      <g id="pitchlines"></g>
      <g id="marks"></g>
      <g id="discs"></g>
    </svg>
    <div class="pitchhint" id="pitchhint"></div>
  </section>

  <ol id="feed"><li class="empty">waiting for the first event…</li></ol>

  <div id="summary" class="overlay hidden">
    <div class="card">
      <div class="ft-head">
        <div class="ft-teams">
          <span id="ft-home"></span>
          <span class="ft-score" id="ft-score">0 – 0</span>
          <span id="ft-away"></span>
        </div>
        <div class="ft-sub">Full time</div>
      </div>
      <div id="ft-rows"></div>
      <div class="ft-foot" id="ft-foot"></div>
    </div>
  </div>

<script>
  const el = (id) => document.getElementById(id);
  const feed = el("feed"), status = el("status"), matchup = el("matchup");
  const scoreEl = el("score"), clockEl = el("clock");
  const MAX_ROWS = 250;   // keep the DOM bounded on a long stream

  // Team ids arrive in the `meta` frame, so the renderer never hard-codes them.
  let HOME = null, AWAY = null, score = {};

  function setStatus(text, up) {
    status.textContent = text;
    status.className = up ? "up" : "down";
  }
  const clock = (e) => e.minute + (e.stoppage_minute ? "+" + e.stoppage_minute : "");

  // ------------------------------------------------------------------ //
  // The pitch -- an honest action map.
  //
  // Coordinates are the simulator's fixed frame: x=0 is the home goal-line,
  // x=100 the away goal-line, y=0/100 the touchlines. That frame represents a
  // real 105m x 68m pitch, so we draw it in metres and keep the geometry honest.
  //
  // Two rules are load-bearing, and both are the point of the project:
  //   1. Every mark is exactly one real event. Nothing is interpolated.
  //   2. The actor disc appears at the actor's TRUE (x, y) and only fades --
  //      it never glides. Faking motion would be inventing data.
  // ------------------------------------------------------------------ //
  const NS = "http://www.w3.org/2000/svg";
  const sx = (x) => (x / 100) * 105;
  const sy = (y) => (y / 100) * 68;
  const mk = (tag, attrs) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  const marksLayer = el("marks");
  const discsLayer = el("discs");

  const TEAM_COLOR = {};   // filled from the meta frame
  const passColor = (e) => (e.outcome === "complete" ? "#4ade80" : "#f87171");
  const shotColor = (e) =>
    e.outcome === "goal" ? "#4ade80" :
    e.outcome === "saved" ? "#fbbf24" :
    e.outcome === "woodwork" ? "#facc15" :
    e.outcome === "blocked" ? "#94a3b8" : "#64748b";

  // A lens is just a filter + style over the SAME event list. Switching is
  // client-side, so the server never has to know what the viewer is looking at.
  const LENSES = {
    action: {
      label: "Action",
      hint: "every event · lines for passes and shots · disc = the actor, at their true position",
      dot: (e) => ({ r: e.type === "shot" ? 1.0 : 0.7, fill: TEAM_COLOR[e.team], opacity: 0.8 }),
      line: (e) => (e.end_x != null
        ? { x2: e.end_x, y2: e.end_y, stroke: TEAM_COLOR[e.team], width: 0.22, opacity: 0.45 }
        : null),
    },
    passing: {
      label: "Passing",
      hint: "every pass · green complete, red incomplete · thicker line = harder pass",
      dot: (e) => (e.type === "pass" ? { r: 0.55, fill: passColor(e), opacity: 0.9 } : null),
      line: (e) => (e.type === "pass" && e.end_x != null
        ? { x2: e.end_x, y2: e.end_y, stroke: passColor(e),
            width: 0.18 + 0.5 * ((e.pass_difficulty || 0) / 100), opacity: 0.7 }
        : null),
    },
    possession: {
      label: "Possession",
      hint: "where each team's play happens (territory) · one dot per real event",
      dot: (e) => ({ r: 0.55, fill: TEAM_COLOR[e.team], opacity: 0.32 }),
      line: () => null,
    },
    shots: {
      label: "Shots & xG",
      hint: "every shot · radius grows with xG · colour = outcome",
      dot: (e) => (e.type === "shot"
        ? { r: 0.8 + 3.4 * Math.sqrt(e.shot_quality || 0), fill: shotColor(e), opacity: 0.85 }
        : null),
      line: (e) => (e.type === "shot" && e.end_x != null
        ? { x2: e.end_x, y2: e.end_y, stroke: "#cbd5e1", width: 0.18, opacity: 0.35 }
        : null),
    },
  };
  let currentLens = "action";
  const events = [];   // the accumulated stream, kept client-side

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

  function drawEvent(e) {
    const lens = LENSES[currentLens];
    const ln = lens.line(e);
    if (ln) {
      marksLayer.appendChild(mk("line", {
        x1: sx(e.x), y1: sy(e.y), x2: sx(ln.x2), y2: sy(ln.y2),
        stroke: ln.stroke, "stroke-width": ln.width, opacity: ln.opacity,
        "stroke-linecap": "round",
      }));
    }
    const d = lens.dot(e);
    if (d) {
      marksLayer.appendChild(mk("circle", { cx: sx(e.x), cy: sy(e.y), r: d.r, fill: d.fill, opacity: d.opacity }));
    }
  }

  function showDisc(e) {
    const g = mk("g", {});
    g.setAttribute("class", "disc");
    g.appendChild(mk("circle", { cx: sx(e.x), cy: sy(e.y), r: 2.1,
                                 fill: TEAM_COLOR[e.team], stroke: "#0b0f14", "stroke-width": 0.25 }));
    const t = mk("text", { x: sx(e.x), y: sy(e.y), "text-anchor": "middle",
                           "dominant-baseline": "central", "font-size": 2.5,
                           "font-weight": 700, fill: "#0b0f14" });
    t.textContent = e.player_number;
    g.appendChild(t);
    discsLayer.appendChild(g);
    g.addEventListener("animationend", () => g.remove());
  }

  function renderAll() {
    marksLayer.replaceChildren();
    for (const e of events) drawEvent(e);
    el("pitchhint").textContent = LENSES[currentLens].hint;
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

  drawPitch();
  buildLensBar();
  renderAll();

  // EventSource is the browser half of SSE: it reconnects on its own if the
  // connection drops, which is why SSE suits a live feed better than polling.
  const es = new EventSource("/events?seed=42&speed=60");

  es.addEventListener("meta", (msg) => {
    const m = JSON.parse(msg.data);
    HOME = m.home_id; AWAY = m.away_id;
    score = {}; score[HOME] = 0; score[AWAY] = 0;
    TEAM_COLOR[HOME] = "#3b82f6"; TEAM_COLOR[AWAY] = "#f59e0b";
    matchup.textContent = m.home + " vs " + m.away;
    el("strip").classList.remove("hidden");
    setStatus("streaming " + m.events + " events", true);
  });

  es.addEventListener("event", (msg) => {
    const e = JSON.parse(msg.data);
    const first = feed.querySelector(".empty");
    if (first) first.remove();

    const li = document.createElement("li");
    if (e.type === "shot" && e.outcome === "goal") li.className = "goal";

    let extra = "";
    if (e.shot_quality != null) extra = "xG " + e.shot_quality.toFixed(2);
    else if (e.in_box === true) extra = "in box";

    li.innerHTML =
      '<span class="t">' + clock(e) + "'</span>" +
      '<span class="type">' + e.type + "</span>" +
      '<span class="team">' + e.team + "</span>" +
      '<span class="out">' + e.player_name + " · " + e.outcome + "</span>" +
      '<span class="extra">' + extra + "</span>";

    feed.appendChild(li);
    while (feed.children.length > MAX_ROWS) feed.removeChild(feed.firstChild);
    feed.scrollTop = feed.scrollHeight;

    // The pitch: keep the event, draw its mark, and flash the actor disc.
    events.push(e);
    drawEvent(e);
    showDisc(e);

    clockEl.textContent = clock(e) + "'";
    if (e.type === "shot" && e.outcome === "goal") {
      score[e.team] = (score[e.team] || 0) + 1;
      scoreEl.textContent = score[HOME] + " – " + score[AWAY];
    }
  });

  // A rolling MatchState from metrics.py -- the live numbers, not the ticker.
  es.addEventListener("state", (msg) => {
    const s = JSON.parse(msg.data);
    const ph = s.control[HOME], pa = s.control[AWAY];
    el("pos-h").style.width = (ph * 100) + "%";
    el("pos-a").style.width = (pa * 100) + "%";
    el("chips").innerHTML =
      "poss <b>" + (ph * 100).toFixed(0) + "%</b> · " +
      "shots <b>" + s.shots[HOME] + "–" + s.shots[AWAY] + "</b> · " +
      "xG <b>" + s.shot_quality_total[HOME].toFixed(2) + "–" +
      s.shot_quality_total[AWAY].toFixed(2) + "</b> · " +
      "tempo <b>" + s.tempo.toFixed(0) + "</b>/min · " +
      "chaos <b>" + s.chaos.toFixed(2) + "</b>";
  });

  // The full-time reveal: the MatchSummary, rendered as a stat screen. Every
  // number here was rebuilt from the observable stream alone.
  es.addEventListener("summary", (msg) => {
    const s = JSON.parse(msg.data);
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
      const homePct = total > 0 ? (hv / total * 100) : 50;
      const d = document.createElement("div");
      d.className = "row";
      d.innerHTML =
        '<span class="hv">' + fmt(hv) + "</span>" +
        '<span class="lbl">' + label + "</span>" +
        '<span class="av">' + fmt(av) + "</span>" +
        '<div class="bar"><i style="width:' + homePct + '%"></i></div>';
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

    const at = s.added_time_s || {};
    const mmss = (x) => Math.floor(x / 60) + ":" + String(x % 60).padStart(2, "0");
    let foot = "contested ball " + s.contested_pct.toFixed(1) + "%";
    if (at["1"] != null) {
      foot += " · added time " + mmss(at["1"]) + " + " + mmss(at["2"] || 0);
    }
    el("ft-foot").textContent = foot;

    el("summary").classList.remove("hidden");
  });

  es.addEventListener("done", () => {
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
    """Serve the single-page analysis console."""
    return INDEX_HTML
