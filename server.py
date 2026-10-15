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
Six kinds of frame ride the one long-lived connection:

    meta     -- match identity, once, at the start
    event    -- the raw observable event stream (the Ingest layer)
    overlay  -- a timed key-moment payload, whenever the Explain stage finds one
    state    -- a periodic MatchState, rebuilt from observables by metrics.py
    summary  -- the full-time MatchSummary, just before the stream closes
    done     -- terminal marker

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

Still absent: the narrator. The discipline already holds, though: every frame is
built from observables, and latent ground truth never crosses the boundary --
`explainer.py`, like `metrics.py`, cannot import the simulator.

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
from collections.abc import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from explainer import MatchExplainer
from overlay import payload_for
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
# the live numbers move, sparse enough not to flood the stream. The client pairs
# each frame with the number of events it had received, so it can replay the
# numbers in sync with playback instead of on arrival.
STATE_EVERY_N_EVENTS = 40


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
    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"LIVE-{seed}", seed=seed)
    sim.simulate(live=True)  # adds kickoff / half-time / full-time markers
    home_id, away_id = sim.home.team_id, sim.away.team_id

    explainer = MatchExplainer(
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
            }
        ),
    )

    ht_emitted = False
    for i, event in enumerate(sim.observable_stream()):
        # Only the OBSERVABLE stream is sent. The latent ground truth (control,
        # chaos, pass intent, big-chance/error labels) never crosses this
        # boundary -- enforced by the fact that `observable_stream()` cannot
        # yield it, and neither `metrics.py` nor `explainer.py` imports the
        # simulator to reach it.
        insight = explainer.update(event)
        yield _sse("event", event.model_dump_json())

        # A key moment, explained with citations, as a timed overlay payload. The
        # client buffers these with the events and shows each one when its own
        # playhead reaches it, so the overlay stays in step with playback.
        if insight is not None:
            yield _sse("overlay", payload_for(insight).model_dump_json())

        if (i + 1) % STATE_EVERY_N_EVENTS == 0:
            yield _sse("state", explainer.state().model_dump_json())
        if not ht_emitted and event.type.value == "half_time":
            yield _sse("htsummary", explainer.summary().model_dump_json())
            ht_emitted = True

    # Full time: the aggregate the summary screen renders.
    yield _sse("summary", explainer.summary().model_dump_json())
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
  let summaryData = null, htData = null, doneFlag = false, summaryShown = false, htSummaryShown = false, pausedForHt = false;
  let lastPossTeam = null;   // for detecting possession-change moments
  let es = null;

  const SPEEDS = [8, 15, 30, 60, 120, 300];
  const AUTO_PAUSE_MS = 1800;   // linger on a goal / chance so the eye can land

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
    // no pause here -- reaching the last event reveals the summary.

    if (e.type === "half_time" && !htSummaryShown) {
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
      playClock += dt * speed;
      advance(ts);
    }
    maybeReveal();
    requestAnimationFrame(frame);
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

  el("replay").onclick = () => resetPlayback();
  // The full-time card is a reveal, not a trap: click the backdrop to go back to
  // the pitch. Replay or a new match will reveal it again.
  el("summary").onclick = (ev) => {
    if (ev.target === el("summary")) {
      el("summary").classList.add("hidden");
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
    el("summary").classList.add("hidden");
  }

  function connect(seed) {
    if (es) es.close();
    events.length = 0;
    stateFrames.length = 0;
    for (const k in overlayByAnchor) delete overlayByAnchor[k];
    summaryData = null; doneFlag = false;
    // A new match invalidates the buffered halves. Replay must *keep* them --
    // the EventSource is already closed by then, so there is nothing to re-fetch
    // them from, and clearing them left the half-time card empty on replay.
    htData = null;
    HOME = null; AWAY = null;
    matchup.textContent = "connecting…";
    setStatus("connecting…", false);
    el("strip").classList.add("hidden");
    resetPlayback();

    es = new EventSource("/events?seed=" + seed);
    es.addEventListener("meta", (msg) => {
      const m = JSON.parse(msg.data);
      HOME = m.home_id; AWAY = m.away_id;
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
    es.addEventListener("done", () => { doneFlag = true; if (es) es.close(); });
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
