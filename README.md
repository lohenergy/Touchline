# Touchline

Match intelligence that explains *why* a moment mattered — and can prove it.

Built for the **Microsoft Inside the Game: Developer Hackathon** (Oct 2026).

## Demo

▶ **[Watch the 2-minute walkthrough](https://youtu.be/cxZFeOUeVKA)**

A screen recording of a local run: synthetic match data, no deployment, no
network. Things to look for:

- **The provenance chip** on the desk — which model wrote the copy, or the
  deterministic fallback when no provider is configured.
- **A key moment's receipts** — hover the "why" line in the ticker to see the
  event IDs the claim is true because of.
- **"Show me"** — replays the build-up to a cited moment, then hands the viewer
  back to the desk.
- **The orchestration trace** — six agents, their statuses and call counts, and
  the Verifier's disputed-claim count.

---

## The problem

Football data is presented brilliantly and understood poorly.

Official apps and broadcast graphics tell you *what* happened — possession,
shots, distance covered — and stop there. They cannot tell you **why** a moment
mattered, because a leaderboard can only rank one value.

So someone has to do the translation by hand. The person who feels this most is
**the analyst or producer on the desk**: minutes to turn an event stream into a
narrative they will have to defend on air, with no way to check whether the story
is true. The stats football argues about most — *was that a big chance? did that
mistake lead to the goal?* — are not even in the feed; a human labels them by eye.

And interpretation is contested. Every viewer watches with a stake in the result
— the diehard fan sees the decisions go against their team and disputes them —
and the decisions most worth disputing are exactly the ones no feed can settle.
Claims get made; nobody can check them.

The data needed to build a fix is closed to nearly everyone, which is why the
hackathon mandates synthetic, football-realistic data.

## The approach

Touchline turns a live event stream into **grounded** match intelligence. Every
claim is tied to the specific events that produced it — the receipts — and
packaged as timed, machine-readable overlay payloads a broadcaster's graphics
system can consume directly.

The differentiator is **verifiability**. Because the data is synthetic, the
simulator knows the truth behind each moment — including the subjective stats
real feeds lack, like whether a shot was a big chance or a mistake led to a
goal. So the system can *score its own explanations* against a known answer and
ship with a measured error bar, rather than asking anyone to trust them.

The same intelligence is then rendered per audience: a **Narrative** mode tells
the story in plain language; a **Depth** mode supplies the evidence behind a
claim. This is not a promise that bias disappears — it is a guarantee that every
claim can be checked.

---

## Architecture

The pipeline follows the brief's five stages:

| Stage | What happens | Where |
|---|---|---|
| **Ingest** | Synthetic events stream in real time | `simulator.py` |
| **Interpret** | Events → `MatchState` (control, chaos, tempo, shot quality) | `metrics.py` *(Phase 1)* |
| **Explain** | Detect key moments, attribute *why* using cited evidence | `explainer.py` *(Phase 2)* |
| **Render** | Timed overlay payloads consumed by a graphics layer | `overlay.py` *(Phase 2)* |
| **Personalize** | Narrative and Depth audience modes (plus languages) from one intelligence layer | *(Phase 2)* |

## Repository layout

```
schemas.py      Domain contracts: MatchEvent, LatentState, MatchState, MatchSummary,
                Insight, OverlayPayload (+ pass types, stoppage ledger, contested
                possession, save/block/woodwork events, in-box geometry,
                big-chance/error truth)
simulator.py    Synthetic match generator (observable events + latent ground truth)
metrics.py      Interpret stage: observable events -> MatchState / MatchSummary
explainer.py    Explain stage: key moments + cited reasons, scored against truth
overlay.py      Render stage: insights -> timed overlay payloads
llm.py          The only module that opens a socket: provider plans (Azure
                OpenAI / GitHub Models), endpoint normalisation, one JSON-mode
                POST. Fails soft by returning None instead of raising
narrator.py     Narrator stage: the grounded desk -- four gates, plus local and
                LLM providers behind one interface
verifier.py     Verify stage: the critic. Rules each claim supported /
                misattributed / unsupported by the claim's roster id, attaches
                the ruling to the claim, surfaces disputes rather than deleting
                them, and never writes a sentence of its own
pass_classifier.py  Phase-2 classifier: pass intent from geometry alone
agents.py       Agent layer: named roles, declared contracts (reads/writes), one
                shared blackboard, and a trace of who ran and how it went
validate.py     Calibration harness: does the synthetic data look like football?
test_events.py  Invariant tests: is the event stream semantically correct?
test_metrics.py Tests: can the match be rebuilt from observables alone?
test_pass_classifier.py  Does intent recovery actually work?
test_explainer.py        Are the cited claims real, past, and honest?
test_overlay.py          Does every moment render exactly once, in order?
test_llm.py              Do the provider plans resolve, and do they fail soft?
test_narrator.py         Do the four gates hold, and does the desk fail soft?
test_verifier.py         Does a ruling join back to its claim, and does an
                         unreadable ruling leave the claim standing?
test_agents.py           Do the contracts bind, and does a failure stay visible?
server.py       Web surface: FastAPI + SSE console (pitch, live strip, ticker,
                full-time summary) — one long-lived stream, no build step
```

---

## The one design rule that matters most

**The latent state must *cause* the observables, and must never *be* one of them.**

`MatchEvent` contains only what a real provider could observe. `LatentState`
(control, chaos, phase) travels in a separate channel and is opened by exactly
two pieces of code: the trainer and the scorer. If latent state leaked into the
event stream, every model downstream would "predict" a value it was handed — a
failure called *data leakage*, and the most common way an ML project produces
impressive numbers that mean nothing.

There is a second, subtler half to this: a quantity is only learnable if it
leaves a trace in the observables. `control` raises a team's pass completion
rate, which lengthens their sequences, which produces more final-third entries
and more shots. That causal chain is what makes control recoverable from events
alone — and it is why the explanations are non-trivial.

## The Interpret stage: meaning from events alone

`metrics.py` is the layer that turns events into *meaning*. Its one rule: it may
read observable events and nothing else — it does not import the simulator at
all, so it has no way to reach the latent truth. That is the event-contract
discipline pushed one stage downstream: an interpretation must be *recoverable
from what a camera could see*, or it is not an interpretation.

Possession here is a *time* decomposition, not a pass count. Every second between
two events is attributed to home, away, or a contested (loose-ball) phase using
only observable cues, and the three buckets sum to the elapsed clock by
construction (`test_metrics.py` asserts it). The objective record it rebuilds —
score, shots, saves, blocks, woodwork, xG, discipline, added time — equals the
simulator's own counters, which were computed from full internal state rather
than from the stream. Agreement is a real check, not a tautology.

Two quantities make no claim of accuracy: `control` and `chaos`. Phase 1 ships
transparent *baselines* for them — a rolling possession share, and the recent
share of unstructured events — and Phase 2 learns better estimators and scores
them against `control_true` / `chaos_true`. A weak, *measured* baseline is worth
more than a confident, unfalsifiable one.

## The Explain stage: claims you can check, and one you cannot

`explainer.py` picks out the key moments and says *why* each one mattered, with
every reason citing the `event_id`s it is true because of. It wraps `MatchMetrics`
and reads only observable events — like `metrics.py`, it cannot import the
simulator, so it has no code path to the truth. A moment is a structure, not a
sentence: "Chance — J. Smith, struck from 12 m (xG 0.41), built from a 14 m pass
by A. Cole" is three clauses, each bound to events a viewer already saw.

The reasons are **tagged by kind**, and that is a correctness decision, not
cosmetics. The scoring harness selects a claim by `reason.kind == CREATOR`, never
by matching English, so rewording a sentence can never silently break the audit.

Because the data is synthetic, every claim can be *graded*. `explainer.py` scores
its own explanations on a held-out block of 40 matches:

| claim | precision | recall | F1 | base rate |
|---|---|---|---|---|
| "we called it a chance" | 0.96 | 0.92 | **0.94** | 0.080 |
| "we credited the right pass" | 1.00 | 0.95 | 0.97 | 0.849 |
| "a give-away was a real error" | 0.04 | 1.00 | 0.07 | **0.036** |

The chance result is the real one: an observable xG bar recovers a latent label
almost exactly. The creator result is a *conformance* check — the attribution rule
mirrors the simulator's own book-keeping, and across 79 big chances it names the
identical pass 75 times and a *different* pass zero times.

The third row is the interesting one, and it changed the design. It reads
"precision 0.04", which sounds like a broken detector until you compare it to the
base rate: **0.04 against a base rate of 0.036**. When precision sits at the base
rate, the flag carries *no information* — knowing a deep give-away happened tells
you essentially nothing about whether it was a genuine error, because only ~4% of
them are, and the tape offers nothing to separate those from the rest.

So the tempting third claim — "this chance came from that give-away" — was
**removed**. It reads beautifully and it would have been wrong nineteen times in
twenty. The give-away still appears, as its own observable moment; it is never
asserted as the *cause* of a later one. This is the discipline the project exists
to demonstrate: with real data, nobody could have checked that sentence, and it
would have shipped.

## The Render stage: the feed *is* the product

`overlay.py` turns insights into a timed overlay timeline — show this card at
this match-second, hold it this long, anchor it here, cite these events. It is
deliberately the dumbest stage: it decides *when* and *how long*, never *what is
true*. Every word it emits was produced and cited upstream, so an overlay bug can
invent nothing; the worst it can do is show a real claim at the wrong moment,
which is a timing bug and easy to see.

The architecture puts Touchline *upstream* of the graphics layer, so the payload
is renderer-agnostic by construction: a kind, a start time, a duration, an anchor,
and the cited events. Our web page is simply its first consumer, and a broadcast
graphics box would be the next, with nothing else changing. Hold times are a
broadcast judgement, not a derived quantity — a goal earns longer on screen than a
give-away, because the eye needs longer to take in "4–5" than "they lost it 30 m
out" — so they live as three named constants you can argue with.

## The web surface: one stream, four surfaces

`server.py` is the *composition root* — the one place where the pieces are
assembled behind HTTP (a Telegram bot's `main.py`, with routes instead of
handlers). It serves one page and one long-lived SSE stream, and emits six
kinds of frame:

```
meta     match identity, once, at the start
event    the raw observable event stream (the Ingest layer)
overlay  a timed key-moment payload from the Explain stage, with citations
state    a periodic MatchState, rebuilt by metrics.py
summary  the full-time MatchSummary, just before the stream closes
done     terminal marker
```

The page renders three live surfaces — the pitch, the live numbers, and the event
ticker — plus a fourth that is *revealed* rather than browsed: the full-time
summary, which is the metrics engine given a face. Key moments in the ticker now
arrive **pre-explained**: the server sends the claim and the events it cites, and
the page renders that verdict instead of guessing at it. The old client-side
`keyMoment()` heuristic is gone — one definition of "a chance", in one place,
upstream of the graphics.

**The client owns the clock.** The server streams the match as fast as it can
produce it; the page buffers the events and advances a *playhead* at a speed the
viewer chooses, so pause, speed, replay and "new match" need no cooperation from
the server. (This is the fix for the first honest failure of the renderer: at the
old fixed pace a full match flew past at ~19 events/second, so passes were
unreadable. Pacing belongs where the controls are.) Playback auto-pauses on a
goal, a chance, or the half-time whistle so the eye can land on it, then resumes;
reaching full time reveals the summary instead of pausing.

The pitch is an honest action map, drawn to scale (105m × 68m). **Every mark is
one real event**, and nothing is invented between them. The actor disc sits at
the actor's *true* `(x, y)` with their shirt number and **jumps** between events —
it never glides, because animating position between two events would be
manufacturing motion the data does not contain. Lenses are client-side filters
over the same event list, so the server never has to know what the viewer is
looking at:

| Lens | Shows |
|---|---|
| **Action** | every event, with pass/shot lines and the actor disc |
| **Passing** | passes only; green complete / red incomplete; **thickness = distance** (short / medium / long); the current possession sequence stays bright while older play dims |
| **Possession** | where each team's play happens (territory) — one dot per event |
| **Shots & xG** | shots only; the ball marks the strike and grows with xG; a dashed line runs to the target; colour = outcome (goal / saved / woodwork / blocked / off target) |

Width encodes **distance, not difficulty**, deliberately: a stylised width is a
*visible* property of the pass, whereas difficulty is our own opinion and belongs
in a claim the explainer must defend, not in a stroke.

The shot's destination is geometry-honest: a goal or a save finishes inside the
posts, woodwork exactly on a post, a miss beyond it, and a block short of the
line. The first draft sent *every* shot to the byline at the shooter's own
lateral position, which made a wide shot look like it went out of play and a goal
look like a miss. That looked like a rendering bug but was a **data** bug: the
shot endpoint is decided in the simulator, so it was fixed there, where the
outcome is known — the renderer only draws what it is given.

**Key moments are observable only.** The legend names three: a possession change
(yellow), a *chance* — a shot with xG of at least 0.35 (green), and a *deep
giveaway* — an incomplete pass in the passer's own third (red). The simulator's
latent "big chance" and "error leading to a chance" tiers never appear here; they
exist to be scored, and putting them on the analyst surface would be exactly the
leakage the project forbids. What the viewer sees is what a viewer could see.

The chance threshold is not a stylistic choice. It was selected on a disjoint
tuning block as the xG bar that best recovers the simulator's latent
`big_chance_true` label, and it earns its keep: on a held-out block it separates
big chances at **F1 0.94** against an 8% base rate. The earlier rule — "in the box
*or* xG ≥ 0.15" — was replaced because every big chance is already in the box, so
the box term added hundreds of harmless pot-shots and cost ~60 points of
precision for no recall. xG already encodes the geometry.

**The live strip is two-sided wherever the stat is.** Possession is a *share*, so
the strip reads `poss 55% – 45%` — a lone "55%" would read as if one team owned
the ball. The same split is painted faintly (~12% opacity) across the pitch, one
number per half, as a headline you can read at a glance without it competing with
the event marks. Possession here is the time-based figure derived from
observables, so it belongs on the analyst surface. The ticker lists events
**newest first**, so the latest action is always in view without scrolling.

One connection carries everything, and every frame is built from observables.
The `MatchExplainer` in the server — which wraps `MatchMetrics` — is fed exactly
the events it sends, the same bytes the browser receives, so the interpretation
can only ever know what the viewer knows.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python simulator.py        # generate one match and print a summary
python metrics.py          # interpret one match; print the full-time summary
python explainer.py        # score the explanations, then show one match's key moments
python overlay.py          # print one match's timed overlay timeline
python narrator.py 81 --live   # ask the desk for one reveal (needs a provider)
python verifier.py 81          # have the attribution critic rule on that draft
python verifier.py 81 --live   # and again with whatever .env configures
python eval_verifier.py        # score the critic: precision, recall, false positives
python eval_verifier.py --live # the same labelled probes against the model critic
python pass_classifier.py  # recover pass intent from geometry; print the scores
python validate.py 120     # calibration report over 120 matches
python test_events.py      # invariant tests on the event stream
python test_metrics.py     # tests: the match rebuilt from observables alone
python test_pass_classifier.py   # tests: intent recovered, no truth leaked
python test_explainer.py   # tests: cited claims are real, past and honest
python test_overlay.py     # tests: every moment renders once, in order
python test_llm.py         # tests: plans resolve, and fail soft when they cannot
python test_narrator.py    # tests: the four gates, and the soft fallback
python test_verifier.py    # tests: rulings join to claims, doubt keeps claims
python test_agents.py      # tests: contracts bind, failures stay visible

python -m uvicorn server:app --reload   # the console, at http://127.0.0.1:8000
```

`validate.py` exits non-zero if any metric falls outside its reference range, so
it can gate CI. `test_events.py` does the same for the invariants below.

To run the console exactly as the demo does — offline, with the deterministic
desk and the attribution-only critic — set the provider explicitly:

```bash
TOUCHLINE_LLM_PROVIDER=local python -m uvicorn server:app --reload
```

Provider configuration is optional; with no `.env` and no provider variable the
desk and critic fall back to their local implementations, and the page says so
on the provenance chip.

## Two kinds of correctness, and why you need both

`validate.py` checks **marginals** — do the averages look like football? That is
necessary but not sufficient. A model can match every average and still be
structurally absurd. The clearest example is a bug this project actually hit:

> Fouls were attributed to the team *in possession* — but fouls are committed by
> the team *without* the ball, and the fouled side *keeps* possession for the
> free kick. The simulator produced a plausible number of fouls per match the
> whole time. Every aggregate looked healthy. Every discipline and possession
> statistic downstream was quietly wrong.

`test_events.py` checks **invariants** — properties that must hold for every
event in every match, independent of the averages. It asserts that:

- every type declared in `EventType` is actually emitted (a declared-but-dead
  type is a contract the downstream code will code against and never exercise);
- a foul is attributed to the defending team, and possession is retained after
  it;
- a corner is retained by the attacking side — including a corner won from a
  blocked shot, where the possession must *not* flip;
- `card_type` is set on `CARD` events and nowhere else;
- a substitution replaces a genuinely different player;
- no latent field ever rides along inside an observable `MatchEvent`;
- pass intent (`pass_type_true`) exists *only* on the latent channel of `PASS`
  events, never on the events themselves;
- the display clock never exceeds 45 in the first half or 90 in the second —
  added time is reported as `stoppage_minute`, not by inflating `minute`;
- the board's added-time figure equals the sum of its stated causes, so the
  explanation cannot contradict the number it explains;
- the three possession buckets (home / away / contested) sum, every match, to
  the full match clock — a decomposition that silently drops time is worse
  than none;
- each loose-ball contest emits between one and two `LOOSE_BALL` events, so the
  observable count brackets the ground-truth counter exactly;
- every saved shot has exactly one `SAVE` event by the defending keeper, and
  every blocked shot one `BLOCK` event by a defender — the credited player is
  never the shooter;
- a `WOODWORK` outcome exists and is never counted as a shot on target, and the
  observed `in_box` flag always matches the shot's origin geometry;
- the big-chance tier, being latent, appears *only* on shot snapshots, and its
  bounds are consistent with the observable xG to within the bounded noise;
- a pass marked `chance_created_true` is always followed, before any possession
  change, by a big-chance shot from the same team — the creator link can never
  dangle;
- an `error_true` pass is always an `INCOMPLETE` pass taken deep in the
  possessor's own third, and every error-gifted shot has a preceding giveaway to
  walk back to.

That last group is where the invariant layer really earns its keep: these are
the properties that let the explainer make *derivable* claims. If the added-time
ledger did not sum to the announced figure, the system would be telling lies
with a straight face.

If `control_true` ever appeared on a `MatchEvent`, predicting it would be
reading the answer off the input.

## Four mechanisms that make the simulation honest

Four choices in the simulator exist purely so the *explainers* built on top can
be grounded rather than decorative.

**1. Added time is explained, not announced.** Each half accumulates a stoppage
ledger: every goal, substitution and card contributes a known number of seconds,
and the added time is the sum — plus a routine allowance. The `HALF_TIME` /
`FULL_TIME` events carry a `StoppageBreakdown`, so the overlay can show `+4:10
(card 1:00, goal 0:40, sub 1:30, routine 1:00)`. The number and its
justification are the same object.

**2. Possession is a time decomposition, not a pass count.** Newer modelling
sums, every match, to the full clock: home seconds, away seconds, and time the
ball was *contested* (a loose ball both sides fought for). Contest frequency is
driven by latent `chaos`, and it leaves an observable trace — `LOOSE_BALL`
events — so a downstream metric engine can recover it rather than admire it. A
pass-count possession proxy cannot express "nobody had it", which is precisely
the moment the broadcast story peaks.

**3. Pass intent is ground truth.** `_choose_pass` decides *why* a ball is
played — short, long, through, cross — and that intent rides only in the latent
channel. Phase 2 builds a geometric classifier that must recover it from
`x/y/end_x/end_y` alone and scores itself against the label. It is a miniature
of the whole project: the machine must infer intent from observable traces, and
we get to measure how well it does — with real data you never could.

**4. The analyst's adjectives are ground truth.** Two stats that real football
analysts mark by eye — *big chances* and *errors leading to chances* — are
produced by the simulator instead, in the latent channel. A shot's chance tier
is decided on a latent copy of its quality (bounded noise the observable xG does
not carry, so the tier is learnable without being trivial); an error is a failed
pass deep in the possessor's own third that hands the opposition a *boosted*
chance. Both leave only an observable trace — the shot's geometry, the giveaway
— for a downstream classifier to recover and score against the label. Same story
as pass intent: a stat that is subjective in real football becomes measurable
here, and we can say how well we measure it.

## Calibration status

Synthetic aggregates over 120 matches, against published reference ranges for
professional league football:

| Metric | Synthetic | Reference |
|---|---|---|
| Goals per match | 3.04 | 2.4–3.2 |
| Shots per match | 29.7 | 20–30 |
| Shots inside the box | 64% | 55–75% |
| Woodwork (both teams) | 0.77 | 0.5–2.0 |
| Saves per match | 7.14 | 4–9 |
| Blocks per match | 9.65 | 4–12 |
| Big chances per match | 2.51 | 2–6 |
| Errors leading to chances | 1.59 | 0.5–4 |
| Shots on target | 10.2 | 7–11 |
| xG per match | 2.97 | 2.2–3.4 |
| xG per shot | 0.10 | 0.08–0.14 |
| Passes per match | 1074 | 800–1200 |
| Pass completion | 84.5% | 78–88% |
| Home possession *of controlled time* | 50.4% | 42–58% |
| Contested possession | 7.5% | 3–18% |
| Loose-ball phases | 73 | — |
| Added time, first half | 3:02 | — |
| Added time, second half | 4:53 | — |
| Added time, total | 7:56 | 2–16 min |
| Corners per match | 10.5 | 8–12 |
| Offsides per match | 3.5 | 1.5–4.5 |
| Fouls per match | 24.1 | 18–28 |
| Yellow cards | 4.1 | 2–6 |
| Red cards | 0.32 | 0–0.5 |
| Substitutions | 4.9 | 4–10 |

Matching aggregate marginals is **necessary but not sufficient** — data can match
every average and still be structurally absurd. The invariant tests above cover
the semantic layer; sequence structure, correlation and game-state effects are
validated separately in later layers.

> The reference ranges are commonly cited figures that still need to be properly
> sourced and cited before submission. An unsourced number in a validation report
> is worse than no report: it looks rigorous while being unfalsifiable. The
> contested-possession and added-time ranges are *design targets* set from the
> simulator's own causal structure; they are the ones most in need of a real
> source before submission. The Phase-0.6 rows — shots inside the box, woodwork,
> saves, blocks, big chances and errors — are likewise provisional and need
> sourcing (Opta/Statsbomb community figures for the first four; the last two are
> inherently *labelled by our simulator*, so their ranges are design targets, not
> league statistics).

## Status

Phase 0 complete, and Phase 0.6 (the stat bundle) with it:

- **Phase 0**: contracts and simulator, including the completed event stream
  (offsides, corners, cards, substitutions), stoppage time with an explainable
  ledger, three-way possession with contested time, and pass-intent ground
  truth.
- **Phase 0.6**: the objective trio (shots inside the box, woodwork, attributed
  saves and blocks) riding as observable events, and the two subjective stats
  made measurable — *big chances* and *errors leading to chances* — carried as
  latent ground truth that downstream classifiers will be scored against.
  21 invariant tests and 25 calibrated metrics green.
- **Phase 1 (started)**: the Interpret stage (`metrics.py`) rebuilds a match's
  objective record from observable events alone — score, shots, saves, blocks,
  woodwork, xG, passes, discipline, added time, and a *time-based* possession
  decomposition with contested time. `test_metrics.py` checks the reconstruction
  against the simulator's own counters and asserts the possession buckets sum to
  the clock. `control` and `chaos` ship as transparent baselines, to be learned
  and scored against `control_true` / `chaos_true` in Phase 2.
- **Phase 1 (web surface)**: `server.py` streams one SSE connection carrying
  `meta` / `event` / `state` / `summary` / `done` (Phase 2 adds the `overlay`
  frame). The page renders the event
  ticker, a live numbers strip from periodic `state` frames, and — revealed at
  full time — the `summary`, which is the metrics engine given a face.
- **Phase 1 (pitch renderer)**: the action map, drawn to scale, with four
  client-side lenses (Action, Passing, Possession, Shots & xG) and numbered actor
  discs at the actor's true position that jump rather than glide. Shirt numbers
  are a new observable on `MatchEvent` (a camera can read them), guarded by an
  invariant test.
- **Phase 1 (playback + key moments)**: the client owns the clock — play/pause,
  speed presets, replay, "new match" and six curated seed presets, with auto-pause
  on goals, chances and the half-time whistle. Shots became outcome-aware in the
  simulator (goal / save inside the posts, woodwork on a post, miss beyond it,
  block short of the line; verified geometrically over 30 matches). Passes gained
  a distance-encoded thickness and a bright current-possession sequence; key
  moments (possession change, chance, deep giveaway) got ticker highlighting and a
  legend. All observable-only.
- **Phase 1 (surface polish)**: possession is shown two-sided in the strip and
  painted faintly on the pitch, one number per half; the ticker runs newest-first;
  the full-time card gained a substitutions row; replay now clears the live
  numbers rather than briefly showing the previous run's. Presentation only — no
  new quantity reaches the analyst surface.
- **Phase 2 (pass-intent classifier)**: `pass_classifier.py` recovers pass intent
  (short / long / through / cross) from the pass geometry alone. The
  attacking-local frame is rebuilt from the event's `team` and the home id, and
  four thresholds were selected on a disjoint tuning block *before* scoring. On a
  held-out block of 40 matches it reaches **91.0%** (95% CI 90.7–91.3) against a
  64.8% majority baseline; per-class F1 is 0.93 short, 1.00 long, 0.82 through,
  0.73 cross. `test_pass_classifier.py` asserts the module cannot import the
  simulator (no truth channel) and that every class is genuinely recovered. The
  residual error is explainable rather than hidden: near the byline the
  simulator's own endpoint clamp erases the direction of travel, and a cross is
  genuinely hard to separate from wide progression.
- **Phase 2 (explainer)**: `explainer.py` is the Explain stage. It recognises the
  observable key moments (goal, chance, deep give-away) and attributes *why*, with
  every reason citing the `event_id`s it is true because of — shot origin and xG,
  and the completed pass that built it. The citations are not decoration:
  `test_explainer.py` walks every citation across 15 matches and asserts it names a
  real event that does not postdate the moment, so a claim can always be audited
  against the tape. It wraps `MatchMetrics`, so the server drives Interpret and
  Explain from one event feed.
- **Phase 2 (explainer scoring)**: because the data is synthetic, the explainer
  *grades its own explanations*. On a held-out block of 40 matches the chance model
  recovers the latent big-chance label at **F1 0.94** against an 8% base rate, and
  the creator attribution names the identical pass as the simulator on 75 of 79 big
  chances and a different one zero times. The third candidate claim — that a chance
  came from a give-away — was *measured and removed*: at precision 0.04 against a
  0.036 base rate the flag carries no information, so it would have been wrong
  nineteen times in twenty and is never asserted. Instead the give-away is surfaced
  as its own observable moment, never as the cause of a later one, and
  `ReasonKind` has no `CAUSE` member.
- **Phase 2 (overlay)**: `overlay.py` turns insights into a timed overlay feed —
  start time, hold time, anchor, cited events — sorted in clock order, with a
  full-time summary payload that holds until dismissed. `test_overlay.py` pins the
  contract a graphics layer depends on: every insight renders exactly once, at its
  own clock, and the summary is last.
- **Phase 2 (wired into the web surface)**: `server.py` now emits an `overlay`
  frame per key moment alongside the event stream, and the page renders the
  server's cited reasons in the ticker (hover a clause to see the events it cites).
  The client-side `keyMoment()` heuristic was deleted — one definition of "a
  chance", upstream of the graphics, is the whole architectural point. A
  pre-existing bug was found and fixed in the process: on **replay** the half-time
  card rendered empty, because `resetPlayback` discarded the buffered half that the
  closed `EventSource` could no longer re-send. It is now cleared on a *new match*
  and kept on replay, like the full-time summary.

- **Phase 2 (Verifier, second pass)**: `verifier.py` reviews the desk's draft
  without touching it. `apply_verdicts` attaches each ruling to the claim it
  addresses, addressed by the claim's roster id, so a disputed sentence stays on
  screen next to the fact the critic cited against it. `disputed_claims` counts
  them; nothing is deleted. `test_verifier.py` pins the join (a ruling reaches
  its claim, never a lookalike) and the failure mode (an unreadable review leaves
  every claim standing and marks the reveal unverified).
- **Phase 2 (critic evaluation harness)**: `eval_verifier.py` scores the critic
  against a labelled set that includes untouched true claims as false-positive
  probes and corrupted claims (team swaps, wrong numbers, wrong players, wrong
  distances) as detection probes. The local critic is attribution-only, so it
  catches contradictions it can see in the cited events but not every number or
  name mutation; run it with `--live` to score the configured model critic
  instead. Reporting precision separately from the false-positive rate is the
  point: a critic that never objects would otherwise look perfect.
- **Phase 2 (orchestration trace)**: the `done` frame carries the agent trace,
  and the console renders it in a footer panel — one row per agent with status,
  call count and a detail line, plus the Verifier's disputed-claim count. The
  panel is observability, not a second control surface: it reports what ran, in
  what order, and what it produced.

## The Verifier: criticize, don't rewrite

`verifier.py` is the critic stage. A second pass — never chained to the Desk's
write — checks claims against the same observable evidence the desk was given,
and attaches rulings by structural position (via `claim_id` on the roster), never
by text matching. It does not read the simulator's latent layer; a critic with
privileged access could not be trusted to judge what the viewer can see.

- **Disputes, not deletions.** The schema carries `disputed_claims` and embedded
  `ruling` objects. A ruling may be `supported`, `misattributed`, or
  `unsupported`. Claims are never dropped wholesale; disputes are surfaced.
- **Four gates** still apply to the desk (`narrator.py`): temporal validity,
  event existence, specificity, and attribution discipline.
- **Omission is not an error.** The Verifier agrees before it rejects; it does
  like-with-like and reasons in terms of the provided events/insights.
- **Measurable precision/recall.** `eval_verifier.py` runs a labelled harness
  with untouched true claims as false-positive probes and corrupted mutations
  (team swaps, wrong numbers/players/locations). The local critic is
  attribution-only and sees the same cited ids on a mutated sentence, so it
  catches contradictions but not every mutation; `--live` scores the configured
  model critic. The harness reports precision, recall and the false-positive rate
  separately, because "never objects" must not score as perfect.

## Agentic orchestration

`agents.py` defines named roles (`GENERATE`, `INTERPRET`, `EXPLAIN`, `RENDER`,
`DESK`, `VERIFY`) with declared contracts (reads/writes), a shared blackboard,
and a trace. `server.py` assembles them into a pipeline and exposes an
**orchestration trace panel** in the UI (from the `done` frame). The trace shows
each agent's status/calls/detail and the number of disputed claims surfaced by
the Verifier.

## Public repo readiness

- `.env.example` documents required/optional environment variables (no secrets).
- `LICENSE` included (MIT).
- All clubs, players and matches are fictional. No real football data,
  trademarks or match footage are used. No betting/productisation of odds/value
  analysis; any value/odds framing appears only as model-vs-market
  explainability if at all. The demo is a local recording — no Azure deployment
  is executed as part of the demo path.
