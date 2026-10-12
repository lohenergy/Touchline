# Touchline

Match intelligence that explains *why* a moment mattered — and can prove it.

Built for the **Microsoft Inside the Game: Developer Hackathon** (Oct 2026).

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
| **Explain** | Detect key moments, attribute *why* using cited evidence | `explainer.py` *(Phase 1)* |
| **Render** | Timed overlay payloads consumed by a graphics layer | `overlay.py` *(Phase 1)* |
| **Personalize** | Narrative and Depth audience modes (plus languages) from one intelligence layer | *(Phase 2)* |

## Repository layout

```
schemas.py      Domain contracts: MatchEvent, LatentState, MatchState, MatchSummary
                (+ pass types, stoppage ledger, contested possession, save/block/
                woodwork events, in-box geometry, big-chance/error truth)
simulator.py    Synthetic match generator (observable events + latent ground truth)
metrics.py      Interpret stage: observable events -> MatchState / MatchSummary
validate.py     Calibration harness: does the synthetic data look like football?
test_events.py  Invariant tests: is the event stream semantically correct?
test_metrics.py Tests: can the match be rebuilt from observables alone?
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

## The web surface: one stream, four surfaces

`server.py` is the *composition root* — the one place where the pieces are
assembled behind HTTP (a Telegram bot's `main.py`, with routes instead of
handlers). It serves one page and one long-lived SSE stream, and emits five
kinds of frame:

```
meta     match identity, once, at the start
event    the raw observable event stream (the Ingest layer)
state    a periodic MatchState, rebuilt by metrics.py
summary  the full-time MatchSummary, just before the stream closes
done     terminal marker
```

The page renders three live surfaces — the pitch, the live numbers, and the event
ticker — plus a fourth that is *revealed* rather than browsed: the full-time
summary, which is the metrics engine given a face.

The pitch is an honest action map, drawn to scale (105m × 68m). **Every mark is
one real event**, and nothing is invented between them. The actor disc appears at
the actor's *true* `(x, y)` with their shirt number, then fades — it never
glides, because animating position between two events would be manufacturing
motion the data does not contain. Lenses are client-side filters over the same
event list, so the server never has to know what the viewer is looking at:

| Lens | Shows |
|---|---|
| **Action** | every event, with pass/shot lines and the actor disc |
| **Passing** | passes only; green complete / red incomplete; thickness = difficulty |
| **Possession** | where each team's play happens (territory) — one dot per event |
| **Shots & xG** | shots only; radius grows with xG; colour = outcome |

One connection carries everything, and every frame is built from observables.
The `MatchMetrics` instance in the server is fed exactly the events it sends —
the same bytes the browser receives — so the interpretation can only ever know
what the viewer knows.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python simulator.py        # generate one match and print a summary
python metrics.py          # interpret one match; print the full-time summary
python validate.py 120     # calibration report over 120 matches
python test_events.py      # invariant tests on the event stream
python test_metrics.py     # tests: the match rebuilt from observables alone

python -m uvicorn server:app --reload   # the console, at http://127.0.0.1:8000
```

`validate.py` exits non-zero if any metric falls outside its reference range, so
it can gate CI. `test_events.py` does the same for the invariants below.

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

## Three mechanisms that make the simulation honest

Three choices in the simulator exist purely so the *explainers* built on top can
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
channel. Phase 1 builds a geometric classifier that must recover it from
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
  `meta` / `event` / `state` / `summary` / `done`. The page renders the event
  ticker, a live numbers strip from periodic `state` frames, and — revealed at
  full time — the `summary`, which is the metrics engine given a face.
- **Phase 1 (pitch renderer)**: the action map, drawn to scale, with four
  client-side lenses (Action, Passing, Possession, Shots & xG) and numbered actor
  discs at the actor's true position that fade rather than glide. Shirt numbers
  are a new observable on `MatchEvent` (a camera can read them), guarded by an
  invariant test.

Next — the geometric pass classifier scored against `pass_type_true`, then the
explainer (`explainer.py`) and overlay payloads (`overlay.py`).

All clubs, players and matches are fictional. No real football data, trademarks
or match footage are used.
