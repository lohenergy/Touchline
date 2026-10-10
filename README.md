# Touchline

Match intelligence that explains *why* a moment mattered, not just what happened.

Built for the **Microsoft Inside the Game: Developer Hackathon** (Oct 2026).

---

## The problem

Football data is presented brilliantly and understood poorly.

Official apps and broadcast graphics tell you *what* happened — possession,
shots, distance covered, who completed the most passes — and stop there. They
cannot tell you **why** a moment mattered, because a leaderboard can only rank a
single value. Worse, the same match means different things to different people:
an analyst wants pressing triggers, a casual fan wants a story, a fantasy
manager wants one player's involvement. Broadcast infrastructure builds one
product for all of them.

And the data needed to build a fix is closed to nearly everyone, which is why
the hackathon mandates synthetic, football-realistic data.

## The approach

Touchline turns a live event stream into **grounded** match intelligence: every
claim is tied to the specific events that produced it, packaged as timed,
machine-readable overlay payloads a broadcaster's graphics system can consume
directly, and reshaped per audience and language.

The differentiator is **verifiability**. Because the data is synthetic, the
simulator knows which team was genuinely controlling the match at any instant —
ground truth that real match data does not have. That means the system can
*score* its own explanations against a known answer, and ship with a measured
error bar, rather than asking anyone to trust them.

---

## Architecture

The pipeline follows the brief's five stages:

| Stage | What happens | Where |
|---|---|---|
| **Ingest** | Synthetic events stream in real time | `simulator.py` |
| **Interpret** | Events → `MatchState` (control, chaos, tempo, shot quality) | `metrics.py` *(Phase 1)* |
| **Explain** | Detect key moments, attribute *why* using cited evidence | `explainer.py` *(Phase 1)* |
| **Render** | Timed overlay payloads consumed by a graphics layer | `overlay.py` *(Phase 1)* |
| **Personalize** | Audience modes and languages from one intelligence layer | *(Phase 2)* |

## Repository layout

```
schemas.py      Domain contracts: MatchEvent, LatentState, MatchState (+ pass
                types, stoppage ledger, contested possession)
simulator.py    Synthetic match generator (observable events + latent ground truth)
validate.py     Calibration harness: does the synthetic data look like football?
test_events.py  Invariant tests: is the event stream semantically correct?
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

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python simulator.py        # generate one match and print a summary
python validate.py 120     # calibration report over 120 matches
python test_events.py      # invariant tests on the event stream
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
  observable count brackets the ground-truth counter exactly.

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

## Calibration status

Synthetic aggregates over 120 matches, against published reference ranges for
professional league football:

| Metric | Synthetic | Reference |
|---|---|---|
| Goals per match | 2.78 | 2.4–3.2 |
| Shots per match | 29.1 | 20–30 |
| Shots on target | 9.8 | 7–11 |
| xG per match | 2.84 | 2.2–3.4 |
| xG per shot | 0.10 | 0.08–0.14 |
| Passes per match | 1079 | 800–1200 |
| Pass completion | 84.4% | 78–88% |
| Home possession *of controlled time* | 50.2% | 42–58% |
| Contested possession | 7.4% | 3–18% |
| Loose-ball phases | 73 | — |
| Added time, first half | 2:44 | — |
| Added time, second half | 4:50 | — |
| Added time, total | 7:34 | 2–16 min |
| Corners per match | 10.4 | 8–12 |
| Offsides per match | 3.6 | 1.5–4.5 |
| Fouls per match | 22.8 | 18–28 |
| Yellow cards | 3.7 | 2–6 |
| Red cards | 0.33 | 0–0.5 |
| Substitutions | 4.8 | 4–10 |

Matching aggregate marginals is **necessary but not sufficient** — data can match
every average and still be structurally absurd. The invariant tests above cover
the semantic layer; sequence structure, correlation and game-state effects are
validated separately in later layers.

> The reference ranges are commonly cited figures that still need to be properly
> sourced and cited before submission. An unsourced number in a validation report
> is worse than no report: it looks rigorous while being unfalsifiable. The
> contested-possession and added-time ranges are *design targets* set from the
> simulator's own causal structure; they are the ones most in need of a real
> source before submission.

## Status

Phase 0 complete: contracts and simulator, including the completed event stream
(offsides, corners, cards, substitutions), stoppage time with an explainable
ledger, three-way possession with contested time, and pass-intent ground truth.
Next — the metrics engine that turns events into `MatchState`.

All clubs, players and matches are fictional. No real football data, trademarks
or match footage are used.
