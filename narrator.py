"""The Narrator stage: one grounded language-model call per scheduled reveal.

WHY THIS IS THE LAST STAGE AND THE ONLY ONE THAT CALLS A MODEL
--------------------------------------------------------------
Every stage before this one is deterministic: the simulator generates a match,
`metrics.py` interprets it, `explainer.py` finds and attributes the key moments,
`overlay.py` times them. All of that is reproducible byte for byte. The Narrator
is the only place a language model is used, and it is deliberately *downstream*
of everything else -- it consumes finished, cited objects. If the model vanished
tomorrow the product would lose its colour, not its correctness.

The model is asked for exactly two things per match, at two fixed points on the
clock: half-time and full-time. Nothing chains, nothing is re-prompted, and the
whole match costs two requests. That is a budget a solo project can reason about,
and it keeps the latency off the critical path -- at half-time the client is
already paused, and at full-time the client is already showing the summary.

THE FOUR GATES
---------------
A language model asked to "describe the match" will invent a stat, a minute or a
name, because writing fluently and being right are different skills. Rather than
beg the model to be careful, this module assumes it will sometimes be wrong and
catches it four times over:

    input gate   the prompt contains the MatchSummary and the Insights -- both
                 observable-only by construction -- and never a latent field.
                 `test_narrator.py` pins this by asserting no latent key name can
                 appear in a serialised prompt.
    citation     every `event_id` a claim cites must resolve to a real observable
                 event. A claim citing an event that does not exist is dropped.
    number       every figure a claim writes must appear in the facts the model
                 was shown -- and in the right *place*. A claim citing nothing may
                 use only the fact sheet's figures; a claim may quote a single
                 moment's figure only if it cites that moment. This is scoped
                 rather than global because a global whitelist answers "did this
                 number come from us?" but not "did it belong to the subject of
                 this sentence?", and the difference is a false claim that passes.
    causation    a claim may not assert that one event caused another. The Explain
                 stage has no CAUSE insight because the data does not support one
                 (a deep incomplete pass precedes a chance 3.6% of the time), and a
                 pundit's instinct is to assert causation anyway. The gates above
                 cannot see that: every id and figure in such a sentence is valid,
                 and the falsehood is in the connective. See `_CAUSAL_RE`.

Only claims that clear all four survive into the `RevealNarrative`. Anything
discarded is counted in `dropped_claims`, and if a persona loses every claim it
is rebuilt from `LocalNarrator`. The panel always renders; the provenance of
every sentence is always visible.

WHY A LOCAL FALLBACK IS NOT OPTIONAL
------------------------------------
The demo must not depend on the network. `LocalNarrator` composes the same three
personas from templates built out of the explainer's own cited reasons, so its
citations and numbers are valid by construction. It is the default provider, and
it is what runs whenever a call fails, times out, or fails validation. A dead
network degrades the commentary; it cannot break the product.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from schemas import (
    Insight,
    InsightKind,
    MatchSummary,
    NarrativeClaim,
    PersonaId,
    PersonaTake,
    Reveal,
    RevealNarrative,
)

# Load .env for local development. `override=False` (the default) means a real
# environment variable always wins, so a container or CI setting is never
# silently replaced by a stray file on disk.
load_dotenv()

PERSONA_ORDER = (PersonaId.ANALYST, PersonaId.DIEHARD, PersonaId.NUMBERS)

_NUM = re.compile(r"\d+(?:\.\d+)?")

# --------------------------------------------------------------------------- #
# Gate 4: no causation
# --------------------------------------------------------------------------- #
# The Explain stage has no CAUSE insight, and that is a finding, not an
# oversight: the data says a deep incomplete pass is followed by a chance only
# 3.6% of the time, so "this giveaway led to that goal" is not something this
# system can stand behind. A language model asked to write like a pundit asserts
# exactly that anyway -- it is how pundits talk -- and neither the citation gate
# nor the number gate can see it, because every id and figure in the sentence is
# individually valid. The over-claim lives in the *connective*, so that is what
# this gate reads.
#
# This enforces a decision the product already made, mechanically, rather than
# hoping a prompt is obeyed. The intent is "one event made another happen", not
# any incidental use of these words; edit the pattern to widen or narrow it.
_CAUSAL_RE = re.compile(
    r"\b(?:led|leads|leading)\s+to\b"
    r"|\ballow(?:ed|ing|s)\b"
    r"|\bbecause\b"
    r"|\bdue\s+to\b"
    r"|\bas\s+a\s+result\b"
    r"|\bcaus(?:e|ed|es|ing)\b"
    r"|\bresult(?:ed|s|ing)\s+in\b"
    r"|\bcapit(?:alised|alized|alising|alizing)\s+on\b"
    r"|\bthanks\s+to\b"
    r"|\bcontribut(?:e|ed|es|ing)\b"
    r"|\btrigger(?:ed|s|ing)\b"
    r"|\bspark(?:ed|s|ing)\b"
    r"|\benabl(?:e|ed|es|ing)\b"
    r"|\bpunish(?:ed|ing|es)\b"
    r"|\bexploit(?:ed|ing|s)\b"
    r"|\bpaved\s+the\s+way\b"
    r"|\bstemming\s+from\b"
    r"|\bset(?:ting)?\s+up\b"
    r"|\bhelp(?:ed|ing|s)\b"
    r"|\bforced\b"
    r"|\bturned\b[^.]{0,40}\binto\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# The context handed to the narrator
# --------------------------------------------------------------------------- #


class NarrativeContext(BaseModel):
    """Everything the Narrator is allowed to know at one reveal.

    Note what is *not* here: no latent truth, no raw event stream. Handing a
    model 1,700 events would cost a fortune in tokens and invite it to wander;
    handing it the summary plus the handful of moments the Explain stage already
    found and cited is both cheaper and more honest, because every citable id is
    one the tape can actually show.
    """

    match_id: str
    reveal: Reveal
    clock_s: int
    summary: MatchSummary
    insights: list[Insight] = []


def build_context(
    match_id: str,
    reveal: Reveal,
    clock_s: int,
    summary: MatchSummary,
    insights: list[Insight],
) -> NarrativeContext:
    return NarrativeContext(
        match_id=match_id,
        reveal=reveal,
        clock_s=clock_s,
        summary=summary,
        insights=list(insights),
    )


# --------------------------------------------------------------------------- #
# Gate 2: the citation pool
# --------------------------------------------------------------------------- #


def citable_event_ids(insights: list[Insight]) -> set[str]:
    """Every id a claim is allowed to cite: the anchors and their evidence.

    An `Insight` already carries a validated audit trail -- `anchor_event_id` is
    the moment itself and `evidence_event_ids` is the union of every reason's
    citations. Reusing that set means the narrator can cite nothing that the
    Explain stage did not already stand behind.
    """
    ids: set[str] = set()
    for insight in insights:
        ids.add(insight.anchor_event_id)
        ids.update(insight.evidence_event_ids)
    return ids


# --------------------------------------------------------------------------- #
# Gate 3: the number audit
# --------------------------------------------------------------------------- #
# The model may only write a figure it was shown. To check that mechanically we
# build the set of every number present in the facts, in the several forms a
# writer might use it -- 0.56 as "0.56", "0.6", or "56%" -- and then demand that
# each number in a claim appear in that set. It is deliberately a *whitelist*:
# a number we did not supply is exactly the failure mode we are guarding
# against, so "not found" must mean "drop", never "allow to be safe".


def _number_forms(value: object) -> set[str]:
    """Every reasonable written form of one numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return set()
    forms: set[str] = set()
    if isinstance(value, int):
        forms.add(str(value))
        forms.add(f"{float(value):.0f}")
    else:
        for dp in (0, 1, 2):
            forms.add(f"{value:.{dp}f}")
    # A proportion is often written as a percentage: possession 0.561 -> "56%".
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        for dp in (0, 1):
            forms.add(f"{value * 100:.{dp}f}")
    return forms


def _collect_numbers(obj: object, into: set[str]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        into.update(_number_forms(obj))
    elif isinstance(obj, str):
        into.update(_NUM.findall(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_numbers(value, into)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_numbers(value, into)


def allowed_numbers(ctx: NarrativeContext) -> set[str]:
    """Every figure the model is permitted to write, derived from its inputs.

    Sourced from exactly what the model is shown: the tidy fact sheet, plus each
    moment's headline and the text of its already-cited reasons (which is where
    figures like a single shot's xG value live). If it is not in here, the model
    did not learn it from us.

    Note this is the *whole* supply. It is not the set a given claim is judged
    against -- see `_scoped_numbers`, which refuses to let a moment-level figure
    be used by a sentence that cites nothing.
    """
    allowed: set[str] = set()
    _collect_numbers(_facts_for_prompt(ctx.summary), allowed)
    for insight in ctx.insights:
        allowed.update(_NUM.findall(insight.headline))
        allowed.add(str(insight.minute))
        for reason in insight.reasons:
            allowed.update(_NUM.findall(reason.text))
    return allowed


def _scoped_numbers(
    ctx: NarrativeContext,
) -> tuple[set[str], Callable[[NarrativeClaim], set[str]]]:
    """Scope the number whitelist to what a *given* claim is entitled to use.

    A global whitelist answers "did this number come from us?" but not "did it
    belong to the thing this sentence is about?". A live run exposed the gap: the
    model wrote "Northbridge's goals included one with an xG of 0.48", citing
    nothing. 0.48 was in the supply -- it is a *saved* shot's xG -- so the gate
    waved it through, and the sentence was false.

    So the whitelist is split in two:

        a sentence about the match as a whole (no citation) may use the fact
        sheet's figures and nothing else;
        a sentence about a specific moment may additionally use that moment's
        own figures -- but only those, and only if it cites the moment.

    That closes the hole by making the citation *buy* the access to moment-level
    numbers, rather than letting any supplied figure be attached to any subject.
    """
    summary_numbers: set[str] = set()
    _collect_numbers(_facts_for_prompt(ctx.summary), summary_numbers)

    per_event: dict[str, set[str]] = {}
    for insight in ctx.insights:
        numbers = set(_NUM.findall(insight.headline))
        numbers.add(str(insight.minute))
        for reason in insight.reasons:
            numbers.update(_NUM.findall(reason.text))
        ids = {insight.anchor_event_id, *insight.evidence_event_ids}
        for event_id in ids:
            per_event.setdefault(event_id, set()).update(numbers)

    def numbers_for(claim: NarrativeClaim) -> set[str]:
        if not claim.event_ids:
            return summary_numbers
        allowed = set(summary_numbers)
        for event_id in claim.event_ids:
            allowed |= per_event.get(event_id, set())
        return allowed

    return summary_numbers, numbers_for


def _audit_claims(
    claims: list[NarrativeClaim],
    citable: set[str],
    numbers_for: Callable[[NarrativeClaim], set[str]],
) -> tuple[list[NarrativeClaim], list[str]]:
    """Apply the citation, number and causation gates.

    Returns (survivors, rejection notes). All three checks are whitelists in
    spirit: a claim survives only by proving it is grounded, and anything that
    cannot prove it is dropped rather than trusted. The notes are returned rather
    than merely counted so the run log can say which sentence died and why --
    a gate nobody can audit is just as opaque as a model nobody can audit.
    """
    survivors: list[NarrativeClaim] = []
    rejected: list[str] = []
    for claim in claims:
        bad = [event_id for event_id in claim.event_ids if event_id not in citable]
        if bad:
            rejected.append(f"citation {bad[0]!r} is not a citable event :: {claim.text!r}")
            continue
        allowed = numbers_for(claim)
        stray = [t for t in _NUM.findall(claim.text) if t not in allowed]
        if stray:
            where = "the fact sheet" if not claim.event_ids else "the moments it cites"
            rejected.append(
                f"number {stray[0]!r} is not in {where} :: {claim.text!r}"
            )
            continue
        if _CAUSAL_RE.search(claim.text):
            rejected.append(f"asserts causation :: {claim.text!r}")
            continue
        survivors.append(claim)
    return survivors, rejected


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #
# The system prompt carries the persona definitions and the hard rules. The user
# prompt carries only data. Keeping them apart matters: the rules are static and
# cacheable, and the data is the only thing that changes per reveal.


SYSTEM_PROMPT = """You are the virtual studio desk for Touchline, a football \
match-intelligence system. At each reveal, three pundits give their reading of \
the match. They share one set of facts and differ only in voice.

THE THREE CHAIRS
- "analyst": measured and tactical. Reads the shape of the game -- territory, \
patterns, where the play kept happening. Calm authority, no speculation.
- "diehard": a partisan supporter. Emotional and one-eyed about their own side, \
but never invents a fact to make a point.
- "numbers": an expected-goals purist. Opens with figures and only trusts what it \
can point a number at.

HARD RULES (a claim that breaks one is discarded automatically)
1. Use ONLY the facts in MATCH FACTS and KEY MOMENTS below. Never invent a score, \
statistic, minute, player or event.
2. If a sentence is about a specific moment, cite the observable event ids it is \
true because of in "event_ids", using only ids from CITABLE EVENT IDS. Cite every \
id a sentence depends on. If a sentence is about the match as a whole, use an \
empty list -- that is allowed and means "grounded in the summary". The citation is \
what buys access to a moment's figures: an uncited sentence may use only the \
numbers in MATCH FACTS, while a sentence may quote a single moment's number (a \
shot's xG, a minute, a distance) only if it cites that moment. Never attach a \
moment's number to a sentence about a different subject.
3. Any number you write must be one you were given, in the place it was given. Do \
not do arithmetic, do not round into a number that is not shown, do not estimate.
4. Describe, never explain why. This system does not model causation and no claim \
that one event caused another is supportable, so sentences that connect events \
are discarded before anyone sees them. Never write that one thing "led to", \
"allowed", "caused", "resulted in" or "set up" another, never say one thing \
happened "because of", "due to" or "thanks to" another, never say a side \
"capitalised on", "punished" or "exploited" something, and never "turn X into Y". \
Report what happened and let the reader draw the conclusion.
5. Attribution is exact. When a sentence names a team, every event it cites must \
belong to that team -- check the "team" field of every moment you cite. Do not \
describe one team's giveaways using the other team's events.
6. Do not characterise a chance with a quality word like "high quality", "big", \
"clear-cut" or "clinical" unless the xG shown for it is 0.35 or more. Otherwise \
just state the xG and let the number speak.
7. No betting, no odds, no wagering, no market talk of any kind.
8. Output JSON only. No prose, no markdown fences.

OUTPUT SHAPE (exactly this)
{"personas":[
  {"persona":"analyst","headline":"<=90 chars","claims":[{"text":"one sentence","event_ids":["..."]}]},
  {"persona":"diehard","headline":"...","claims":[{"text":"...","event_ids":[]}]},
  {"persona":"numbers","headline":"...","claims":[{"text":"...","event_ids":["..."]}]}
]}
Give all three personas, in that order, with two to three claims each."""


def _facts_for_prompt(summary: MatchSummary) -> dict:
    """The summary rewritten with human names as keys.

    The model reads names, not ids, and this is the only shape of the facts it
    is ever shown -- so it is also, by construction, the whitelist the number
    audit draws from.
    """
    home, away = summary.home_name, summary.away_name
    by_name = {
        home: summary.home,
        away: summary.away,
    }

    def per_team(source: dict) -> dict:
        return {home: source.get(summary.home), away: source.get(summary.away)}

    return {
        "match": {"home": home, "away": away},
        "score": f"{summary.score.get(summary.home, 0)}-"
                 f"{summary.score.get(summary.away, 0)}",
        "possession_pct": {
            name: round((summary.possession.get(team_id, 0.0)) * 100)
            for name, team_id in by_name.items()
        },
        "contested_pct": round(summary.contested_pct),
        "shots": per_team(summary.shots),
        "shots_on_target": per_team(summary.shots_on_target),
        "shots_in_box": per_team(summary.shots_in_box),
        "woodwork": per_team(summary.woodwork),
        "saves": per_team(summary.saves),
        "blocks": per_team(summary.blocks),
        "xg": {k: round(v, 2) for k, v in per_team(summary.xg).items()},
        "passes": per_team(summary.passes),
        "pass_completion_pct": {
            k: round(v * 100) for k, v in per_team(summary.pass_completion).items()
        },
        "pass_difficulty_mean": {
            k: round(v, 1) for k, v in per_team(summary.pass_difficulty_mean).items()
        },
        "corners": per_team(summary.corners),
        "offsides": per_team(summary.offsides),
        "fouls": per_team(summary.fouls),
        "yellows": per_team(summary.yellows),
        "reds": per_team(summary.reds),
        "substitutions": per_team(summary.substitutions),
        "territory_pct": {
            k: round(v * 100) for k, v in per_team(summary.territory).items()
        },
    }


def _moments_for_prompt(ctx: NarrativeContext) -> list[dict]:
    """The key moments, flattened to what the model needs to write about them."""
    names = {ctx.summary.home: ctx.summary.home_name,
             ctx.summary.away: ctx.summary.away_name}
    moments = []
    for insight in ctx.insights:
        moments.append(
            {
                "minute": insight.minute,
                "kind": insight.kind.value,
                "team": names.get(insight.team, insight.team),
                "player": insight.player_name,
                "headline": insight.headline,
                "evidence": [
                    {"event_id": r.event_ids[0] if r.event_ids else "",
                     "text": r.text, "event_ids": r.event_ids}
                    for r in insight.reasons
                ],
            }
        )
    return moments


def build_user_prompt(ctx: NarrativeContext) -> str:
    payload = {
        "reveal": "half-time" if ctx.reveal is Reveal.HALF_TIME else "full-time",
        "minute": ctx.clock_s // 60,
        "match_facts": _facts_for_prompt(ctx.summary),
        "key_moments": _moments_for_prompt(ctx),
        "citable_event_ids": sorted(citable_event_ids(ctx.insights)),
    }
    return (
        "MATCH FACTS, KEY MOMENTS and the CITABLE EVENT IDS are below.\n"
        "Write the studio desk's reading for this reveal as JSON.\n\n"
        + json.dumps(payload, indent=2)
    )


# --------------------------------------------------------------------------- #
# Parsing the model's answer
# --------------------------------------------------------------------------- #


class NarrativeParseError(ValueError):
    """The model's output was not the JSON object this stage requires."""


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_personas(raw: str) -> list[PersonaTake]:
    """Parse the model's JSON into validated `PersonaTake` objects.

    Raises `NarrativeParseError` on anything unexpected -- bad JSON, a missing
    list, a claim with no text. Callers treat that as "fall back", never as
    "carry on with whatever we got".
    """
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise NarrativeParseError(f"not JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("personas"), list):
        raise NarrativeParseError("missing 'personas' list")
    takes: list[PersonaTake] = []
    for entry in data["personas"]:
        try:
            takes.append(PersonaTake.model_validate(entry))
        except ValidationError as exc:
            raise NarrativeParseError(f"bad persona: {exc}") from exc
    if not takes:
        raise NarrativeParseError("no personas")
    return takes


# --------------------------------------------------------------------------- #
# The deterministic fallback -- and the default provider
# --------------------------------------------------------------------------- #


def _claims_from_insight(insight: Insight, limit: int = 3) -> list[NarrativeClaim]:
    """Turn an insight's own cited reasons into claims.

    Reusing the reasons as sentences is what makes the fallback valid by
    construction: the citations were checked when the Explain stage built them,
    and the figures in them were fed into the number audit's whitelist.
    """
    claims = [
        NarrativeClaim(text=reason.text, event_ids=list(reason.event_ids))
        for reason in insight.reasons[:limit]
    ]
    if not claims:
        claims = [NarrativeClaim(text=insight.headline, event_ids=[insight.anchor_event_id])]
    return claims


def _local_personas(ctx: NarrativeContext) -> list[PersonaTake]:
    """Compose the three personas from templates over the cited moments."""
    summary = ctx.summary
    home, away = summary.home_name, summary.away_name
    score = (f"{summary.score.get(summary.home, 0)}-"
             f"{summary.score.get(summary.away, 0)}")

    goals = [i for i in ctx.insights if i.kind is InsightKind.GOAL]
    chances = [i for i in ctx.insights if i.kind is InsightKind.CHANCE]
    giveaways = [i for i in ctx.insights if i.kind is InsightKind.DEEP_GIVEAWAY]

    # The Analyst: the most decisive moment, explained in its own words.
    lead = goals[-1] if goals else (chances[0] if chances else None)
    if lead is not None:
        analyst_claims = _claims_from_insight(lead)
    elif giveaways:
        analyst_claims = _claims_from_insight(giveaways[0])
    else:
        analyst_claims = [NarrativeClaim(
            text=f"A settled {score} with little between the sides.",
            event_ids=[],
        )]

    # The Diehard: partisan colour, built on a real moment so it stays cited.
    if goals:
        pick = goals[0]
        diehard_claims = [NarrativeClaim(
            text=f"That {pick.minute}' goal changed the whole mood of the game.",
            event_ids=[pick.anchor_event_id],
        )] + _claims_from_insight(pick, limit=1)
    else:
        diehard_claims = [NarrativeClaim(
            text="No goals, but plenty to argue about all the same.",
            event_ids=[],
        )]

    # The Numbers: only figures, and only ones it was given.
    xg_home = round(summary.xg.get(summary.home, 0.0), 2)
    xg_away = round(summary.xg.get(summary.away, 0.0), 2)
    numbers_claims = [NarrativeClaim(
        text=f"{home} finished on {xg_home} expected goals to "
             f"{away}'s {xg_away}.",
        event_ids=[],
    )]
    if chances:
        top = max(chances, key=lambda i: i.clock_s)
        numbers_claims.append(_claims_from_insight(top, limit=1)[0])

    return [
        PersonaTake(persona=PersonaId.ANALYST,
                    headline=f"{home} {score} {away}", claims=analyst_claims),
        PersonaTake(persona=PersonaId.DIEHARD,
                    headline="From the terraces", claims=diehard_claims),
        PersonaTake(persona=PersonaId.NUMBERS,
                    headline=f"xG {xg_home} - {xg_away}", claims=numbers_claims),
    ]


def local_narrative(ctx: NarrativeContext) -> RevealNarrative:
    personas = _local_personas(ctx)
    return RevealNarrative(
        narrative_id=f"{ctx.match_id}-{ctx.reveal.value}-local",
        match_id=ctx.match_id,
        reveal=ctx.reveal,
        clock_s=ctx.clock_s,
        personas=personas,
        provider="local",
        model="touchline-templates-v1",
        validated=False,
        dropped_claims=0,
    )


class LocalNarrator:
    """The always-available narrator. No network, no key, fully deterministic.

    It is the default provider precisely because it cannot fail: the product's
    worst case should be plainer prose, not a dead panel.
    """

    provider = "local"
    model = "touchline-templates-v1"

    async def narrate(self, ctx: NarrativeContext) -> RevealNarrative:
        return local_narrative(ctx)


# --------------------------------------------------------------------------- #
# The HTTP narrator
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HttpPlan:
    """Everything provider-specific, resolved once at startup."""

    provider: str
    url: str
    headers: dict[str, str]
    body_model: str | None     # value for the request's "model" field, if any
    model_label: str           # what we record in the narrative


class ChatCompletionsNarrator:
    """A narrator backed by an OpenAI-compatible chat-completions endpoint.

    The same class serves Azure OpenAI and GitHub Models; the only differences
    are the URL, the auth header and whether the model travels in the body --
    all of which live in an `HttpPlan`. That is what makes the provider a config
    choice rather than a code change.
    """

    def __init__(self, plan: HttpPlan, timeout_s: float = 30.0,
                 max_tokens: int = 700, temperature: float = 0.3,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.plan = plan
        self.provider = plan.provider
        self.model = plan.model_label
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        # Low by default: this is a report on fixed facts, not a creative writing
        # task, and the demo video should not differ from the run the judges make.
        self.temperature = temperature
        # `transport` exists so tests can drive the real request path against a
        # scripted response. Production passes nothing and gets a real socket.
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, ctx: NarrativeContext, json_mode: bool) -> dict:
        payload: dict = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ctx)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.plan.body_model:
            payload["model"] = self.plan.body_model
        return payload

    async def _post(self, ctx: NarrativeContext) -> str:
        """Return the raw assistant text, retrying once without JSON mode.

        `response_format` is supported by the current chat models but not by
        every endpoint; the retry keeps a 400 about that field from costing us
        the reveal.
        """
        response = await self._client.post(
            self.plan.url, headers=self.plan.headers, json=self._payload(ctx, True)
        )
        if response.status_code == 400 and "response_format" in response.text:
            response = await self._client.post(
                self.plan.url, headers=self.plan.headers,
                json=self._payload(ctx, False),
            )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"] or ""

    async def narrate(self, ctx: NarrativeContext) -> RevealNarrative:
        try:
            raw = await self._post(ctx)
            takes = parse_personas(raw)
        except Exception as exc:  # noqa: BLE001 -- any failure means "fall back"
            print(f"[narrator] {self.provider} call failed ({type(exc).__name__}: "
                  f"{exc}); using local fallback")
            return local_narrative(ctx)

        return self._assemble(ctx, takes)

    def _assemble(self, ctx: NarrativeContext, takes: list[PersonaTake]) -> RevealNarrative:
        """Apply the gates, backfill any persona that lost every claim."""
        citable = citable_event_ids(ctx.insights)
        _, numbers_for = _scoped_numbers(ctx)
        fallback = {t.persona: t for t in _local_personas(ctx)}

        kept_personas: list[PersonaTake] = []
        dropped = 0
        validated = True
        for persona_id in PERSONA_ORDER:
            take = next((t for t in takes if t.persona == persona_id), None)
            if take is None:
                print(f"[narrator] no {persona_id.value} take in the reply; "
                      f"using the deterministic one")
                kept_personas.append(fallback[persona_id])
                validated = False
                continue
            survivors, rejected = _audit_claims(take.claims, citable, numbers_for)
            dropped += len(rejected)
            for note in rejected:
                print(f"[narrator] gate dropped a {persona_id.value} claim: {note}")
            if not survivors:
                kept_personas.append(fallback[persona_id])
                validated = False
            else:
                kept_personas.append(take.model_copy(update={"claims": survivors}))

        return RevealNarrative(
            narrative_id=f"{ctx.match_id}-{ctx.reveal.value}-{self.provider}",
            match_id=ctx.match_id,
            reveal=ctx.reveal,
            clock_s=ctx.clock_s,
            personas=kept_personas,
            provider=self.provider,
            model=self.model,
            validated=validated,
            dropped_claims=dropped,
        )


# --------------------------------------------------------------------------- #
# Configuration: turning the environment into a narrator
# --------------------------------------------------------------------------- #


def _normalize_endpoint(endpoint: str) -> str:
    """Reduce a pasted endpoint URL to the bare origin we append paths to.

    Copying from the Azure or Foundry portal usually hands you the *full* URL for
    one operation, e.g. ``https://res.services.ai.azure.com/openai/v1/responses``.
    Appending our own path to that yields ``/openai/v1/responses/openai/v1/
    chat/completions`` and an opaque ``404 Resource not found`` that looks exactly
    like a missing deployment. Trimming at the first API segment makes the portal's
    copy button harmless. The host is never touched -- ``.openai.azure.com`` is a
    domain, not a path segment, so only ``/openai`` and ``/api`` are cut.
    """
    endpoint = endpoint.strip().rstrip("/")
    for marker in ("/openai/", "/api/", "/openai", "/api"):
        idx = endpoint.find(marker)
        if idx != -1:
            return endpoint[:idx].rstrip("/")
    return endpoint


def _azure_plan() -> HttpPlan:
    endpoint = _normalize_endpoint(os.environ["AZURE_OPENAI_ENDPOINT"])
    key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
    if api_version:
        url = (f"{endpoint}/openai/deployments/{deployment}"
               f"/chat/completions?api-version={api_version}")
    else:
        # The OpenAI-compatible v1 surface needs no api-version at all.
        url = f"{endpoint}/openai/v1/chat/completions"
    return HttpPlan(
        provider="azure",
        url=url,
        headers={"Content-Type": "application/json", "api-key": key},
        body_model=deployment,
        model_label=deployment,
    )


def _github_plan() -> HttpPlan:
    token = os.environ["GITHUB_MODELS_TOKEN"]
    model = os.environ.get("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini")
    return HttpPlan(
        provider="github",
        url="https://models.github.ai/inference/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        },
        body_model=model,
        model_label=model,
    )


def narrator_from_env() -> "Narrator":
    """Build the narrator the environment asks for, or fall back to local.

    An unset or misconfigured provider is not an error the demo should die of --
    it is a reason to use the deterministic narrator.
    """
    provider = os.environ.get("TOUCHLINE_LLM_PROVIDER", "local").strip().lower()
    timeout = float(os.environ.get("TOUCHLINE_LLM_TIMEOUT_S", "30"))
    max_tokens = int(os.environ.get("TOUCHLINE_LLM_MAX_TOKENS", "700"))
    temperature = float(os.environ.get("TOUCHLINE_LLM_TEMPERATURE", "0.3"))

    if provider == "local":
        return LocalNarrator()
    try:
        plan = {"azure": _azure_plan, "github": _github_plan}[provider]()
    except KeyError:
        print(f"[narrator] unknown provider {provider!r}; using local")
        return LocalNarrator()
    except (KeyError, ValueError) as exc:
        print(f"[narrator] {provider} is missing config ({exc}); using local")
        return LocalNarrator()
    print(f"[narrator] provider={plan.provider} model={plan.model_label}")
    return ChatCompletionsNarrator(
        plan, timeout_s=timeout, max_tokens=max_tokens, temperature=temperature
    )


@runtime_checkable
class Narrator(Protocol):
    """The one method the server needs. Both implementations satisfy it."""

    provider: str
    model: str

    async def narrate(self, ctx: NarrativeContext) -> RevealNarrative: ...


# --------------------------------------------------------------------------- #
# Manual smoke run:  ./.venv/bin/python narrator.py [seed] [--live]
# --------------------------------------------------------------------------- #


def _context_for_seed(seed: int, reveal: Reveal) -> NarrativeContext:
    from explainer import MatchExplainer
    from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"SMOKE-{seed}", seed=seed)
    sim.simulate(live=True)
    explainer = MatchExplainer(
        sim.match_id, sim.home.team_id, sim.away.team_id,
        home_name=sim.home.name, away_name=sim.away.name,
    )
    for event in sim.observable_stream():
        explainer.update(event)
        if reveal is Reveal.HALF_TIME and event.type.value == "half_time":
            break
    clock_s = 2700 if reveal is Reveal.HALF_TIME else 5400
    return build_context(
        sim.match_id, reveal, clock_s, explainer.summary(), explainer.insights()
    )


async def _main(argv: list[str]) -> None:
    seed = int(argv[0]) if argv and argv[0].isdigit() else 81
    # Without --live we deliberately use the deterministic narrator, so this
    # smoke run never depends on a key, a quota or the network.
    narrator: Narrator = narrator_from_env() if "--live" in argv else LocalNarrator()
    for reveal in (Reveal.HALF_TIME, Reveal.FULL_TIME):
        ctx = _context_for_seed(seed, reveal)
        narrative = await narrator.narrate(ctx)
        print(f"\n=== {reveal.value.upper()}  provider={narrative.provider} "
              f"validated={narrative.validated} dropped={narrative.dropped_claims} ===")
        for take in narrative.personas:
            print(f"\n  [{take.persona.value}] {take.headline}")
            for claim in take.claims:
                cite = f"  cites={claim.event_ids}" if claim.event_ids else "  (summary)"
                print(f"    - {claim.text}{cite}")
    if isinstance(narrator, ChatCompletionsNarrator):
        await narrator.aclose()


if __name__ == "__main__":
    import sys

    asyncio.run(_main(sys.argv[1:]))
