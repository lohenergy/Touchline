"""The critic: a second agent that reads the desk's work and rules on it.

WHY A SECOND CALL
-----------------
Touchline's desk (`narrator.py`) already has four deterministic gates between the
model and the viewer. They catch a fabricated citation, a fabricated number, a
causal overclaim and a cross-persona fact. What they structurally cannot catch is
a claim that every one of those tests passes and is still wrong:

    "Riverside Athletic's pressure told in the second half."

Every word of that can be grounded -- Riverside exists, the events it cites are
real, the figures check out -- and the sentence can still attribute a moment to
the wrong team, or describe a real event inaccurately. Gates compare claims to
*identifiers*; nothing in that comparison reads the sentence.

So this stage asks a different reader the question the gates cannot: given the
evidence behind the events this claim cites, is the claim true as written?

WHAT IT IS NOT
--------------
It is not a continuation. The critic never writes commentary, never suggests a
replacement and never sees the conversation that produced the draft. It is handed
a finished draft and its evidence, and returns one ruling per claim. That is the
whole interface, and it is deliberately too narrow for a second call to become a
way of smuggling prose past the gates.

It also has no delete key. A claim it rules against *stays on the desk*, with the
ruling attached to it: the sentence, the verdict, the reason and the fact the
critic cited against it all travel together in the narrative. An earlier version
removed rejected claims, which was worse in a way that took live runs to see -- a
false rejection and a true finding both produced a missing sentence, so the only
evidence that the critic had ever been wrong was an absence nobody could inspect.
Now the disagreement is on screen, and a reader can tell a good ruling from a bad
one without trusting this module at all.

TWO CRITICS, ONE INTERFACE
--------------------------
`LocalVerifier` is deterministic and always available. It can only check the
thing a machine can check without reading English: whether a claim naming a
single team cites moments that team was involved in. That is a real class of
error -- it is the "misattributed" case in its clearest form -- and it is a small
fraction of what a reader would catch, so it says so in its own name:
`touchline-attribution-v1`.

`ChatCompletionsVerifier` is the same interface on top of a model, and it also
checks the semantics. When it is unreachable the reveal falls back to the
attribution critic and the narrative records that it did (`verifier="local"`,
`verified=False`) rather than pretending the review happened.

Run it:
    ./.venv/bin/python verifier.py 81              # the attribution critic
    ./.venv/bin/python verifier.py 81 --live       # whatever .env configures
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llm import ChatJsonClient, HttpPlan, provider_plan, strip_fences
# The fact sheet is shared with the desk on purpose: a critic checking the desk
# against a *different* set of facts would be answering a different question. The
# critic is downstream of the writer, so depending on the writer's module is the
# honest shape of that relationship.
from narrator import facts_for_prompt
from schemas import (
    ClaimVerdict,
    Insight,
    MatchSummary,
    NarrativeClaim,
    PersonaId,
    Reveal,
    RevealNarrative,
    Verdict,
)

# A judge should give the same answer twice. The desk gets 0.3 so it does not
# read like a template; the critic gets 0, because variation here is noise in a
# verdict, not character in a sentence.
VERIFIER_TEMPERATURE = 0.0

LOCAL_PROVIDER = "local"
LOCAL_MODEL = "touchline-attribution-v1"

# How much of a quoted fact it takes to count as a quoted fact. A rejection is
# only honoured if the critic says what it is overriding the claim with, and
# "no" is not a fact.
MIN_CONTRADICTION = 10


# --------------------------------------------------------------------------- #
# What the critic is given
# --------------------------------------------------------------------------- #
class VerificationContext(BaseModel):
    """The whole of the critic's world.

    Three things and nothing else: the draft it is judging, the fact sheet the
    desk was given, and the key moments the desk was given. There is no fourth
    field for ground truth to hide in -- the same structural argument as the
    desk's input gate, applied to the reader rather than the writer.
    """

    draft: RevealNarrative
    summary: MatchSummary
    insights: list[Insight] = Field(default_factory=list)


class Review(BaseModel):
    """The outcome of one review: the rulings, and how they were arrived at."""

    provider: str
    model: str
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    complete: bool = Field(
        default=False, description="A ruling was returned for every claim asked about."
    )
    by_model: bool = Field(
        default=False,
        description="The critic was a language model. False for the attribution "
        "critic, which never claims to have read the sentence.",
    )
    fell_back: bool = Field(
        default=False,
        description="The configured critic was unavailable and a stand-in ruled.",
    )

    @property
    def disputed(self) -> int:
        """How many claims the critic ruled against.

        The critic counts its own disagreements so that no two stages have to
        agree on what a verdict word means. Note this is what the *review* found;
        `RevealNarrative.disputed_claims` is what ended up marked on a claim, and
        the two differ only if a ruling addressed a claim that was not there.
        """
        return sum(1 for verdict in self.verdicts if verdict.verdict is not Verdict.SUPPORTED)


@runtime_checkable
class Verifier(Protocol):
    provider: str
    model: str

    async def review(self, ctx: VerificationContext) -> Review: ...


# --------------------------------------------------------------------------- #
# Addressing claims by structure, never by their English
# --------------------------------------------------------------------------- #
def claim_roster(draft: RevealNarrative) -> list[tuple[PersonaId, NarrativeClaim]]:
    """Every claim in the draft, in the order the desk wrote them.

    The id a claim carries is its position in this roster -- assigned once by
    `RevealNarrative` so that the number and the order cannot disagree. Walk this
    list to reach a claim from a ruling, never to decide which claim a ruling is
    about.
    """
    return [(take.persona, claim) for take in draft.personas for claim in take.claims]


def claim_index(draft: RevealNarrative) -> dict[int, tuple[PersonaId, NarrativeClaim]]:
    """The same roster, addressed by the id the claims carry."""
    return {claim.claim_id: (persona, claim) for persona, claim in claim_roster(draft)}


def evidence_for(claim: NarrativeClaim, insights: list[Insight]) -> list[Insight]:
    """The moments a claim is allowed to stand on: the ones it cited.

    Deliberately narrow. A critic handed the whole match can "verify" a claim
    against something the writer never saw, which would make the check
    unfalsifiable for the reader -- the point is to test the claim against its
    own stated evidence.
    """
    ids = set(claim.event_ids)
    if not ids:
        return []
    return [
        insight
        for insight in insights
        if ids & ({insight.anchor_event_id} | set(insight.evidence_event_ids))
    ]


# --------------------------------------------------------------------------- #
# The deterministic critic: attribution, and only attribution
# --------------------------------------------------------------------------- #
def _team_ruling(
    persona: PersonaId,
    claim: NarrativeClaim,
    cited: list[Insight],
    summary: MatchSummary,
) -> ClaimVerdict | None:
    """Flag a claim that credits exactly one team, when none of it is theirs.

    Three guards, each of which exists because the alternative would produce a
    false accusation against a sentence that is fine:

      * only when exactly one team is named. "Riverside responded to Northbridge"
        names both, and no single attribution can be wrong.
      * only when the claim cites something. A claim grounded in the summary as
        a whole has no evidence to contradict it.
      * only when the cited moments belong to the *other* team entirely. A claim
        spanning both sides is a summary, not a misattribution.

    The honest limitation: this reads two strings. It cannot tell a scorer from an
    assister, or a true sentence from a plausible one -- which is what
    `ChatCompletionsVerifier` is for, and why the two share an interface.
    """
    names = {
        team: name
        for team, name in (
            (summary.home, summary.home_name),
            (summary.away, summary.away_name),
        )
        if name
    }
    mentioned = [team for team, name in names.items() if name in claim.text]
    cited_teams = {insight.team for insight in cited}
    if len(mentioned) != 1 or not cited_teams or cited_teams & set(mentioned):
        return None
    credited = names[mentioned[0]]
    others = " and ".join(sorted(names.get(t, t) for t in cited_teams))
    return ClaimVerdict(
        claim_id=claim.claim_id,
        persona=persona,
        verdict=Verdict.MISATTRIBUTED,
        reason=f"The claim credits {credited}, but none of the cited moments are theirs.",
        contradicts=f"the cited moments belong to {others}",
    )


def local_verdicts(ctx: VerificationContext) -> list[ClaimVerdict]:
    """Rule on every claim with the checks a rule can make."""
    rulings: list[ClaimVerdict] = []
    for persona, claim in claim_roster(ctx.draft):
        cited = evidence_for(claim, ctx.insights)
        ruling = _team_ruling(persona, claim, cited, ctx.summary)
        rulings.append(
            ruling
            if ruling is not None
            else ClaimVerdict(
                claim_id=claim.claim_id,
                persona=persona,
                verdict=Verdict.SUPPORTED,
                reason="No attribution conflict in the cited moments.",
            )
        )
    return rulings


class LocalVerifier:
    """The critic that never needs a network, and never claims more than it does."""

    provider = LOCAL_PROVIDER
    model = LOCAL_MODEL

    async def review(self, ctx: VerificationContext) -> Review:
        return Review(
            provider=self.provider,
            model=self.model,
            verdicts=local_verdicts(ctx),
            complete=True,
            by_model=False,
            fell_back=False,
        )


# --------------------------------------------------------------------------- #
# The model critic
# --------------------------------------------------------------------------- #
VERIFIER_SYSTEM_PROMPT = """You are the fact-checker on a football broadcast \
desk. You do not write commentary; you rule on commentary that somebody else has \
written.

For each claim you are given the claim, and the basis it stands on:

  basis "cited_moments"  The claim cites specific moments. Check it against the \
evidence listed for it, and against the match facts and key moments.
  basis "match_facts"    The claim cites nothing. Check it against the match \
facts and key moments. It is not missing evidence -- the fact sheet is its \
evidence.

You are also given "key_moments": every moment the desk was allowed to draw on, \
in order. A claim that cites two goals may still be right about a third one, so \
check a claim that generalises across moments against this index, not only against \
the moments it named.

Rule each claim one of three ways:

  supported      the evidence shows what the claim says.
  misattributed  what the claim describes did happen, but it is credited to the \
wrong team or the wrong player.
  unsupported    the claim gets a checkable fact wrong, or asserts one that \
appears nowhere in the evidence.

Rules:
1. Use only the evidence you were given -- the cited moments and the match facts. \
You have no other source, and reaching for one would defeat the purpose of asking \
you.
2. Judge numbers, teams, players, locations, outcomes and sequences of play. Do \
not judge opinion, emphasis, mood or partisanship: "that goal changed the mood" is \
a matter of taste, so rule it supported. The desk deliberately includes a partisan \
voice, and ruling the passion out would silence it, not improve it.
3. A claim may state several things at once. Rule it supported when the evidence \
shows every part that states a fact: the opinion parts of a sentence are not \
checked, and they do not sink it. Two different figures in one sentence are not a \
contradiction -- a possession share of 50% and a territory share of 55% are two \
separate facts, and a sentence carrying both is true. Words like "but", "however", \
"despite" and "still" are the writer's emphasis, and assert nothing to check.
4. Match like quantity with like. Every reason in the evidence is labelled with \
its kind, and differently labelled numbers measure different things: an "origin" \
distance is how far from goal the shot was struck, a "creator" distance is how far \
the pass before it travelled. Never compare one against the other. Read which \
quantity the claim is actually about before you rule it wrong.
5. Rule unsupported or misattributed only when you can quote, in "contradicts", \
the specific fact or moment the claim gets wrong. If you cannot quote one, the \
verdict is supported: thin evidence is not an error, and it is not your place to \
demand more. An omission is not an error either -- a claim that names two of a \
team's three goals is not wrong about the third, and a claim is silent about far \
more than it says. Only a claim that states something false can be rejected. Read \
your quote back before you send it: if it sounds like agreement with the claim, \
you have not found a contradiction, and the verdict is supported. A ruling that \
says "unsupported" while its own reason confirms the claim contradicts itself and \
is discarded unread, so if your reason concludes the claim is right, the verdict \
is "supported".
6. Rule on every claim in the list, by its id. Never add a claim, never rewrite \
one, and never offer commentary of your own.
7. Reply with JSON only, in exactly this shape:
{"verdicts": [{"id": 0, "verdict": "supported", "reason": "one short sentence", \
"contradicts": ""}]}
"""


def build_verification_prompt(ctx: VerificationContext) -> str:
    """The draft, its claims, the evidence behind each cited event, and the index.

    The fact sheet is the *same* one the desk was handed -- `facts_for_prompt` is
    shared deliberately, not lazily. A critic checking the desk against a
    different set of facts would be answering a different question.

    `basis` exists because an empty `evidence` list is a trap. A claim grounded in
    the summary has no cited moments, and an empty list beside it reads as "no
    evidence to show" -- which is exactly backwards, since the fact sheet is
    plenty. Naming what the claim stands on removes the ambiguity instead of
    asking the model to remember a rule about it.

    `key_moments` is the index of every moment the desk was allowed to draw on.
    Without it the critic's world is only what a claim happened to cite, and a
    true sentence loses: "Northbridge scored three goals in the first 40 minutes,
    the first at minute 1 and the third at minute 40" cites two goals, asserts
    three, and was rejected because the second was invisible. The index is the
    same observable insight set the desk had, so this is not a second source --
    it is the first one, listed.
    """
    names = {
        ctx.summary.home: ctx.summary.home_name,
        ctx.summary.away: ctx.summary.away_name,
    }
    claims = []
    for persona, claim in claim_roster(ctx.draft):
        claims.append(
            {
                # The id is not this function's counter; it is the claim's own,
                # stamped by the contract. Two counters would drift.
                "id": claim.claim_id,
                "persona": persona.value,
                "claim": claim.text,
                "cites": list(claim.event_ids),
                "basis": "cited_moments" if claim.event_ids else "match_facts",
                "evidence": [
                    {
                        "moment": insight.headline,
                        "minute": insight.minute,
                        "team": names.get(insight.team, insight.team),
                        "player": insight.player_name,
                        # `kind` travels with the text because the text alone is
                        # ambiguous. A goal carries "14 m from goal" and "a 7 m
                        # pass"; without the label, a critic asked whether the
                        # goals came from passes inside 9 m compared the pass
                        # against the wrong one of those two numbers and rejected
                        # a true claim. The evidence keeps its own shape.
                        "why": [
                            {"kind": reason.kind.value, "text": reason.text}
                            for reason in insight.reasons
                        ],
                    }
                    for insight in evidence_for(claim, ctx.insights)
                ],
            }
        )
    return json.dumps(
        {
            "match_facts": facts_for_prompt(ctx.summary),
            "key_moments": [
                {
                    "minute": insight.minute,
                    "team": names.get(insight.team, insight.team),
                    "moment": insight.headline,
                }
                for insight in ctx.insights
            ],
            "claims": claims,
        },
        indent=2,
        sort_keys=True,
    )


class VerificationParseError(ValueError):
    """The critic's output was not the JSON object this stage requires."""


def parse_verdicts(raw: str, draft: RevealNarrative) -> list[ClaimVerdict]:
    """Turn the critic's JSON into validated rulings, one per claim id.

    Lenient by design, and lenient in one direction only: a ruling that cannot be
    read (an id that matches no claim, an unknown verdict word, a missing verdict)
    is *discarded*, which leaves the claim unmarked. The alternative -- treating an
    unreadable ruling as a verdict -- would let a confused model put a warning on a
    sentence it never actually judged, and the caller's `complete` flag already
    records that the review was partial.

    The same direction applies to a ruling that names nothing. A critic asked to
    quote the fact it is overriding will sometimes answer `unsupported` and then
    explain, in its own reason, that the claim is correct. That ruling contradicts
    itself, and `contradicts` is the field that makes the contradiction visible to
    a parser rather than to a reader: no quoted fact, no ruling.
    """
    try:
        data = json.loads(strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise VerificationParseError(f"not JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("verdicts"), list):
        raise VerificationParseError("missing 'verdicts' list")

    index = claim_index(draft)
    rulings: dict[int, ClaimVerdict] = {}
    for row in data["verdicts"]:
        if not isinstance(row, dict):
            continue
        idx = row.get("id")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx not in index:
            continue
        try:
            verdict = Verdict(str(row.get("verdict", "")).strip().lower())
        except ValueError:
            continue
        contradicts = str(row.get("contradicts") or "").strip()
        if verdict is not Verdict.SUPPORTED and len(contradicts) < MIN_CONTRADICTION:
            continue
        if idx in rulings:  # the first ruling on a claim is the one that counts
            continue
        persona, _ = index[idx]
        reason = str(row.get("reason") or "").strip() or "no reason given"
        rulings[idx] = ClaimVerdict(
            claim_id=idx,
            persona=persona,
            verdict=verdict,
            reason=reason,
            contradicts=contradicts,
        )
    return [rulings[i] for i in sorted(rulings)]


class ChatCompletionsVerifier:
    """The critic backed by a chat-completions endpoint.

    Every failure -- a dead socket, a timeout, unreadable JSON -- resolves to the
    attribution critic plus `fell_back=True`. A review that could not happen must
    not cost the commentary, and must not be reported as if it had happened.
    """

    def __init__(
        self,
        plan: HttpPlan,
        *,
        timeout_s: float = 30.0,
        max_tokens: int = 700,
        transport=None,
    ) -> None:
        self.plan = plan
        self.provider = plan.provider
        self.model = plan.model_label
        self._llm = ChatJsonClient(
            plan,
            timeout_s=timeout_s,
            temperature=VERIFIER_TEMPERATURE,
            max_tokens=max_tokens,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._llm.aclose()

    async def review(self, ctx: VerificationContext) -> Review:
        try:
            raw = await self._llm.complete(
                VERIFIER_SYSTEM_PROMPT, build_verification_prompt(ctx)
            )
            rulings = parse_verdicts(raw, ctx.draft)
        except Exception as exc:  # noqa: BLE001 -- any failure means "stand in"
            print(f"[verifier] {self.provider} call failed ({type(exc).__name__}: "
                  f"{exc}); using the attribution critic")
            return Review(
                provider=LOCAL_PROVIDER,
                model=LOCAL_MODEL,
                verdicts=local_verdicts(ctx),
                complete=True,
                by_model=False,
                fell_back=True,
            )
        return Review(
            provider=self.provider,
            model=self.model,
            verdicts=rulings,
            complete=len(rulings) == len(claim_roster(ctx.draft)),
            by_model=True,
            fell_back=False,
        )


# --------------------------------------------------------------------------- #
# Applying a review
# --------------------------------------------------------------------------- #
def apply_verdicts(draft: RevealNarrative, review: Review) -> RevealNarrative:
    """The narrative that reaches the viewer: the draft, with its rulings attached.

    Nothing is removed, and that is the design. An earlier version dropped every
    claim the critic ruled against, which made one thing look like another: a
    false rejection and a real finding both produced a missing sentence, and the
    only evidence of a mistake was an absence. Measured against live runs, the
    critic catches corrupted claims reliably and still disputes a true one now
    and then -- so the failure it must not have is the silent one.

    Attaching the ruling to the claim also puts both halves of the argument on
    screen together. The desk's sentence and the fact the critic cited against it
    are read side by side, which is the only arrangement in which a viewer can
    catch the critic being wrong. That is not a concession to a weak critic; it
    is what makes the critic's accuracy checkable by the person reading it.

    A ruling addressed to a claim that is not in the draft is dropped, because a
    verdict about nothing is nothing.
    """
    rulings = {ruling.claim_id: ruling for ruling in review.verdicts}
    takes = [
        take.model_copy(
            update={
                "claims": [
                    claim.model_copy(update={"ruling": rulings.get(claim.claim_id)})
                    for claim in take.claims
                ]
            }
        )
        for take in draft.personas
    ]
    disputed = sum(
        1
        for take in takes
        for claim in take.claims
        if claim.ruling is not None and claim.ruling.verdict is not Verdict.SUPPORTED
    )
    return draft.model_copy(
        update={
            "personas": takes,
            "verifier": review.provider,
            "verified": review.complete and review.by_model,
            "disputed_claims": disputed,
        }
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def verifier_from_env() -> Verifier:
    """The critic the environment asks for, or the attribution critic."""
    plan = provider_plan()
    if plan is None:
        return LocalVerifier()
    timeout = float(os.environ.get("TOUCHLINE_LLM_TIMEOUT_S", "30"))
    max_tokens = int(os.environ.get("TOUCHLINE_LLM_MAX_TOKENS", "700"))
    print(f"[verifier] provider={plan.provider} model={plan.model_label}")
    return ChatCompletionsVerifier(plan, timeout_s=timeout, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# A smoke test you can run by hand
# --------------------------------------------------------------------------- #
def _main() -> None:
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 81
    live = "--live" in sys.argv

    from narrator import (  # noqa: PLC0415 -- the CLI is the only user of this
        _context_for_seed,
        local_narrative,
        narrator_from_env,
    )

    narrative_ctx = _context_for_seed(seed, Reveal.FULL_TIME)
    if live:
        draft = asyncio.run(narrator_from_env().narrate(narrative_ctx))
    else:
        draft = local_narrative(narrative_ctx)

    critic = verifier_from_env() if live else LocalVerifier()
    review = asyncio.run(
        critic.review(
            VerificationContext(
                draft=draft,
                summary=narrative_ctx.summary,
                insights=narrative_ctx.insights,
            )
        )
    )

    print(f"\ndraft: provider={draft.provider} model={draft.model} "
          f"validated={draft.validated} dropped={draft.dropped_claims}")
    print(f"review: provider={review.provider} model={review.model} "
          f"complete={review.complete} by_model={review.by_model} "
          f"fell_back={review.fell_back}\n")
    index = claim_index(draft)
    for ruling in review.verdicts:
        mark = " " if ruling.verdict is Verdict.SUPPORTED else "x"
        _, claim = index[ruling.claim_id]
        print(f" {mark} [{ruling.claim_id}] {ruling.persona.value:<8} "
              f"{ruling.verdict.value:<14} {ruling.reason}")
        if ruling.contradicts:
            print(f"        overrides: {ruling.contradicts}")
        print(f"        {claim.text}")

    final = apply_verdicts(draft, review)
    print(f"\ndisputed={final.disputed_claims} verified={final.verified} "
          f"verifier={final.verifier} "
          f"(every claim still on the desk: {len(claim_roster(final))})")


if __name__ == "__main__":
    _main()
