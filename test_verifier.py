"""Tests for the critic (`verifier.py`).

WHY THESE TESTS EXIST
---------------------
The critic is the one stage whose failure mode is silent. A dead renderer drops a
card and the trace says so; a dead model drops a panel. A critic that quietly
approves everything looks *exactly* like a critic that found nothing wrong, and
the whole claim of this project -- that a system can score its own explanations --
rests on the difference being visible.

So these tests protect three things:

  1. A ruling joins back to the sentence it judged. Verdicts are matched by index,
     and an index is only meaningful if the roster order is stable and the claim
     text travels with the ruling.
  2. A critic that cannot read a ruling *keeps the claim*. Leniency points one
     way on purpose: an unreadable ruling must never become a quiet deletion.
  3. Failure is reported, not absorbed. A critic that fell back, a review that
     did not cover every claim and the deterministic critic itself are three
     different things, and `verified` must be False for the first two.

Run directly:      ./.venv/bin/python test_verifier.py
Or under pytest:   ./.venv/bin/python -m pytest test_verifier.py
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib

import httpx

import narrator as narr
from llm import HttpPlan
from narrator import local_narrative
from schemas import (
    ClaimVerdict,
    NarrativeClaim,
    PersonaId,
    PersonaTake,
    Reveal,
    RevealNarrative,
    Verdict,
)
from verifier import (
    VERIFIER_SYSTEM_PROMPT,
    ChatCompletionsVerifier,
    LocalVerifier,
    Review,
    VerificationContext,
    VerificationParseError,
    apply_verdicts,
    build_verification_prompt,
    claim_roster,
    evidence_for,
    local_verdicts,
    parse_verdicts,
)

LATENT_KEYS = (
    "control_true",
    "chaos_true",
    "chance_created_true",
    "big_chance_true",
    "error_true",
    "pass_type_true",
    "from_error_true",
    "dominant_team",
)

SEED = 81


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def ft_context():
    """A real full-time context: the summary and insights of a real match."""
    return narr._context_for_seed(SEED, Reveal.FULL_TIME)


def a_home_moment(ctx):
    return next(i for i in ctx.insights if i.team == ctx.summary.home)


def an_away_moment(ctx):
    return next(i for i in ctx.insights if i.team == ctx.summary.away)


def review_context(draft, ctx) -> VerificationContext:
    return VerificationContext(draft=draft, summary=ctx.summary, insights=ctx.insights)


def run(coro):
    return asyncio.run(coro)


def attr_review(draft, ctx) -> Review:
    return run(LocalVerifier().review(review_context(draft, ctx)))


def draft_saying(persona: PersonaId, text: str, event_ids: list[str]):
    """A one-chair draft with a single hand-written claim, for surgical cases."""
    ctx = ft_context()
    take = PersonaTake(
        persona=persona,
        headline="A headline.",
        claims=[NarrativeClaim(text=text, event_ids=event_ids)],
    )
    base = local_narrative(ctx)
    return ctx, base.model_copy(update={"personas": [take]})


def controlled_draft():
    """Three claims in one chair, so what a rejection removed is unambiguous."""
    ctx = ft_context()
    take = PersonaTake(
        persona=PersonaId.ANALYST,
        headline="Read.",
        claims=[
            NarrativeClaim(text="First claim.", event_ids=[a_home_moment(ctx).anchor_event_id]),
            NarrativeClaim(text="Second claim.", event_ids=[an_away_moment(ctx).anchor_event_id]),
            NarrativeClaim(text="Third claim.", event_ids=[]),
        ],
    )
    base = local_narrative(ctx)
    # Rebuild through the constructor rather than `model_copy`, so the roster
    # validator runs and numbers the three claims 0, 1, 2. A copied draft keeps
    # the source ids, which would give every claim here the same id and make the
    # "which claim did the critic rule on?" assertions meaningless.
    return ctx, RevealNarrative(**{**base.model_dump(), "personas": [take.model_dump()]})


def fallbacks(ctx):
    return {take.persona: take for take in local_narrative(ctx).personas}


def ruled(*pairs) -> Review:
    """A model review that ruled `supported` on nothing and `pairs` as given."""
    return Review(
        provider="azure",
        model="gpt-narrator",
        verdicts=list(pairs),
        complete=True,
        by_model=True,
    )


def reject(persona: PersonaId, claim: NarrativeClaim, reason="invented") -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim.claim_id,
        persona=persona,
        verdict=Verdict.UNSUPPORTED,
        reason=reason,
        contradicts="the match facts say otherwise",
    )


# --------------------------------------------------------------------------- #
# 1. The context, and what it can and cannot hold
# --------------------------------------------------------------------------- #
def test_the_critic_is_handed_a_draft_a_scoreline_and_nothing_else() -> None:
    """The input gate, stated structurally rather than promised in a comment."""
    assert set(VerificationContext.model_fields) == {"draft", "summary", "insights"}


def test_the_verifier_prompt_carries_no_latent_truth() -> None:
    ctx, draft = controlled_draft()
    prompt = build_verification_prompt(review_context(draft, ctx))
    for key in LATENT_KEYS:
        assert key not in prompt, key
    assert "dominant" not in prompt


def test_evidence_is_exactly_what_the_claim_cited() -> None:
    ctx = ft_context()
    moment = a_home_moment(ctx)
    claim = NarrativeClaim(text="A claim.", event_ids=[moment.anchor_event_id])
    cited = evidence_for(claim, ctx.insights)
    assert [i.anchor_event_id for i in cited] == [moment.anchor_event_id]


def test_an_uncited_claim_stands_on_nothing_but_the_facts() -> None:
    ctx = ft_context()
    claim = NarrativeClaim(text="A summary-level claim.", event_ids=[])
    assert evidence_for(claim, ctx.insights) == []


def test_a_claim_is_also_matched_through_its_evidence_ids() -> None:
    """A goal insight cites its assist; a claim citing only the assist still finds it."""
    ctx = ft_context()
    goal = next(i for i in ctx.insights if len(i.evidence_event_ids) > 1)
    assist = [e for e in goal.evidence_event_ids if e != goal.anchor_event_id][0]
    claim = NarrativeClaim(text="A claim.", event_ids=[assist])
    assert goal in evidence_for(claim, ctx.insights)


def test_the_roster_index_is_the_join_key_between_a_ruling_and_a_sentence() -> None:
    ctx, draft = controlled_draft()
    roster = claim_roster(draft)
    assert [claim.text for _, claim in roster] == [
        "First claim.", "Second claim.", "Third claim.",
    ]


# --------------------------------------------------------------------------- #
# 2. Parsing a model's rulings without letting it delete quietly
# --------------------------------------------------------------------------- #
def test_a_ruling_joins_back_to_the_sentence_it_judged() -> None:
    ctx, draft = draft_and_live()
    roster = claim_roster(draft)
    raw = json.dumps({"verdicts": [
        {"id": 1, "verdict": "unsupported", "reason": "no",
         "contradicts": "the facts say otherwise"},
    ]})
    rulings = parse_verdicts(raw, draft)
    assert len(rulings) == 1
    assert rulings[0].persona is roster[1][0]
    assert rulings[0].claim_id == roster[1][1].claim_id
    assert rulings[0].verdict is Verdict.UNSUPPORTED
    assert rulings[0].reason == "no"


def draft_and_live():
    ctx = ft_context()
    return ctx, local_narrative(ctx)


def test_fenced_json_is_accepted() -> None:
    _, draft = draft_and_live()
    body = json.dumps({"verdicts": [{"id": 0, "verdict": "supported", "reason": "ok"}]})
    rulings = parse_verdicts(f"```json\n{body}\n```", draft)
    assert rulings[0].verdict is Verdict.SUPPORTED


def test_an_unreadable_ruling_leaves_the_claim_standing() -> None:
    """The single most important direction of failure in this module."""
    _, draft = draft_and_live()
    raw = json.dumps({
        "verdicts": [
            {"id": 999, "verdict": "unsupported", "reason": "out of range"},
            {"id": 0, "verdict": "maybe", "reason": "not a verdict",
             "contradicts": "a well-quoted fact"},
            "not an object",
            {"verdict": "unsupported", "reason": "no id",
             "contradicts": "a well-quoted fact"},
            {"id": 0, "verdict": "unsupported", "reason": "the only usable one",
             "contradicts": "the facts say otherwise"},
        ]
    })
    rulings = parse_verdicts(raw, draft)
    assert len(rulings) == 1
    assert rulings[0].verdict is Verdict.UNSUPPORTED
    assert rulings[0].reason == "the only usable one"


def test_a_duplicate_ruling_keeps_the_first() -> None:
    _, draft = draft_and_live()
    raw = json.dumps({"verdicts": [
        {"id": 0, "verdict": "unsupported", "reason": "first",
         "contradicts": "the facts say the opposite"},
        {"id": 0, "verdict": "supported", "reason": "second"},
    ]})
    rulings = parse_verdicts(raw, draft)
    assert len(rulings) == 1
    assert rulings[0].reason == "first"


def test_a_rejection_that_quotes_nothing_is_not_honoured() -> None:
    """A verdict and its own reason disagreeing is a rejection with no evidence.

    Observed live, verbatim: `unsupported`, reason "the claim is correct, but the
    evidence is missing; however, since the claim cites match facts and the facts
    confirm it, it is supported". The label lost; the reason is what the critic
    actually knows, so a ruling that quotes nothing does not get to delete a
    sentence.
    """
    _, draft = draft_and_live()
    raw = json.dumps({"verdicts": [
        {"id": 0, "verdict": "unsupported", "reason": "the facts confirm it"},
        {"id": 1, "verdict": "unsupported", "reason": "no", "contradicts": "  "},
        {"id": 2, "verdict": "misattributed", "reason": "wrong team"},
    ]})
    assert parse_verdicts(raw, draft) == []


def test_a_rejection_that_quotes_the_fact_is_honoured() -> None:
    _, draft = draft_and_live()
    raw = json.dumps({"verdicts": [
        {"id": 0, "verdict": "unsupported", "reason": "possession was not 60%",
         "contradicts": "possession_pct was 50-50"},
    ]})
    rulings = parse_verdicts(raw, draft)
    assert rulings[0].verdict is Verdict.UNSUPPORTED
    assert rulings[0].contradicts == "possession_pct was 50-50"


def test_a_supported_claim_needs_no_contradiction_to_quote() -> None:
    _, draft = draft_and_live()
    raw = json.dumps({"verdicts": [
        {"id": 0, "verdict": "supported", "reason": "the evidence shows it"},
    ]})
    rulings = parse_verdicts(raw, draft)
    assert rulings[0].verdict is Verdict.SUPPORTED
    assert rulings[0].contradicts == ""


def test_the_attribution_critic_quotes_what_it_overrides() -> None:
    ctx = ft_context()
    moment = a_home_moment(ctx)
    claim = f"{ctx.summary.away_name} scored the opener."
    ctx, draft = draft_saying(PersonaId.DIEHARD, claim, [moment.anchor_event_id])
    ruling = attr_review(draft, ctx).verdicts[0]
    assert ruling.contradicts
    assert ctx.summary.home_name in ruling.contradicts


def test_a_missing_verdicts_list_is_an_error_not_an_empty_approval() -> None:
    _, draft = draft_and_live()
    for raw in ("not json", "[]", '{"rulings": []}'):
        try:
            parse_verdicts(raw, draft)
        except VerificationParseError:
            continue
        raise AssertionError(f"{raw!r} should have been rejected as unparseable")


# --------------------------------------------------------------------------- #
# 3. The deterministic critic: attribution, and only attribution
# --------------------------------------------------------------------------- #
def test_the_attribution_critic_rules_on_every_claim() -> None:
    ctx, draft = draft_and_live()
    review = attr_review(draft, ctx)
    assert review.provider == "local"
    assert review.complete is True
    assert review.by_model is False
    assert review.fell_back is False
    assert len(review.verdicts) == len(claim_roster(draft))


def test_the_attribution_critic_catches_a_misattributed_team() -> None:
    ctx = ft_context()
    moment = a_home_moment(ctx)
    claim = f"{ctx.summary.away_name} scored the opener."
    ctx, draft = draft_saying(PersonaId.DIEHARD, claim, [moment.anchor_event_id])
    ruling = attr_review(draft, ctx).verdicts[0]
    assert ruling.verdict is Verdict.MISATTRIBUTED
    assert ctx.summary.away_name in ruling.reason
    assert ctx.summary.home_name in ruling.contradicts


def test_the_attribution_critic_leaves_a_correct_attribution_alone() -> None:
    ctx = ft_context()
    moment = a_home_moment(ctx)
    claim = f"{ctx.summary.home_name} scored the opener."
    ctx, draft = draft_saying(PersonaId.ANALYST, claim, [moment.anchor_event_id])
    assert attr_review(draft, ctx).verdicts[0].verdict is Verdict.SUPPORTED


def test_naming_both_teams_is_not_a_misattribution() -> None:
    """A sentence that mentions both sides cannot have one wrong attribution."""
    ctx = ft_context()
    moment = a_home_moment(ctx)
    claim = (f"{ctx.summary.away_name} responded after {ctx.summary.home_name} "
             "opened the scoring.")
    ctx, draft = draft_saying(PersonaId.ANALYST, claim, [moment.anchor_event_id])
    assert attr_review(draft, ctx).verdicts[0].verdict is Verdict.SUPPORTED


def test_a_claim_grounded_in_the_summary_has_no_attribution_to_check() -> None:
    ctx = ft_context()
    claim = f"{ctx.summary.away_name} finished with more expected goals."
    ctx, draft = draft_saying(PersonaId.NUMBERS, claim, [])
    assert attr_review(draft, ctx).verdicts[0].verdict is Verdict.SUPPORTED


def test_the_attribution_critic_never_claims_to_have_read_the_sentence() -> None:
    """Its own name says what it is, because `provider` is what a viewer sees."""
    assert LocalVerifier.model == "touchline-attribution-v1"
    assert "attribution" in LocalVerifier.model


# --------------------------------------------------------------------------- #
# 4. Applying a review
# --------------------------------------------------------------------------- #
def test_a_rejected_claim_is_removed_counted_and_kept_in_the_record() -> None:
    ctx, draft = controlled_draft()
    persona, claim = claim_roster(draft)[1]
    review = ruled(reject(persona, claim))
    final = apply_verdicts(draft, review)

    # The claim is still on the desk, now marked with its ruling
    assert len(final.personas[0].claims) == 3
    assert final.personas[0].claims[1].ruling is not None
    assert final.personas[0].claims[1].ruling.verdict is Verdict.UNSUPPORTED
    assert final.disputed_claims == 1
    assert final.verifier == "azure"
    assert final.verified is True
    # The ruling travels with the claim.
    assert final.personas[0].claims[1].ruling is not None
    assert final.personas[0].claims[1].ruling.verdict is Verdict.UNSUPPORTED


def test_applying_a_review_does_not_mutate_the_draft() -> None:
    ctx, draft = controlled_draft()
    before = draft.model_dump()
    apply_verdicts(draft, ruled(reject(*claim_roster(draft)[1])))
    assert draft.model_dump() == before


def test_the_deterministic_critic_never_marks_a_reveal_verified() -> None:
    """`verified` mirrors `validated`: a model has to have been involved."""
    ctx, draft = draft_and_live()
    final = apply_verdicts(draft, attr_review(draft, ctx))
    assert final.verified is False
    assert final.verifier == "local"


def test_a_review_that_did_not_cover_every_claim_is_not_verified() -> None:
    ctx, draft = controlled_draft()
    partial = Review(provider="azure", model="gpt-narrator", verdicts=[], complete=False,
                     by_model=True)
    final = apply_verdicts(draft, partial)
    assert final.verified is False
    assert final.disputed_claims == 0
    assert len(final.personas[0].claims) == 3   # an unruled claim stands


def test_a_chair_that_loses_every_claim_is_refilled_not_silenced() -> None:
    ctx, draft = controlled_draft()
    persona, _ = claim_roster(draft)[0]
    review = ruled(*[reject(persona, claim) for _, claim in claim_roster(draft)])
    final = apply_verdicts(draft, review)
    assert final.disputed_claims == 3
    # Claims still there, with rulings attached
    assert len(final.personas[0].claims) == 3
    assert final.personas[0].claims[0].ruling is not None
    assert final.personas[0].headline == "Read."   # the chair is not replaced


def test_a_review_with_nothing_to_say_changes_nothing() -> None:
    ctx, draft = draft_and_live()
    empty = Review(provider="none", model="none")
    final = apply_verdicts(draft, empty)
    assert final.model_dump(exclude={"verifier", "verified", "rejected_claims", "claim_verdicts"}) \
        == draft.model_dump(exclude={"verifier", "verified", "rejected_claims", "claim_verdicts"})
    assert final.verifier == "none"
    assert final.verified is False


# --------------------------------------------------------------------------- #
# 5. The model critic, over a scripted socket
# --------------------------------------------------------------------------- #
def plan() -> HttpPlan:
    return HttpPlan(
        provider="azure",
        url="https://example.invalid/v1/chat/completions",
        headers={"Content-Type": "application/json", "api-key": "secret"},
        body_model="gpt-narrator",
        model_label="gpt-narrator",
    )


def scripted(reply_text: str, status: int = 200, recorder: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        return httpx.Response(status, json={"choices": [{"message": {"content": reply_text}}]})

    return ChatCompletionsVerifier(plan(), transport=httpx.MockTransport(handler))


def test_a_full_model_review_is_verified() -> None:
    ctx, draft = controlled_draft()
    body = json.dumps({"verdicts": [
        {"id": 0, "verdict": "supported", "reason": "fine"},
        {"id": 1, "verdict": "misattributed", "reason": "wrong side",
         "contradicts": "the cited moments are Northbridge's"},
        {"id": 2, "verdict": "supported", "reason": "fine"},
    ]})
    review = run(scripted(body).review(review_context(draft, ctx)))
    assert review.provider == "azure"
    assert review.model == "gpt-narrator"
    assert review.by_model is True
    assert review.complete is True
    assert review.fell_back is False
    assert review.verdicts[1].verdict is Verdict.MISATTRIBUTED

    final = apply_verdicts(draft, review)
    assert final.verified is True
    assert final.disputed_claims == 1
    # Claims are kept: the disputed one stays on the desk with its ruling attached,
    # rather than vanishing and leaving only a count behind.
    assert [c.text for c in final.personas[0].claims] == [
        "First claim.", "Second claim.", "Third claim."
    ]
    assert final.personas[0].claims[1].ruling is not None
    assert final.personas[0].claims[1].ruling.verdict is Verdict.MISATTRIBUTED


def test_the_critic_runs_cold() -> None:
    """A judge that varies run to run is noise wearing a verdict's clothes."""
    ctx, draft = controlled_draft()
    sent: list = []
    run(scripted('{"verdicts": []}', recorder=sent).review(review_context(draft, ctx)))
    assert json.loads(sent[0].content)["temperature"] == 0.0
    assert json.loads(sent[0].content)["model"] == "gpt-narrator"


def test_a_dead_critic_falls_back_and_says_so() -> None:
    ctx, draft = controlled_draft()
    review = run(scripted("", status=500).review(review_context(draft, ctx)))
    assert review.fell_back is True
    assert review.by_model is False
    assert review.provider == "local"
    assert review.verdicts == local_verdicts(review_context(draft, ctx))

    final = apply_verdicts(draft, review)
    assert final.verified is False
    assert final.disputed_claims == 0
    assert final.verifier == "local"


def test_an_unparseable_critic_falls_back_rather_than_approving() -> None:
    ctx, draft = controlled_draft()
    review = run(scripted("I could not review this.").review(review_context(draft, ctx)))
    assert review.fell_back is True
    assert review.complete is True          # the attribution critic did cover it
    assert all(v.verdict is Verdict.SUPPORTED for v in review.verdicts)


def test_the_prompt_hands_the_critic_the_evidence_for_each_claim_it_judges() -> None:
    ctx = ft_context()
    cited = a_home_moment(ctx)
    uncited = an_away_moment(ctx)
    _, draft = draft_saying(PersonaId.ANALYST, "One claim.", [cited.anchor_event_id])
    payload = json.loads(build_verification_prompt(review_context(draft, ctx)))

    assert payload["match_facts"]["match"]["home"] == ctx.summary.home_name
    claim = payload["claims"][0]
    assert claim["claim"] == "One claim."
    assert claim["basis"] == "cited_moments"
    assert [e["moment"] for e in claim["evidence"]] == [cited.headline]
    # A moment the claim did not cite is absent from *its* evidence, but present
    # in the index: a claim that cites two goals may still be right about a third.
    assert uncited.headline not in json.dumps(claim)
    assert uncited.headline in json.dumps(payload["key_moments"])


def test_the_index_lists_every_moment_the_desk_could_draw_on() -> None:
    ctx, draft = draft_and_live()
    payload = json.loads(build_verification_prompt(review_context(draft, ctx)))
    assert len(payload["key_moments"]) == len(ctx.insights)
    assert all(set(m) == {"minute", "team", "moment"} for m in payload["key_moments"])
    assert all(key not in json.dumps(payload["key_moments"]) for key in LATENT_KEYS)


def test_evidence_keeps_the_kind_of_every_reason() -> None:
    """A shot's distance from goal and its assist's length are not the same number.

    Flattening the reasons to bare strings cost a true claim on a live run: the
    critic compared an "origin" distance against a "creator" one and rejected a
    sentence that was right.
    """
    ctx = ft_context()
    goal = next(i for i in ctx.insights if i.kind.value == "goal")
    _, draft = draft_saying(PersonaId.ANALYST, "A goal.", [goal.anchor_event_id])
    payload = json.loads(build_verification_prompt(review_context(draft, ctx)))
    why = payload["claims"][0]["evidence"][0]["why"]
    assert [w["kind"] for w in why] == [r.kind.value for r in goal.reasons]
    assert all(w["text"] for w in why)
    assert "origin" in {w["kind"] for w in why}


def test_a_claim_with_no_citations_is_told_the_fact_sheet_is_its_evidence() -> None:
    """An empty evidence list must not read as "no evidence exists"."""
    ctx = ft_context()
    _, draft = draft_saying(PersonaId.NUMBERS, "A summary claim.", [])
    payload = json.loads(build_verification_prompt(review_context(draft, ctx)))
    assert payload["claims"][0]["basis"] == "match_facts"
    assert payload["claims"][0]["evidence"] == []


def test_the_critic_is_told_it_is_checking_not_writing() -> None:
    assert "do not write commentary" in VERIFIER_SYSTEM_PROMPT
    assert "Never add a claim" in VERIFIER_SYSTEM_PROMPT
    # The partisan chair must survive a literal-minded critic.
    assert "partisanship" in VERIFIER_SYSTEM_PROMPT
    # And the guard that keeps a true sentence on air: no quoted fact, no
    # rejection.
    assert "discarded unread" in VERIFIER_SYSTEM_PROMPT
    assert '"contradicts"' in VERIFIER_SYSTEM_PROMPT
    # Two figures in one sentence are not a contradiction, and an opinion clause
    # must not sink the fact beside it. Both rules exist because the critic got
    # exactly these wrong on a live run.
    assert "are not a contradiction" in VERIFIER_SYSTEM_PROMPT
    assert "do not sink it" in VERIFIER_SYSTEM_PROMPT
    # Like quantity with like: the rule that a shot's distance from goal is not
    # the length of the pass that made it.
    assert "Match like quantity with like" in VERIFIER_SYSTEM_PROMPT
    # And the index: a claim citing two goals may be right about a third.
    assert "key_moments" in VERIFIER_SYSTEM_PROMPT
    # An omission is not a falsehood. Observed live: a claim naming the first and
    # third goals was rejected for not mentioning the second.
    assert "An omission is not an error" in VERIFIER_SYSTEM_PROMPT


def test_the_critic_module_cannot_name_a_latent_quantity() -> None:
    """Nothing in this stage may even mention the truth it is scored against."""
    source = (pathlib.Path(__file__).parent / "verifier.py").read_text()
    for key in LATENT_KEYS:
        assert key not in source, key


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} verifier tests passed")
    raise SystemExit(1 if failures else 0)
