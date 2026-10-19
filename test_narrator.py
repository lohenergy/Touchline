"""Invariant tests for the Narrator stage (`narrator.py`).

WHY THESE TESTS EXIST
---------------------
The Narrator is the one stage that lets a non-deterministic component -- a
language model -- write sentences a viewer will read as fact. Everything else in
the pipeline is reproducible; this is the seam where a confident-sounding lie
could enter. So the tests here are less about "does it work" and more about
"when it is wrong, is it caught".

Three things are pinned:

    1. THE INPUT GATE. No latent field name may ever appear in a serialised
       prompt. This is the leak guard: if someone later adds a convenience
       argument that carries `big_chance_true` into the context, this test fails
       before a single token is spent.

    2. THE OUTPUT GATES. A claim citing an event that does not exist is dropped;
       a claim quoting a number we never supplied is dropped; a claim asserting
       that one event caused another is dropped; a persona that loses every claim
       is rebuilt from the deterministic fallback rather than shown empty.

    3. THE FALLBACK IS GENUINELY VALID. The local narrator's own citations and
       numbers must clear the same gates the model is held to -- otherwise "the
       fallback" would be a loophole through which the very claims the gates exist
       to stop would pass.

The model is never called for real here. `httpx.MockTransport` drives the actual
request-and-parse path against scripted responses, so a network failure, a 500,
or a fenced-JSON reply are all tested without a key, a quota or a socket.

Run directly:      ./.venv/bin/python test_narrator.py
Or under pytest:   ./.venv/bin/python -m pytest test_narrator.py
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx

import narrator as narr
from narrator import (
    ChatCompletionsNarrator,
    HttpPlan,
    LocalNarrator,
    SYSTEM_PROMPT,
    _number_forms,
    _strip_fences,
    allowed_numbers,
    build_context,
    build_user_prompt,
    citable_event_ids,
    local_narrative,
)
from schemas import PersonaId, Reveal

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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def ft_context(seed: int = 81):
    """A real full-time context: summary + insights from the actual pipeline."""
    return narr._context_for_seed(seed, Reveal.FULL_TIME)


def plan() -> HttpPlan:
    return HttpPlan(
        provider="test",
        url="https://example.invalid/chat/completions",
        headers={"Content-Type": "application/json", "api-key": "test"},
        body_model="gpt-narrator",
        model_label="gpt-narrator",
    )


def scripted(reply_text: str, status: int = 200,
             recorder: list | None = None) -> ChatCompletionsNarrator:
    """A real narrator whose socket is a script that returns `reply_text`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        return httpx.Response(
            status, json={"choices": [{"message": {"content": reply_text}}]}
        )

    return ChatCompletionsNarrator(plan(), transport=httpx.MockTransport(handler))


def reply(analyst: list, diehard: list, numbers: list) -> str:
    return json.dumps(
        {
            "personas": [
                {"persona": "analyst", "headline": "The read", "claims": analyst},
                {"persona": "diehard", "headline": "Up the town", "claims": diehard},
                {"persona": "numbers", "headline": "By the numbers", "claims": numbers},
            ]
        }
    )


# --------------------------------------------------------------------------- #
# Gate 1: the input never carries latent truth
# --------------------------------------------------------------------------- #


def test_prompt_never_leaks_latent_truth() -> None:
    ctx = ft_context(81)
    prompt = SYSTEM_PROMPT + build_user_prompt(ctx)
    for key in LATENT_KEYS:
        assert key not in prompt, f"latent key {key!r} leaked into the prompt"
    # The moments we *do* expose are the observable kinds only.
    assert "deep_giveaway" in prompt or "goal" in prompt
    assert "big_chance" not in prompt


def test_prompt_exposes_citable_ids_by_name_keyed_facts() -> None:
    ctx = ft_context(81)
    prompt = build_user_prompt(ctx)
    for event_id in citable_event_ids(ctx.insights):
        assert event_id in prompt
    assert ctx.summary.home_name in prompt  # facts are keyed by team name


# --------------------------------------------------------------------------- #
# Gate 3: number forms
# --------------------------------------------------------------------------- #


def test_number_forms_cover_percent_and_rounding() -> None:
    forms = _number_forms(0.5612)
    assert {"56", "56.1", "0.56", "0.6", "1"} & forms or True  # sanity: non-empty
    for expected in ("56", "56.1", "0.56", "0.6"):
        assert expected in forms, f"{expected} missing from {sorted(forms)}"
    assert "5612" not in forms  # never offer a nonsense form
    assert _number_forms(True) == set()  # bool is not a number here
    assert "12" in _number_forms(12)
    assert "12" in _number_forms(12.4)


def test_allowed_numbers_follow_the_fact_sheet() -> None:
    ctx = ft_context(81)
    allowed = allowed_numbers(ctx)
    home = ctx.summary.home_name
    # The score is offered as "3-3" -> both digits must be usable.
    assert str(ctx.summary.score[ctx.summary.home]) in allowed
    # Possession is shown as an integer percent, so that integer must be usable.
    pct = round(ctx.summary.possession[ctx.summary.home] * 100)
    assert str(pct) in allowed
    # A number nobody supplied must not be.
    assert "9999" not in allowed
    assert home  # silence an unused-variable lint without weakening the test


# --------------------------------------------------------------------------- #
# Gate 2 & 3: the model's claims are audited
# --------------------------------------------------------------------------- #


def test_valid_reply_passes_and_is_flagged_validated() -> None:
    ctx = ft_context(81)
    anchor = ctx.insights[0].anchor_event_id
    model = scripted(
        reply(
            analyst=[{"text": "They pressed and won it back high.", "event_ids": [anchor]}],
            diehard=[{"text": "Never in doubt from where I was sitting.", "event_ids": []}],
            numbers=[{"text": "The xG gap told the story.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    assert out.validated is True
    assert out.dropped_claims == 0
    assert [p.persona for p in out.personas] == list(narr.PERSONA_ORDER)
    assert out.provider == "test"


def test_bad_citation_is_dropped_but_good_claim_survives() -> None:
    ctx = ft_context(81)
    anchor = ctx.insights[0].anchor_event_id
    model = scripted(
        reply(
            analyst=[
                {"text": "This one is real.", "event_ids": [anchor]},
                {"text": "This one cites a ghost.", "event_ids": ["LIVE-81-99999"]},
            ],
            diehard=[{"text": "Colour, no citation needed.", "event_ids": []}],
            numbers=[{"text": "Just the numbers.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    assert out.dropped_claims == 1
    analyst = next(p for p in out.personas if p.persona is PersonaId.ANALYST)
    texts = [c.text for c in analyst.claims]
    assert "This one is real." in texts
    assert "This one cites a ghost." not in texts
    assert out.validated is True  # one bad claim does not condemn the persona


def test_hallucinated_number_is_dropped() -> None:
    ctx = ft_context(81)
    model = scripted(
        reply(
            analyst=[{"text": "They had 99% of the ball.", "event_ids": []}],
            diehard=[{"text": "Loud and proud.", "event_ids": []}],
            numbers=[{"text": "The figures speak.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    assert out.dropped_claims >= 1
    analyst = next(p for p in out.personas if p.persona is PersonaId.ANALYST)
    assert all("99" not in c.text for c in analyst.claims)
    # The persona lost its only claim, so it was rebuilt from the fallback.
    assert out.validated is False


def test_causal_claim_is_dropped_and_plain_description_survives() -> None:
    ctx = ft_context(81)
    anchor = ctx.insights[0].anchor_event_id
    model = scripted(
        reply(
            analyst=[
                {"text": "That giveaway led to the goal.", "event_ids": [anchor]},
                {"text": "They won the ball back high.", "event_ids": [anchor]},
            ],
            diehard=[{"text": "Never in doubt from where I was sitting.", "event_ids": []}],
            numbers=[{"text": "The xG gap told the story.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    assert out.dropped_claims == 1
    analyst = next(p for p in out.personas if p.persona is PersonaId.ANALYST)
    texts = [c.text for c in analyst.claims]
    assert "That giveaway led to the goal." not in texts
    assert "They won the ball back high." in texts
    assert out.validated is True  # the other claim carried the persona


def test_causal_pattern_catches_the_overclaims_we_actually_saw() -> None:
    # Every one of these appeared in a live run against seed 81.
    seen = [
        "Northbridge took the lead, capitalizing on Riverside Athletic's early giveaways.",
        "Their passing accuracy helped them sustain pressure leading to multiple chances.",
        "Our team turned early Riverside errors into goals.",
        "The giveaway allowed Riverside to create scoring chances.",
        "They punished the mistake inside two minutes.",
        "The sending off was due to a reckless challenge.",
        "Sustained pressure forced the turnover.",
        "Three goals in seven minutes resulted in the comeback.",
        "Riverside equalised because Northbridge dropped deep.",
        "Giveaways deep in their own half contributed to Northbridge's goals.",
        "The error triggered a collapse.",
        "A quick restart sparked the move.",
    ]
    for text in seen:
        assert narr._CAUSAL_RE.search(text), text


def test_causal_pattern_leaves_plain_description_alone() -> None:
    # No connective, no causal claim -- these must all survive the gate.
    ok = [
        "Northbridge led 3-0 at the break.",
        "Riverside scored three times after the interval.",
        "The keeper saved it with his feet.",
        "They had 47% of the ball.",
        "Northbridge's three goals all came from inside the box.",
        "A red card and five substitutions late on.",
    ]
    for text in ok:
        assert not narr._CAUSAL_RE.search(text), text


def test_the_local_fallback_obeys_the_policy_it_enforces() -> None:
    # The deterministic desk is held to the same standard as the model, or the
    # fallback would be the one place a causal claim could still reach the panel.
    ctx = ft_context(81)
    for take in local_narrative(ctx).personas:
        for claim in take.claims:
            assert not narr._CAUSAL_RE.search(claim.text), claim.text


def test_moment_number_without_a_citation_is_dropped() -> None:
    # The exact live failure: 0.48 is the xG of a *saved* shot, so it is in the
    # supply, but an uncited sentence may only use the fact sheet's figures.
    ctx = ft_context(81)
    moment_xg = _a_moment_number(ctx)
    model = scripted(
        reply(
            analyst=[{"text": f"One goal came from a {moment_xg} xG shot.",
                      "event_ids": []}],
            diehard=[{"text": "Gutted.", "event_ids": []}],
            numbers=[{"text": "Level at the end.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    analyst = next(p for p in out.personas if p.persona is PersonaId.ANALYST)
    assert all(moment_xg not in c.text for c in analyst.claims)
    assert out.dropped_claims >= 1


def test_the_same_number_survives_once_the_moment_is_cited() -> None:
    ctx = ft_context(81)
    event_id, moment_xg = _a_moment_number_with_id(ctx)
    model = scripted(
        reply(
            analyst=[{"text": f"A {moment_xg} xG shot was saved.", "event_ids": [event_id]}],
            diehard=[{"text": "Gutted.", "event_ids": []}],
            numbers=[{"text": "Level at the end.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    analyst = next(p for p in out.personas if p.persona is PersonaId.ANALYST)
    assert any(moment_xg in c.text for c in analyst.claims), (
        "citing the moment must buy access to its figure"
    )


def test_summary_figures_need_no_citation() -> None:
    ctx = ft_context(81)
    pct = round(ctx.summary.possession[ctx.summary.home] * 100)
    model = scripted(
        reply(
            analyst=[{"text": f"They had {pct}% of the ball.", "event_ids": []}],
            diehard=[{"text": "Gutted.", "event_ids": []}],
            numbers=[{"text": "Level at the end.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    analyst = next(p for p in out.personas if p.persona is PersonaId.ANALYST)
    assert any(f"{pct}%" in c.text for c in analyst.claims)


def test_scoped_numbers_separate_summary_from_moments() -> None:
    ctx = ft_context(81)
    summary_numbers, numbers_for = narr._scoped_numbers(ctx)
    from schemas import NarrativeClaim as NC

    # A moment's figure is not available with no citation...
    moment_id, moment_xg = _a_moment_number_with_id(ctx)
    assert moment_xg not in summary_numbers
    assert moment_xg not in numbers_for(NC(text="x", event_ids=[]))
    # ...and is available once that moment is cited.
    assert moment_xg in numbers_for(NC(text="x", event_ids=[moment_id]))


def test_local_narrator_still_passes_the_scoped_number_gate() -> None:
    # The fallback must satisfy the stricter gate too, or the tightening would
    # silently break the offline demo path.
    ctx = ft_context(81)
    citable = citable_event_ids(ctx.insights)
    _, numbers_for = narr._scoped_numbers(ctx)
    for take in local_narrative(ctx).personas:
        survivors, rejected = narr._audit_claims(take.claims, citable, numbers_for)
        assert not rejected, rejected
        assert len(survivors) == len(take.claims)


def _a_moment_number(ctx) -> str:
    return _a_moment_number_with_id(ctx)[1]


def _a_moment_number_with_id(ctx) -> tuple[str, str]:
    """Find a figure that lives only in a moment, never in the fact sheet.

    Prefers a decimal, because that is the shape of the real failure this gate
    was written for -- a single shot's xG quoted on a sentence about something
    else. Falls back to any moment-only figure.
    """
    summary_numbers, _ = narr._scoped_numbers(ctx)
    fallback: tuple[str, str] | None = None
    for insight in ctx.insights:
        for reason in insight.reasons:
            for token in narr._NUM.findall(reason.text):
                if token in summary_numbers or not reason.event_ids:
                    continue
                if "." in token:
                    return reason.event_ids[0], token
                fallback = fallback or (reason.event_ids[0], token)
    if fallback is not None:
        return fallback
    raise AssertionError("no moment-only number found in this fixture")


def test_persona_losing_every_claim_is_backfilled_not_emptied() -> None:
    ctx = ft_context(81)
    model = scripted(
        reply(
            analyst=[{"text": "Fine.", "event_ids": []}],
            diehard=[{"text": "95% bull.", "event_ids": []}],   # hallucinated number
            numbers=[{"text": "Also fine.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    diehard = next(p for p in out.personas if p.persona is PersonaId.DIEHARD)
    assert diehard.claims, "a persona must never render empty"
    assert out.validated is False


def test_empty_event_ids_claim_is_allowed() -> None:
    ctx = ft_context(81)
    model = scripted(
        reply(
            analyst=[{"text": "A match of two halves.", "event_ids": []}],
            diehard=[{"text": "Gutted.", "event_ids": []}],
            numbers=[{"text": "Numbers, plainly.", "event_ids": []}],
        )
    )
    out = asyncio.run(model.narrate(ctx))
    assert out.validated is True
    assert out.dropped_claims == 0


# --------------------------------------------------------------------------- #
# Failures degrade to the deterministic narrator
# --------------------------------------------------------------------------- #


def test_unparseable_reply_falls_back_to_local() -> None:
    ctx = ft_context(81)
    out = asyncio.run(scripted("I am afraid I cannot do that.").narrate(ctx))
    assert out.provider == "local"
    assert out.validated is False


def test_fenced_json_is_accepted() -> None:
    ctx = ft_context(81)
    body = reply(
        analyst=[{"text": "Neat.", "event_ids": []}],
        diehard=[{"text": "Neat.", "event_ids": []}],
        numbers=[{"text": "Neat.", "event_ids": []}],
    )
    out = asyncio.run(scripted(f"```json\n{body}\n```").narrate(ctx))
    assert out.validated is True
    assert _strip_fences(f"```json\n{body}\n```").startswith("{")


def test_http_error_falls_back_to_local() -> None:
    ctx = ft_context(81)
    out = asyncio.run(scripted("boom", status=500).narrate(ctx))
    assert out.provider == "local"
    assert out.validated is False


# --------------------------------------------------------------------------- #
# The fallback is held to the same standards
# --------------------------------------------------------------------------- #


def test_local_narrative_citations_all_resolve() -> None:
    ctx = ft_context(81)
    narrative = local_narrative(ctx)
    known = citable_event_ids(ctx.insights)
    assert len(narrative.personas) == 3
    for take in narrative.personas:
        for claim in take.claims:
            for event_id in claim.event_ids:
                assert event_id in known, f"{event_id} is not a citable event"


def test_local_narrative_numbers_are_all_grounded() -> None:
    ctx = ft_context(81)
    allowed = allowed_numbers(ctx)
    for take in local_narrative(ctx).personas:
        for claim in take.claims:
            for token in narr._NUM.findall(claim.text):
                assert token in allowed, f"{token!r} in {claim.text!r} was not supplied"


def test_local_narrator_is_deterministic() -> None:
    ctx = ft_context(81)
    first = asyncio.run(LocalNarrator().narrate(ctx))
    second = asyncio.run(LocalNarrator().narrate(ctx))
    assert first.model_dump() == second.model_dump()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_azure_plan_uses_v1_path_when_no_api_version() -> None:
    saved = {k: os.environ.get(k) for k in
             ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
              "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION")}
    try:
        os.environ.update(
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
            AZURE_OPENAI_API_KEY="k",
            AZURE_OPENAI_DEPLOYMENT="gpt-narrator",
            AZURE_OPENAI_API_VERSION="",
        )
        p = narr._azure_plan()
        assert p.url == "https://example.openai.azure.com/openai/v1/chat/completions"
        assert p.headers["api-key"] == "k"
        assert p.body_model == "gpt-narrator"

        os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"
        p2 = narr._azure_plan()
        assert p2.url == (
            "https://example.openai.azure.com/openai/deployments/gpt-narrator"
            "/chat/completions?api-version=2025-01-01-preview"
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_normalize_endpoint_strips_a_pasted_operation_path() -> None:
    cases = {
        # The exact bug: the Foundry portal copies the full operation URL.
        "https://res.services.ai.azure.com/openai/v1/responses": "https://res.services.ai.azure.com",
        "https://res.services.ai.azure.com/openai/v1/": "https://res.services.ai.azure.com",
        "https://res.services.ai.azure.com/openai/v1/chat/completions": "https://res.services.ai.azure.com",
        # A Foundry *project* endpoint is a different path on the same host.
        "https://res.services.ai.azure.com/api/projects/myproj": "https://res.services.ai.azure.com",
        # Already-clean forms are left alone, trailing slash and all.
        "https://res.services.ai.azure.com": "https://res.services.ai.azure.com",
        "https://res.services.ai.azure.com/": "https://res.services.ai.azure.com",
        # The host may legitimately contain ".openai." -- that is a domain, not
        # a path, and must survive untouched.
        "https://football-predictor.openai.azure.com/": "https://football-predictor.openai.azure.com",
        "https://football-predictor.openai.azure.com/openai/deployments/d/chat/completions":
            "https://football-predictor.openai.azure.com",
    }
    for raw, want in cases.items():
        assert narr._normalize_endpoint(raw) == want, raw


def test_azure_plan_survives_the_pasted_full_url() -> None:
    saved = {k: os.environ.get(k) for k in
             ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
              "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION")}
    try:
        os.environ.update(
            AZURE_OPENAI_ENDPOINT="https://res.services.ai.azure.com/openai/v1/responses",
            AZURE_OPENAI_API_KEY="k",
            AZURE_OPENAI_DEPLOYMENT="gpt-narrator",
            AZURE_OPENAI_API_VERSION="",
        )
        p = narr._azure_plan()
        assert p.url == "https://res.services.ai.azure.com/openai/v1/chat/completions"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_temperature_is_low_by_default_and_configurable() -> None:
    ctx = ft_context(81)
    default = ChatCompletionsNarrator(plan())
    assert default._payload(ctx, True)["temperature"] == 0.3
    hot = ChatCompletionsNarrator(plan(), temperature=0.9)
    assert hot._payload(ctx, True)["temperature"] == 0.9
    asyncio.run(default.aclose())
    asyncio.run(hot.aclose())


def test_from_env_reads_temperature_and_builds_an_azure_plan() -> None:
    saved = {k: os.environ.get(k) for k in
             ("TOUCHLINE_LLM_PROVIDER", "AZURE_OPENAI_ENDPOINT",
              "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
              "TOUCHLINE_LLM_TEMPERATURE")}
    try:
        os.environ.update(
            TOUCHLINE_LLM_PROVIDER="azure",
            AZURE_OPENAI_ENDPOINT="https://res.services.ai.azure.com/openai/v1/responses",
            AZURE_OPENAI_API_KEY="k",
            AZURE_OPENAI_DEPLOYMENT="gpt-narrator",
            TOUCHLINE_LLM_TEMPERATURE="0.15",
        )
        built = narr.narrator_from_env()
        assert isinstance(built, ChatCompletionsNarrator)
        assert built.temperature == 0.15
        assert built.plan.url.endswith("/openai/v1/chat/completions")
        asyncio.run(built.aclose())
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_unknown_provider_falls_back_to_local() -> None:
    saved = os.environ.get("TOUCHLINE_LLM_PROVIDER")
    try:
        os.environ["TOUCHLINE_LLM_PROVIDER"] = "wat"
        assert isinstance(narr.narrator_from_env(), LocalNarrator)
    finally:
        if saved is None:
            os.environ.pop("TOUCHLINE_LLM_PROVIDER", None)
        else:
            os.environ["TOUCHLINE_LLM_PROVIDER"] = saved


def test_request_carries_model_and_system_prompt() -> None:
    ctx = ft_context(81)
    sent: list = []
    model = scripted(
        reply(
            analyst=[{"text": "Ok.", "event_ids": []}],
            diehard=[{"text": "Ok.", "event_ids": []}],
            numbers=[{"text": "Ok.", "event_ids": []}],
        ),
        recorder=sent,
    )
    asyncio.run(model.narrate(ctx))
    body = json.loads(sent[0].content.decode())
    assert body["model"] == "gpt-narrator"
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert "MATCH FACTS" in body["messages"][1]["content"]


def test_build_context_is_observable_only() -> None:
    ctx = ft_context(81)
    assert isinstance(ctx.summary, object)
    dumped = build_context(
        ctx.match_id, ctx.reveal, ctx.clock_s, ctx.summary, ctx.insights
    ).model_dump_json()
    for key in LATENT_KEYS:
        assert key not in dumped


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
    print(f"\n{len(tests) - failures}/{len(tests)} narrator tests passed")
    raise SystemExit(1 if failures else 0)
