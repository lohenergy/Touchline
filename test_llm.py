"""Tests for the provider layer (`llm.py`) -- the one module that goes online.

WHY THESE TESTS EXIST
---------------------
Two agents speak to a model now, and every bug that would be *expensive* rather
than *visible* lives in this file's subject:

  * the endpoint. Pasting a full operation URL out of the Foundry portal used to
    produce an opaque 404 that looks exactly like a missing deployment.
  * the retry. `response_format` is a real 400 on some endpoints and the fix is
    to ask again without it -- but a 400 about something *else* must not be
    quietly retried into a second charge.
  * the failure. A half-configured provider is not an exception the demo dies
    of; it is a reason to stay offline, so `provider_plan()` returns None rather
    than raising.

Run directly:      ./.venv/bin/python test_llm.py
Or under pytest:   ./.venv/bin/python -m pytest test_llm.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os

import httpx

from llm import (
    ChatJsonClient,
    HttpPlan,
    azure_plan,
    github_plan,
    normalize_endpoint,
    provider_plan,
)

AZURE_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)


@contextlib.contextmanager
def env(**values):
    """Set environment variables for one test, then put the world back."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def azure_env(endpoint: str = "https://example.openai.azure.com/"):
    return env(
        TOUCHLINE_LLM_PROVIDER="azure",
        AZURE_OPENAI_ENDPOINT=endpoint,
        AZURE_OPENAI_API_KEY="secret-key",
        AZURE_OPENAI_DEPLOYMENT="gpt-narrator",
        AZURE_OPENAI_API_VERSION="",
    )


def scripted_client(replies: list[httpx.Response], recorder: list | None = None):
    """A client whose socket answers with each scripted response in turn."""
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        return queue.pop(0)

    plan = HttpPlan(
        provider="test",
        url="https://example.invalid/chat/completions",
        headers={"Content-Type": "application/json"},
        body_model="gpt-narrator",
        model_label="gpt-narrator",
    )
    return ChatJsonClient(plan, transport=httpx.MockTransport(handler))


def reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


# --------------------------------------------------------------------------- #
# The endpoint
# --------------------------------------------------------------------------- #
def test_normalize_endpoint_strips_a_pasted_operation_path() -> None:
    cases = {
        # The exact bug: the Foundry portal copies the full operation URL.
        "https://res.services.ai.azure.com/openai/v1/responses":
            "https://res.services.ai.azure.com",
        "https://res.services.ai.azure.com/openai/v1/":
            "https://res.services.ai.azure.com",
        "https://res.services.ai.azure.com/openai/v1/chat/completions":
            "https://res.services.ai.azure.com",
        # A Foundry *project* endpoint is a different path on the same host.
        "https://res.services.ai.azure.com/api/projects/myproj":
            "https://res.services.ai.azure.com",
        # Already-clean forms are left alone, trailing slash and all.
        "https://res.services.ai.azure.com": "https://res.services.ai.azure.com",
        "https://res.services.ai.azure.com/": "https://res.services.ai.azure.com",
        # The host may legitimately contain ".openai." -- that is a domain, not
        # a path, and must survive untouched.
        "https://football-predictor.openai.azure.com/":
            "https://football-predictor.openai.azure.com",
        "https://football-predictor.openai.azure.com/openai/deployments/d/chat/completions":
            "https://football-predictor.openai.azure.com",
    }
    for raw, want in cases.items():
        assert normalize_endpoint(raw) == want, raw


def test_azure_plan_uses_the_v1_path_and_carries_the_key_in_a_header() -> None:
    with azure_env():
        plan = azure_plan()
    assert plan.url == "https://example.openai.azure.com/openai/v1/chat/completions"
    assert plan.headers["api-key"] == "secret-key"
    assert plan.body_model == "gpt-narrator"
    # Auth travels as a header. A key in a query string ends up in logs.
    assert "?" not in plan.url
    assert "secret-key" not in plan.url


def test_azure_plan_uses_the_deployments_path_when_an_api_version_is_set() -> None:
    with azure_env() as _:
        os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"
        plan = azure_plan()
    assert plan.url == (
        "https://example.openai.azure.com/openai/deployments/gpt-narrator"
        "/chat/completions?api-version=2025-01-01-preview"
    )


def test_azure_plan_survives_the_pasted_full_url() -> None:
    with azure_env("https://res.services.ai.azure.com/openai/v1/responses"):
        plan = azure_plan()
    assert plan.url == "https://res.services.ai.azure.com/openai/v1/chat/completions"


def test_github_plan_uses_the_models_endpoint_and_a_bearer_token() -> None:
    with env(GITHUB_MODELS_TOKEN="pat", GITHUB_MODELS_MODEL=None):
        plan = github_plan()
    assert plan.url == "https://models.github.ai/inference/chat/completions"
    assert plan.headers["Authorization"] == "Bearer pat"
    assert plan.body_model == "openai/gpt-4o-mini"


# --------------------------------------------------------------------------- #
# Choosing a provider without dying over it
# --------------------------------------------------------------------------- #
def test_provider_plan_is_none_for_the_local_desk() -> None:
    with env(TOUCHLINE_LLM_PROVIDER=None):
        assert provider_plan() is None


def test_provider_plan_is_none_for_an_unknown_provider() -> None:
    """A typo in .env should cost the model, not the demo."""
    with env(TOUCHLINE_LLM_PROVIDER="wat"):
        assert provider_plan() is None


def test_provider_plan_is_none_when_azure_config_is_missing() -> None:
    with env(TOUCHLINE_LLM_PROVIDER="azure", **{k: None for k in AZURE_VARS}):
        assert provider_plan() is None


def test_provider_plan_returns_the_configured_plan() -> None:
    with azure_env():
        plan = provider_plan()
    assert plan is not None and plan.provider == "azure"


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #
def test_a_400_about_response_format_is_retried_without_it() -> None:
    """The one retry that earns its keep: an old endpoint rejecting json mode."""
    sent: list = []
    client = scripted_client(
        [httpx.Response(400, text="unknown parameter: response_format"), reply("{}")],
        recorder=sent,
    )
    try:
        assert asyncio.run(client.complete("sys", "user")) == "{}"
    finally:
        asyncio.run(client.aclose())

    assert len(sent) == 2
    assert "response_format" in json.loads(sent[0].content)
    assert "response_format" not in json.loads(sent[1].content)
    # The retry is a retry, not a change of question.
    assert json.loads(sent[0].content)["messages"] == json.loads(sent[1].content)["messages"]


def test_a_400_about_something_else_is_not_retried() -> None:
    """A second charge for a request that was wrong must not happen quietly."""
    sent: list = []
    client = scripted_client(
        [httpx.Response(400, text="deployment not found")], recorder=sent
    )
    try:
        try:
            asyncio.run(client.complete("sys", "user"))
        except httpx.HTTPStatusError:
            pass
        else:
            raise AssertionError("a real 400 must reach the caller")
    finally:
        asyncio.run(client.aclose())
    assert len(sent) == 1


def test_a_500_is_not_retried() -> None:
    sent: list = []
    client = scripted_client([httpx.Response(500, text="boom")], recorder=sent)
    try:
        try:
            asyncio.run(client.complete("sys", "user"))
        except httpx.HTTPStatusError:
            pass
        else:
            raise AssertionError("a 500 must reach the caller")
    finally:
        asyncio.run(client.aclose())
    assert len(sent) == 1


def test_the_request_carries_the_model_only_when_the_plan_has_one() -> None:
    sent: list = []
    named = scripted_client([reply("{}")], recorder=sent)
    try:
        asyncio.run(named.complete("sys", "user"))
    finally:
        asyncio.run(named.aclose())
    assert json.loads(sent[0].content)["model"] == "gpt-narrator"

    sent.clear()
    anonymous = ChatJsonClient(
        HttpPlan(provider="test", url="https://example.invalid/x",
                 headers={}, body_model=None, model_label="none"),
        transport=httpx.MockTransport(lambda r: (sent.append(r), reply("{}"))[1]),
    )
    try:
        asyncio.run(anonymous.complete("sys", "user"))
    finally:
        asyncio.run(anonymous.aclose())
    assert "model" not in json.loads(sent[0].content)


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
    print(f"\n{len(tests) - failures}/{len(tests)} provider tests passed")
    raise SystemExit(1 if failures else 0)
