"""The only module in Touchline that opens a network connection.

Two agents speak to a language model -- the desk (`narrator.py`) and its critic
(`verifier.py`) -- and neither of them should care whether they are talking to
Azure OpenAI, GitHub Models, or something that does not exist yet. Everything
provider-specific lives here instead:

    HttpPlan          a resolved endpoint, auth header and model label
    provider_plan()   TOUCHLINE_LLM_PROVIDER -> HttpPlan, or None for offline mode
    ChatJsonClient    one POST, one JSON object back, one retry without JSON mode

Keeping this apart from the prompts is what makes the provider a configuration
choice rather than a code change. It also means the rest of the project can be
tested with no network at all: a test hands `ChatJsonClient` an
`httpx.MockTransport` and the real request path runs against a scripted reply.

The environment, in full:

    TOUCHLINE_LLM_PROVIDER     local | azure | github        (default: local)
    AZURE_OPENAI_ENDPOINT      base origin only, e.g. https://res.services.ai.azure.com
    AZURE_OPENAI_API_KEY       sent as the `api-key` header
    AZURE_OPENAI_DEPLOYMENT    the deployment name, not the model name
    AZURE_OPENAI_API_VERSION   blank on the /openai/v1 surface
    GITHUB_MODELS_TOKEN        a PAT with models:read
    GITHUB_MODELS_MODEL        default openai/gpt-4o-mini
    TOUCHLINE_LLM_TIMEOUT_S    default 30
    TOUCHLINE_LLM_MAX_TOKENS   default 700
    TOUCHLINE_LLM_TEMPERATURE  default 0.3
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

# Load .env for local development. `override=False` (the default) means a real
# environment variable always wins, so a container or CI setting is never
# silently replaced by a stray file on disk. This is the only place it happens:
# every module that reads an LLM setting imports this one.
load_dotenv()


def strip_fences(raw: str) -> str:
    """Unwrap a JSON reply that a model dressed in markdown code fences.

    Asking for JSON does not stop a chat model from answering with ```json ...```,
    and the difference between "parse the reply" and "parse the reply politely"
    is the difference between working and falling back to templates. Kept here
    rather than in either prompt module because both critics of the output --
    the desk's parser and the verifier's -- need the same polite version.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# The plan: everything provider-specific, resolved once
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HttpPlan:
    """Everything provider-specific, resolved once at startup."""

    provider: str
    url: str
    headers: dict[str, str]
    body_model: str | None     # value for the request's "model" field, if any
    model_label: str           # what we record in the narrative


def normalize_endpoint(endpoint: str) -> str:
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


def azure_plan() -> HttpPlan:
    endpoint = normalize_endpoint(os.environ["AZURE_OPENAI_ENDPOINT"])
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


def github_plan() -> HttpPlan:
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


_PLANS = {"azure": azure_plan, "github": github_plan}


def provider_plan() -> HttpPlan | None:
    """Resolve the environment into a plan, or None if we are staying offline.

    An unset, unknown or half-configured provider is not an error the demo should
    die of -- it is a reason to use the deterministic path. That is why this
    returns None rather than raising: the caller's fallback *is* the answer.
    """
    provider = os.environ.get("TOUCHLINE_LLM_PROVIDER", "local").strip().lower()
    if provider == "local":
        return None
    build = _PLANS.get(provider)
    if build is None:
        print(f"[llm] unknown provider {provider!r}; staying offline")
        return None
    try:
        return build()
    except (KeyError, ValueError) as exc:
        print(f"[llm] {provider} is missing config ({exc}); staying offline")
        return None


# --------------------------------------------------------------------------- #
# The client: one POST, one JSON object back
# --------------------------------------------------------------------------- #
class ChatJsonClient:
    """An OpenAI-compatible chat-completions call that must answer in JSON.

    The same class serves every provider; the URL, the auth header and whether the
    model travels in the body all live in the `HttpPlan`. The only clever part is
    the retry: `response_format` is understood by current chat models but not by
    every endpoint, and a 400 about that one field should cost a retry, not the
    whole reveal.
    """

    def __init__(
        self,
        plan: HttpPlan,
        *,
        timeout_s: float = 30.0,
        temperature: float = 0.3,
        max_tokens: int = 700,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.plan = plan
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        # `transport` exists so tests can drive the real request path against a
        # scripted response. Production passes nothing and gets a real socket.
        #
        # The client is built here and lives for the process, which is what makes
        # connection reuse work across a whole match. The consequence is that this
        # object is bound to the event loop that first uses it: one client per
        # loop, and nothing here may be shared across two `asyncio.run` calls.
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, system: str, user: str, json_mode: bool) -> dict:
        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.plan.body_model:
            payload["model"] = self.plan.body_model
        return payload

    async def complete(self, system: str, user: str) -> str:
        """Return the raw assistant text, retrying once without JSON mode."""
        response = await self._client.post(
            self.plan.url, headers=self.plan.headers, json=self._payload(system, user, True)
        )
        if response.status_code == 400 and "response_format" in response.text:
            response = await self._client.post(
                self.plan.url, headers=self.plan.headers,
                json=self._payload(system, user, False),
            )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"] or ""
