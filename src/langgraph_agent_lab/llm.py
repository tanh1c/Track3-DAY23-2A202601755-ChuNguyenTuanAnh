"""Provider-agnostic LLM factory used by the graph nodes."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Loading once at module import makes a local .env useful without repeatedly
# re-reading it in every node invocation. Existing process environment wins.
load_dotenv()


def configured_provider() -> str | None:
    """Return the provider selected by the documented environment priority."""
    if os.getenv("GEMINI_API_KEY"):
        return "google"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def _resolve_model(explicit: str | None, default: str) -> str:
    """Resolve an explicit model, environment override, or provider default."""
    return explicit or os.getenv("LLM_MODEL") or default


def _client_options(
    temperature: float,
    timeout: float | None,
    max_retries: int | None,
) -> dict[str, Any]:
    """Build provider client options without overriding provider defaults implicitly."""
    options: dict[str, Any] = {"temperature": temperature}
    if timeout is not None:
        options["timeout"] = timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    return options


def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Create an LLM client from environment configuration.

    Provider priority is intentionally stable: Gemini, OpenAI, then Anthropic.
    `model` overrides `LLM_MODEL`, which overrides the provider default. Optional
    timeout/retry budgets are forwarded explicitly when a caller needs a bounded request.
    """
    provider = configured_provider()
    options = _client_options(temperature, timeout, max_retries)

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=_resolve_model(model, "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            **options,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=_resolve_model(model, "gpt-4o-mini"),
            **options,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=_resolve_model(model, "claude-sonnet-4-20250514"),
            **options,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or "
        "ANTHROPIC_API_KEY in .env or the process environment."
    )
