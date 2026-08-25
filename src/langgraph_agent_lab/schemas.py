"""Structured-output schemas used by LLM-backed nodes."""

from typing import Literal

from pydantic import BaseModel, Field


ClassificationRoute = Literal["simple", "tool", "missing_info", "risky", "error"]


class ClassificationDecision(BaseModel):
    """Validated intent classification returned by the routing LLM."""

    route: ClassificationRoute
    reason: str = Field(min_length=1, max_length=240)


class EvaluationDecision(BaseModel):
    """Validated one-shot LLM-as-judge verdict for a tool result."""

    verdict: Literal["success", "needs_retry"]
    reason: str = Field(min_length=1, max_length=240)
