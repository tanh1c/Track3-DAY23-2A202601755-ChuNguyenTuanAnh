"""Structured-output schemas used by LLM-backed nodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassificationDecision(BaseModel):
    """Validated intent classification returned by the routing LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    reason: str = Field(min_length=1, max_length=240)


class EvaluationDecision(BaseModel):
    """Validated one-shot LLM-as-judge verdict for a tool result."""

    verdict: Literal["success", "needs_retry"]
    reason: str = Field(min_length=1, max_length=240)
