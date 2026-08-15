"""Pydantic schemas for batch theory scoring responses."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TheoryScore(BaseModel):
    theory_id: str = Field(description="Anonymous id, e.g. T014")
    clarity_and_coherence: int
    causal_reasoning: int
    theoretical_depth: int
    creativity: int
    persuasiveness: int
    brief_reasoning: str

    @field_validator(
        "clarity_and_coherence",
        "causal_reasoning",
        "theoretical_depth",
        "creativity",
        "persuasiveness",
    )
    @classmethod
    def score_in_range(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1 or v > 10:
            raise ValueError(f"score must be int in 1–10, got {v!r}")
        return v

    @field_validator("brief_reasoning")
    @classmethod
    def reasoning_nonempty(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError("brief_reasoning must be non-empty")
        return text

    @field_validator("theory_id")
    @classmethod
    def theory_id_nonempty(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError("theory_id must be non-empty")
        return text


class BatchScores(BaseModel):
    scores: list[TheoryScore]
