from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ScoreRange(BaseModel):
    """Defines valid score bounds for a judge."""
    model_config = {"frozen": True}

    min_score: int = Field(default=1, description="Minimum score (inclusive)")
    max_score: int = Field(default=5, description="Maximum score (inclusive)")

    @field_validator("max_score")
    @classmethod
    def max_greater_than_min(cls, v: int, info: Any) -> int:
        min_val = info.data.get("min_score", 1)
        if v <= min_val:
            raise ValueError(f"max_score ({v}) must be greater than min_score ({min_val})")
        return v


class JudgeConfig(BaseModel):
    """Configuration for a judge."""
    model_config = {"frozen": True}

    name: str = Field(description="Unique judge name (alphanumeric + hyphens)")
    criterion: str = Field(description="What this judge evaluates (e.g., 'safety', 'politeness')")
    instructions: str = Field(description="Detailed evaluation instructions")
    score_range: ScoreRange = Field(default_factory=ScoreRange)
    created_at: datetime = Field(default_factory=_now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", v):
            raise ValueError(
                f"Judge name '{v}' must be lowercase alphanumeric with hyphens, "
                "cannot start or end with a hyphen"
            )
        return v


class Principle(BaseModel):
    """A semantic memory entry - a generalizable evaluation principle."""
    model_config = {"frozen": True}

    id: str = Field(default_factory=_generate_id)
    text: str = Field(description="The principle text")
    source_example_ids: list[str] = Field(default_factory=list, description="Example IDs this was derived from")
    created_at: datetime = Field(default_factory=_now)


class Example(BaseModel):
    """An episodic memory entry - a specific evaluation example."""
    model_config = {"frozen": True}

    id: str = Field(default_factory=_generate_id)
    input_text: str = Field(description="The input that was evaluated")
    expert_feedback: str = Field(description="Natural language expert feedback")
    expert_score: int | None = Field(default=None, description="Expert's score")
    judge_output: str | None = Field(default=None, description="Judge's original output")
    judge_score: int | None = Field(default=None, description="Judge's original score")
    created_at: datetime = Field(default_factory=_now)


class FeedbackInput(BaseModel):
    """Input for the align tool."""
    model_config = {"frozen": True}

    input_text: str = Field(description="The input that was evaluated")
    expert_feedback: str = Field(description="Natural language expert feedback")
    expert_score: int | None = Field(default=None, description="Expert's score")
    judge_output: str | None = Field(default=None, description="Judge's original output")
    judge_score: int | None = Field(default=None, description="Judge's original score")


class JudgmentResult(BaseModel):
    """Output from the judge tool."""
    model_config = {"frozen": True}

    score: int = Field(description="Numeric score within judge's range")
    reasoning: str = Field(description="Explanation of the score")
    judge_name: str = Field(description="Name of the judge used")
    principles_used: int = Field(description="Number of principles in working memory")
    examples_retrieved: int = Field(description="Number of episodic examples retrieved")


class AlignmentResult(BaseModel):
    """Output from the align tool."""
    model_config = {"frozen": True}

    judge_name: str
    example_id: str = Field(description="ID of the stored example")
    principles_extracted: list[str] = Field(description="New principles extracted")
    principles_deduplicated: int = Field(default=0, description="Number of duplicate principles filtered")
    total_principles: int = Field(description="Total principles in semantic memory")
    total_examples: int = Field(description="Total examples in episodic memory")


class MemoryStats(BaseModel):
    """Statistics about a judge's memory."""
    model_config = {"frozen": True}

    judge_name: str
    total_principles: int
    total_examples: int
    oldest_principle: datetime | None = None
    newest_principle: datetime | None = None
    oldest_example: datetime | None = None
    newest_example: datetime | None = None
