from __future__ import annotations

import pytest
from datetime import datetime, timezone

from memalign_mcp.models import (
    ScoreRange,
    JudgeConfig,
    Principle,
    Example,
    FeedbackInput,
    JudgmentResult,
    AlignmentResult,
    MemoryStats,
)


class TestScoreRange:
    def test_defaults(self):
        sr = ScoreRange()
        assert sr.min_score == 1
        assert sr.max_score == 5

    def test_custom_range(self):
        sr = ScoreRange(min_score=0, max_score=10)
        assert sr.min_score == 0
        assert sr.max_score == 10

    def test_max_must_exceed_min(self):
        with pytest.raises(ValueError, match="must be greater than"):
            ScoreRange(min_score=5, max_score=5)

    def test_max_less_than_min(self):
        with pytest.raises(ValueError, match="must be greater than"):
            ScoreRange(min_score=5, max_score=3)

    def test_frozen(self):
        sr = ScoreRange()
        with pytest.raises(Exception):
            sr.min_score = 0


class TestJudgeConfig:
    def test_valid_name(self):
        jc = JudgeConfig(name="safety", criterion="test", instructions="test")
        assert jc.name == "safety"

    def test_hyphenated_name(self):
        jc = JudgeConfig(name="code-quality", criterion="test", instructions="test")
        assert jc.name == "code-quality"

    def test_single_char_name(self):
        jc = JudgeConfig(name="x", criterion="test", instructions="test")
        assert jc.name == "x"

    def test_invalid_name_uppercase(self):
        with pytest.raises(ValueError, match="must be lowercase"):
            JudgeConfig(name="Safety", criterion="test", instructions="test")

    def test_invalid_name_starts_with_hyphen(self):
        with pytest.raises(ValueError, match="must be lowercase"):
            JudgeConfig(name="-safety", criterion="test", instructions="test")

    def test_invalid_name_ends_with_hyphen(self):
        with pytest.raises(ValueError, match="must be lowercase"):
            JudgeConfig(name="safety-", criterion="test", instructions="test")

    def test_has_created_at(self):
        jc = JudgeConfig(name="safety", criterion="test", instructions="test")
        assert isinstance(jc.created_at, datetime)

    def test_frozen(self):
        jc = JudgeConfig(name="safety", criterion="test", instructions="test")
        with pytest.raises(Exception):
            jc.name = "other"

    def test_default_score_range(self):
        jc = JudgeConfig(name="safety", criterion="test", instructions="test")
        assert jc.score_range.min_score == 1
        assert jc.score_range.max_score == 5


class TestPrinciple:
    def test_auto_id(self):
        p = Principle(text="test principle")
        assert len(p.id) == 12

    def test_auto_timestamp(self):
        p = Principle(text="test principle")
        assert p.created_at.tzinfo is not None

    def test_source_example_ids_default(self):
        p = Principle(text="test")
        assert p.source_example_ids == []

    def test_frozen(self):
        p = Principle(text="test")
        with pytest.raises(Exception):
            p.text = "changed"


class TestExample:
    def test_auto_id(self):
        e = Example(input_text="test", expert_feedback="good")
        assert len(e.id) == 12

    def test_optional_fields(self):
        e = Example(input_text="test", expert_feedback="good")
        assert e.expert_score is None
        assert e.judge_output is None
        assert e.judge_score is None

    def test_all_fields(self):
        e = Example(
            input_text="test",
            expert_feedback="good",
            expert_score=5,
            judge_output="output",
            judge_score=4,
        )
        assert e.expert_score == 5
        assert e.judge_score == 4

    def test_frozen(self):
        e = Example(input_text="test", expert_feedback="good")
        with pytest.raises(Exception):
            e.input_text = "changed"


class TestFeedbackInput:
    def test_minimal(self):
        fi = FeedbackInput(input_text="test", expert_feedback="good")
        assert fi.input_text == "test"
        assert fi.expert_feedback == "good"

    def test_frozen(self):
        fi = FeedbackInput(input_text="test", expert_feedback="good")
        with pytest.raises(Exception):
            fi.input_text = "changed"


class TestJudgmentResult:
    def test_creation(self):
        jr = JudgmentResult(
            score=4,
            reasoning="Well done",
            judge_name="safety",
            principles_used=3,
            examples_retrieved=5,
        )
        assert jr.score == 4
        assert jr.judge_name == "safety"

    def test_frozen(self):
        jr = JudgmentResult(
            score=4, reasoning="r", judge_name="j",
            principles_used=0, examples_retrieved=0,
        )
        with pytest.raises(Exception):
            jr.score = 5


class TestAlignmentResult:
    def test_creation(self):
        ar = AlignmentResult(
            judge_name="safety",
            example_id="abc123",
            principles_extracted=["p1"],
            principles_deduplicated=0,
            total_principles=1,
            total_examples=1,
        )
        assert ar.judge_name == "safety"
        assert len(ar.principles_extracted) == 1


class TestMemoryStats:
    def test_creation(self):
        ms = MemoryStats(
            judge_name="safety",
            total_principles=5,
            total_examples=10,
        )
        assert ms.total_principles == 5
        assert ms.oldest_principle is None
