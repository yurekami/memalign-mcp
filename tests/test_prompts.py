from __future__ import annotations
import pytest
from memalign_mcp.prompts import (
    PRINCIPLE_EXTRACTION_SYSTEM,
    DEDUPLICATION_SYSTEM,
    format_principle_extraction_user,
    format_judgment_system,
    format_judgment_user,
    format_deduplication_user,
)


class TestPrincipleExtractionPrompts:
    def test_system_prompt_exists(self):
        assert "principles" in PRINCIPLE_EXTRACTION_SYSTEM.lower()
        assert "JSON" in PRINCIPLE_EXTRACTION_SYSTEM

    def test_user_prompt_with_existing_principles(self):
        result = format_principle_extraction_user(
            criterion="safety",
            existing_principles=["Be safe", "Avoid harm"],
            input_text="test input",
            expert_feedback="test feedback",
        )
        assert "safety" in result
        assert "Be safe" in result
        assert "test input" in result
        assert "test feedback" in result

    def test_user_prompt_without_existing_principles(self):
        result = format_principle_extraction_user(
            criterion="safety",
            existing_principles=[],
            input_text="test input",
            expert_feedback="test feedback",
        )
        assert "None yet" in result

    def test_user_prompt_with_scores(self):
        result = format_principle_extraction_user(
            criterion="safety",
            existing_principles=[],
            input_text="input",
            expert_feedback="feedback",
            expert_score=5,
            judge_output="judge said this",
            judge_score=3,
        )
        assert "5" in result
        assert "3" in result
        assert "disagreement" in result.lower() or "scored" in result.lower()

    def test_no_disagreement_when_scores_match(self):
        result = format_principle_extraction_user(
            criterion="safety",
            existing_principles=[],
            input_text="input",
            expert_feedback="feedback",
            expert_score=5,
            judge_output="output",
            judge_score=5,
        )
        assert "disagreement" not in result.lower()


class TestJudgmentPrompts:
    def test_system_with_principles_and_examples(self):
        result = format_judgment_system(
            criterion="safety",
            instructions="Evaluate safety",
            min_score=1,
            max_score=5,
            principles=["Be kind", "Be safe"],
            examples=[{"input": "test", "feedback": "good", "score": "5"}],
        )
        assert "safety" in result
        assert "Be kind" in result
        assert "1" in result and "5" in result

    def test_system_without_principles_or_examples(self):
        result = format_judgment_system(
            criterion="safety",
            instructions="test",
            min_score=1,
            max_score=5,
            principles=[],
            examples=[],
        )
        assert "safety" in result
        assert "Principles" not in result or "Evaluation Principles" not in result

    def test_user_prompt(self):
        result = format_judgment_user("test input")
        assert "test input" in result

    def test_user_prompt_with_context(self):
        result = format_judgment_user("test input", context="extra context")
        assert "test input" in result
        assert "extra context" in result


class TestDeduplicationPrompts:
    def test_system_prompt(self):
        assert "duplicate" in DEDUPLICATION_SYSTEM.lower()
        assert "unique" in DEDUPLICATION_SYSTEM.lower()

    def test_user_prompt(self):
        result = format_deduplication_user("new principle", ["existing one", "another one"])
        assert "new principle" in result
        assert "existing one" in result
        assert "another one" in result
