from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock
from memalign_mcp.judgment import JudgmentEngine
from memalign_mcp.models import JudgeConfig, ScoreRange


class TestJudgmentEngine:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.call_json = AsyncMock(return_value={
            "score": 4,
            "reasoning": "The response is safe and appropriate.",
        })
        return llm

    @pytest.fixture
    def engine(self, mock_config, mock_llm):
        from memalign_mcp.memory_store import MemoryStore
        store = MemoryStore("test-judge", mock_config)
        return JudgmentEngine(mock_config, store, mock_llm)

    @pytest.mark.asyncio
    async def test_judge_returns_result(self, engine, sample_judge_config):
        result = await engine.judge(sample_judge_config, "test input")
        assert result.score == 4
        assert result.reasoning == "The response is safe and appropriate."
        assert result.judge_name == "safety"

    @pytest.mark.asyncio
    async def test_judge_clamps_score(self, engine, mock_llm):
        mock_llm.call_json = AsyncMock(return_value={"score": 10, "reasoning": "test"})
        config = JudgeConfig(name="test", criterion="test", instructions="test", score_range=ScoreRange(min_score=1, max_score=5))
        result = await engine.judge(config, "test")
        assert result.score == 5

    @pytest.mark.asyncio
    async def test_judge_missing_score_raises(self, engine, mock_llm, sample_judge_config):
        mock_llm.call_json = AsyncMock(return_value={"reasoning": "no score"})
        with pytest.raises(ValueError, match="missing 'score'"):
            await engine.judge(sample_judge_config, "test")

    @pytest.mark.asyncio
    async def test_judge_with_context(self, engine, sample_judge_config):
        result = await engine.judge(sample_judge_config, "test input", context="extra")
        assert result.score == 4
