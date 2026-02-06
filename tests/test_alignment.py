from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock
from memalign_mcp.alignment import AlignmentEngine
from memalign_mcp.models import FeedbackInput


class TestAlignmentEngine:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.call_json = AsyncMock(return_value={
            "principles": [{"text": "Always be safe"}],
            "reasoning": "Safety is important",
        })
        llm.call = AsyncMock(return_value="unique")
        return llm

    @pytest.fixture
    def engine(self, mock_config, mock_llm):
        from memalign_mcp.memory_store import MemoryStore
        store = MemoryStore("test-judge", mock_config)
        return AlignmentEngine(mock_config, store, mock_llm)

    @pytest.mark.asyncio
    async def test_align_stores_example(self, engine, sample_feedback):
        result = await engine.align("safety", sample_feedback)
        assert result.example_id is not None
        assert result.total_examples == 1

    @pytest.mark.asyncio
    async def test_align_extracts_principles(self, engine, sample_feedback):
        result = await engine.align("safety", sample_feedback)
        assert len(result.principles_extracted) == 1
        assert "Always be safe" in result.principles_extracted
        assert result.total_principles == 1

    @pytest.mark.asyncio
    async def test_align_handles_extraction_failure(self, engine, mock_llm, sample_feedback):
        mock_llm.call_json = AsyncMock(side_effect=ValueError("parse error"))
        result = await engine.align("safety", sample_feedback)
        assert result.principles_extracted == []
        assert result.total_examples == 1
