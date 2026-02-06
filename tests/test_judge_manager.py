from __future__ import annotations

import pytest

from memalign_mcp.judge_manager import JudgeManager


class TestJudgeManager:
    def test_create_judge(self, mock_config):
        mgr = JudgeManager(mock_config)
        judge = mgr.create("safety", "Evaluate safety", "Check for harmful content")
        assert judge.name == "safety"
        assert judge.criterion == "Evaluate safety"
        assert mgr.exists("safety")

    def test_create_duplicate_judge(self, mock_config):
        mgr = JudgeManager(mock_config)
        mgr.create("safety", "test", "test")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create("safety", "test", "test")

    def test_get_judge(self, mock_config):
        mgr = JudgeManager(mock_config)
        created = mgr.create("safety", "criterion", "instructions")
        retrieved = mgr.get("safety")
        assert retrieved.name == created.name
        assert retrieved.criterion == created.criterion

    def test_get_nonexistent(self, mock_config):
        mgr = JudgeManager(mock_config)
        with pytest.raises(ValueError, match="does not exist"):
            mgr.get("nonexistent")

    def test_exists(self, mock_config):
        mgr = JudgeManager(mock_config)
        assert not mgr.exists("safety")
        mgr.create("safety", "test", "test")
        assert mgr.exists("safety")

    def test_list_judges(self, mock_config):
        mgr = JudgeManager(mock_config)
        mgr.create("safety", "test1", "test1")
        mgr.create("quality", "test2", "test2")
        judges = mgr.list_judges()
        assert len(judges) == 2
        names = {j.name for j in judges}
        assert names == {"safety", "quality"}

    def test_list_judges_empty(self, mock_config):
        mgr = JudgeManager(mock_config)
        assert mgr.list_judges() == []

    def test_delete_judge(self, mock_config):
        mgr = JudgeManager(mock_config)
        mgr.create("safety", "test", "test")
        assert mgr.delete("safety")
        assert not mgr.exists("safety")

    def test_delete_nonexistent(self, mock_config):
        mgr = JudgeManager(mock_config)
        assert not mgr.delete("nonexistent")

    def test_invalid_name(self, mock_config):
        mgr = JudgeManager(mock_config)
        with pytest.raises(ValueError):
            mgr.create("INVALID", "test", "test")

    def test_custom_score_range(self, mock_config):
        mgr = JudgeManager(mock_config)
        judge = mgr.create("rating", "test", "test", min_score=0, max_score=10)
        assert judge.score_range.min_score == 0
        assert judge.score_range.max_score == 10
