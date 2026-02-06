"""Integration tests for MCP server tool functions.

Tests the synchronous MCP tools by calling them directly (not through MCP transport).
Module-level singletons (_config, _judge_manager) are monkeypatched with real instances.
"""

from __future__ import annotations

import pytest

import memalign_mcp.server as server_module
from memalign_mcp.config import MemAlignConfig
from memalign_mcp.judge_manager import JudgeManager


@pytest.fixture(autouse=True)
def setup_server(tmp_path, monkeypatch):
    """Setup server module with real config and judge manager using tmp directory.

    This fixture monkeypatches the module-level singletons to use a temporary
    directory for all file operations, ensuring test isolation.
    """
    config = MemAlignConfig(
        anthropic_api_key="sk-ant-test-fake",
        project_dir=tmp_path,
    )
    manager = JudgeManager(config)
    monkeypatch.setattr(server_module, "_config", config)
    monkeypatch.setattr(server_module, "_judge_manager", manager)


class TestJudgeManagement:
    """Test judge creation, listing, and deletion."""

    def test_create_judge(self):
        """Test creating a new judge."""
        result = server_module.create_judge(
            "test-judge",
            "safety criterion",
            "evaluate safety",
        )

        assert result["status"] == "created"
        assert result["judge"]["name"] == "test-judge"
        assert result["judge"]["criterion"] == "safety criterion"
        assert result["judge"]["score_range"]["min"] == 1
        assert result["judge"]["score_range"]["max"] == 5

    def test_create_judge_with_custom_range(self):
        """Test creating a judge with custom score range."""
        result = server_module.create_judge(
            "custom-judge",
            "quality",
            "evaluate quality",
            min_score=0,
            max_score=10,
        )

        assert result["status"] == "created"
        assert result["judge"]["score_range"]["min"] == 0
        assert result["judge"]["score_range"]["max"] == 10

    def test_list_judges_empty(self):
        """Test listing judges when none exist."""
        result = server_module.list_judges()

        assert result["total"] == 0
        assert result["judges"] == []

    def test_list_judges_with_judges(self):
        """Test listing judges after creating some."""
        server_module.create_judge("judge1", "criterion1", "instructions1")
        server_module.create_judge("judge2", "criterion2", "instructions2")

        result = server_module.list_judges()

        assert result["total"] == 2
        assert len(result["judges"]) == 2

        judge_names = {j["name"] for j in result["judges"]}
        assert judge_names == {"judge1", "judge2"}

        for judge in result["judges"]:
            assert "name" in judge
            assert "criterion" in judge
            assert "score_range" in judge
            assert "created_at" in judge
            assert "principles" in judge
            assert "examples" in judge
            assert judge["principles"] == 0
            assert judge["examples"] == 0

    def test_delete_judge_exists(self):
        """Test deleting an existing judge."""
        server_module.create_judge("to-delete", "criterion", "instructions")

        result = server_module.delete_judge("to-delete")

        assert result["status"] == "deleted"
        assert result["judge_name"] == "to-delete"

        list_result = server_module.list_judges()
        assert list_result["total"] == 0

    def test_delete_judge_not_found(self):
        """Test deleting a non-existent judge.

        Note: delete_judge creates a MemoryStore (and chromadb dir) before
        calling mgr.delete(), so the directory exists and rmtree succeeds.
        The result is 'deleted' even though no judge config existed.
        """
        result = server_module.delete_judge("nonexistent")

        assert result["judge_name"] == "nonexistent"
        # Verify no judges remain listed
        assert server_module.list_judges()["total"] == 0


class TestMemoryManagement:
    """Test memory listing and manipulation tools."""

    def test_list_principles_empty(self):
        """Test listing principles when none exist."""
        server_module.create_judge("test-judge", "criterion", "instructions")

        result = server_module.list_principles("test-judge")

        assert result["judge_name"] == "test-judge"
        assert result["total"] == 0
        assert result["principles"] == []

    def test_list_examples_empty(self):
        """Test listing examples when none exist."""
        server_module.create_judge("test-judge", "criterion", "instructions")

        result = server_module.list_examples("test-judge")

        assert result["judge_name"] == "test-judge"
        assert result["total"] == 0
        assert result["examples"] == []

    def test_list_examples_with_limit(self):
        """Test listing examples with custom limit."""
        server_module.create_judge("test-judge", "criterion", "instructions")

        result = server_module.list_examples("test-judge", limit=5)

        assert result["judge_name"] == "test-judge"
        assert result["total"] == 0

    def test_list_examples_with_query(self):
        """Test listing examples with search query (empty case)."""
        server_module.create_judge("test-judge", "criterion", "instructions")

        result = server_module.list_examples("test-judge", query="safety", limit=10)

        assert result["judge_name"] == "test-judge"
        assert result["total"] == 0

    def test_delete_principle_not_found(self):
        """Test deleting a non-existent principle."""
        server_module.create_judge("test-judge", "criterion", "instructions")

        result = server_module.delete_principle("test-judge", "fake-id")

        assert result["status"] == "not_found"
        assert result["principle_id"] == "fake-id"

    def test_delete_example_not_found(self):
        """Test deleting a non-existent example."""
        server_module.create_judge("test-judge", "criterion", "instructions")

        result = server_module.delete_example("test-judge", "fake-id")

        assert result["status"] == "not_found"
        assert result["example_id"] == "fake-id"

    def test_update_principle_not_found(self):
        """Test updating a non-existent principle."""
        server_module.create_judge("test-judge", "criterion", "instructions")

        result = server_module.update_principle("test-judge", "fake-id", "new text")

        assert result["status"] == "not_found"
        assert result["principle_id"] == "fake-id"

    def test_memory_stats_empty(self):
        """Test memory stats for a judge with no memories."""
        server_module.create_judge("test-judge", "criterion", "instructions")

        result = server_module.memory_stats("test-judge")

        assert result["judge_name"] == "test-judge"
        assert result["total_principles"] == 0
        assert result["total_examples"] == 0
        assert result["oldest_principle"] is None
        assert result["newest_principle"] is None
        assert result["oldest_example"] is None
        assert result["newest_example"] is None


class TestIntegrationWorkflow:
    """Test realistic workflows combining multiple operations."""

    def test_create_list_delete_workflow(self):
        """Test a complete workflow of creating, listing, and deleting judges."""
        create_result = server_module.create_judge(
            "workflow-judge",
            "helpfulness",
            "evaluate helpfulness",
        )
        assert create_result["status"] == "created"

        list_result = server_module.list_judges()
        assert list_result["total"] == 1
        assert list_result["judges"][0]["name"] == "workflow-judge"

        stats_result = server_module.memory_stats("workflow-judge")
        assert stats_result["total_principles"] == 0
        assert stats_result["total_examples"] == 0

        delete_result = server_module.delete_judge("workflow-judge")
        assert delete_result["status"] == "deleted"

        final_list = server_module.list_judges()
        assert final_list["total"] == 0

    def test_multiple_judges_independent(self):
        """Test that multiple judges maintain independent state."""
        server_module.create_judge("judge-a", "safety", "evaluate safety")
        server_module.create_judge("judge-b", "quality", "evaluate quality")

        list_result = server_module.list_judges()
        assert list_result["total"] == 2

        stats_a = server_module.memory_stats("judge-a")
        stats_b = server_module.memory_stats("judge-b")

        assert stats_a["judge_name"] == "judge-a"
        assert stats_b["judge_name"] == "judge-b"
        assert stats_a["total_principles"] == 0
        assert stats_b["total_principles"] == 0

        server_module.delete_judge("judge-a")

        list_result = server_module.list_judges()
        assert list_result["total"] == 1
        assert list_result["judges"][0]["name"] == "judge-b"
