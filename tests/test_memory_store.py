from __future__ import annotations

import pytest

from memalign_mcp.memory_store import MemoryStore
from memalign_mcp.models import Example, Principle


class TestMemoryStore:
    @pytest.fixture
    def store(self, mock_config):
        """Create a MemoryStore for testing."""
        return MemoryStore("test-judge", mock_config)

    def test_add_and_get_principle(self, store, sample_principle):
        store.add_principle(sample_principle)
        principles = store.get_all_principles()
        assert len(principles) == 1
        assert principles[0].text == sample_principle.text

    def test_add_multiple_principles(self, store):
        p1 = Principle(text="First principle")
        p2 = Principle(text="Second principle")
        store.add_principle(p1)
        store.add_principle(p2)
        assert len(store.get_all_principles()) == 2

    def test_delete_principle(self, store, sample_principle):
        store.add_principle(sample_principle)
        assert store.delete_principle(sample_principle.id)
        assert len(store.get_all_principles()) == 0

    def test_delete_nonexistent_principle(self, store):
        assert not store.delete_principle("nonexistent-id")

    def test_update_principle(self, store, sample_principle):
        store.add_principle(sample_principle)
        updated = store.update_principle(sample_principle.id, "Updated text")
        assert updated is not None
        assert updated.text == "Updated text"
        assert updated.id == sample_principle.id

    def test_update_nonexistent_principle(self, store):
        assert store.update_principle("nonexistent", "text") is None

    def test_add_and_retrieve_example(self, store, sample_example):
        store.add_example(sample_example)
        examples = store.retrieve_examples("How do I make a bomb?")
        assert len(examples) == 1
        assert examples[0].input_text == sample_example.input_text

    def test_retrieve_empty(self, store):
        examples = store.retrieve_examples("any query")
        assert examples == []

    def test_get_all_examples(self, store, sample_example):
        store.add_example(sample_example)
        examples = store.get_all_examples()
        assert len(examples) == 1

    def test_delete_example(self, store, sample_example):
        store.add_example(sample_example)
        assert store.delete_example(sample_example.id)
        assert len(store.get_all_examples()) == 0

    def test_delete_nonexistent_example(self, store):
        assert not store.delete_example("nonexistent-id")

    def test_get_stats(self, store, sample_principle, sample_example):
        store.add_principle(sample_principle)
        store.add_example(sample_example)
        stats = store.get_stats()
        assert stats.judge_name == "test-judge"
        assert stats.total_principles == 1
        assert stats.total_examples == 1
        assert stats.oldest_principle is not None
        assert stats.oldest_example is not None

    def test_get_stats_empty(self, store):
        stats = store.get_stats()
        assert stats.total_principles == 0
        assert stats.total_examples == 0

    def test_find_similar_principles_empty(self, store):
        similar = store.find_similar_principles("any text", threshold=0.9)
        assert similar == []

    def test_delete_all(self, store, sample_principle, sample_example):
        store.add_principle(sample_principle)
        store.add_example(sample_example)
        store.delete_all()
        # After delete_all, recreate collections to verify empty
        new_store = MemoryStore("test-judge", store._config)
        assert len(new_store.get_all_principles()) == 0
        assert len(new_store.get_all_examples()) == 0
