"""ChromaDB-backed dual memory store for judge alignment."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from memalign_mcp.config import MemAlignConfig
from memalign_mcp.embeddings import LazyEmbeddingFunction
from memalign_mcp.models import Example, MemoryStats, Principle

logger = logging.getLogger(__name__)


class MemoryStore:
    """ChromaDB-backed dual memory store for a single judge.

    Manages two collections:
    - Semantic memory: stores generalizable principles (ALL loaded at judgment time)
    - Episodic memory: stores specific examples (top-k retrieved at judgment time)
    """

    def __init__(self, judge_name: str, config: MemAlignConfig) -> None:
        self._judge_name = judge_name
        self._config = config
        self._embedding_fn = LazyEmbeddingFunction(config.embedding_model)

        # Persistent ChromaDB client stored per-judge
        db_path = config.memalign_dir / judge_name / "chromadb"
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))

        # Get or create collections
        self._semantic = self._client.get_or_create_collection(
            name=f"{judge_name}_semantic",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._episodic = self._client.get_or_create_collection(
            name=f"{judge_name}_episodic",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # === Semantic Memory (Principles) ===

    def add_principle(self, principle: Principle) -> None:
        """Add a principle to semantic memory.

        Args:
            principle: The principle to store.
        """
        self._semantic.upsert(
            ids=[principle.id],
            documents=[principle.text],
            metadatas=[{
                "source_example_ids": ",".join(principle.source_example_ids),
                "created_at": principle.created_at.isoformat(),
            }],
        )
        logger.info("Added principle %s: %s", principle.id, principle.text[:80])

    def get_all_principles(self) -> list[Principle]:
        """Get ALL principles from semantic memory.

        Returns:
            List of all stored principles.
        """
        results = self._semantic.get(include=["documents", "metadatas"])
        principles = []
        for id_, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            source_ids = [s for s in meta.get("source_example_ids", "").split(",") if s]
            principles.append(Principle(
                id=id_,
                text=doc,
                source_example_ids=source_ids,
                created_at=datetime.fromisoformat(meta["created_at"]),
            ))
        return principles

    def find_similar_principles(self, text: str, threshold: float) -> list[tuple[Principle, float]]:
        """Find principles similar to the given text.

        Used for deduplication - checks if a new principle is too similar to existing ones.

        Args:
            text: The principle text to check against.
            threshold: Cosine similarity threshold (e.g. 0.90).

        Returns:
            List of (principle, similarity_score) tuples above threshold.
        """
        count = self._semantic.count()
        if count == 0:
            return []

        results = self._semantic.query(
            query_texts=[text],
            n_results=min(count, 5),
            include=["documents", "metadatas", "distances"],
        )

        similar = []
        for id_, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns distance; for cosine space, similarity = 1 - distance
            similarity = 1.0 - distance
            if similarity >= threshold:
                source_ids = [s for s in meta.get("source_example_ids", "").split(",") if s]
                principle = Principle(
                    id=id_,
                    text=doc,
                    source_example_ids=source_ids,
                    created_at=datetime.fromisoformat(meta["created_at"]),
                )
                similar.append((principle, similarity))

        return similar

    def delete_principle(self, principle_id: str) -> bool:
        """Delete a principle by ID.

        Returns:
            True if deleted, False if not found.
        """
        try:
            existing = self._semantic.get(ids=[principle_id])
            if not existing["ids"]:
                return False
            self._semantic.delete(ids=[principle_id])
            return True
        except Exception:
            return False

    def update_principle(self, principle_id: str, new_text: str) -> Principle | None:
        """Update a principle's text.

        Args:
            principle_id: ID of the principle to update.
            new_text: New text for the principle.

        Returns:
            Updated principle, or None if not found.
        """
        existing = self._semantic.get(ids=[principle_id], include=["metadatas"])
        if not existing["ids"]:
            return None

        meta = existing["metadatas"][0]
        self._semantic.update(
            ids=[principle_id],
            documents=[new_text],
        )

        source_ids = [s for s in meta.get("source_example_ids", "").split(",") if s]
        return Principle(
            id=principle_id,
            text=new_text,
            source_example_ids=source_ids,
            created_at=datetime.fromisoformat(meta["created_at"]),
        )

    # === Episodic Memory (Examples) ===

    def add_example(self, example: Example) -> None:
        """Add an example to episodic memory.

        The document is the concatenation of input_text and expert_feedback,
        which is what gets embedded for retrieval. Full details stored in metadata.
        """
        document = f"{example.input_text}\n{example.expert_feedback}"
        metadata: dict[str, Any] = {
            "input_text": example.input_text,
            "expert_feedback": example.expert_feedback,
            "created_at": example.created_at.isoformat(),
        }
        if example.expert_score is not None:
            metadata["expert_score"] = example.expert_score
        if example.judge_output is not None:
            metadata["judge_output"] = example.judge_output
        if example.judge_score is not None:
            metadata["judge_score"] = example.judge_score

        self._episodic.upsert(
            ids=[example.id],
            documents=[document],
            metadatas=[metadata],
        )
        logger.info("Added example %s", example.id)

    def retrieve_examples(self, query: str, k: int | None = None) -> list[Example]:
        """Retrieve top-k most similar examples to the query.

        Args:
            query: The input text to find similar examples for.
            k: Number of examples to retrieve. Defaults to config.retrieval_k.

        Returns:
            List of examples sorted by similarity (most similar first).
        """
        k = k or self._config.retrieval_k
        count = self._episodic.count()
        if count == 0:
            return []

        results = self._episodic.query(
            query_texts=[query],
            n_results=min(count, k),
            include=["metadatas"],
        )

        examples = []
        for id_, meta in zip(results["ids"][0], results["metadatas"][0]):
            examples.append(Example(
                id=id_,
                input_text=meta["input_text"],
                expert_feedback=meta["expert_feedback"],
                expert_score=meta.get("expert_score"),
                judge_output=meta.get("judge_output"),
                judge_score=meta.get("judge_score"),
                created_at=datetime.fromisoformat(meta["created_at"]),
            ))

        return examples

    def get_all_examples(self, limit: int = 100) -> list[Example]:
        """Get examples from episodic memory (non-query, for listing).

        Args:
            limit: Maximum number of examples to return.

        Returns:
            List of examples.
        """
        count = self._episodic.count()
        if count == 0:
            return []

        results = self._episodic.get(
            include=["metadatas"],
            limit=min(count, limit),
        )

        examples = []
        for id_, meta in zip(results["ids"], results["metadatas"]):
            examples.append(Example(
                id=id_,
                input_text=meta["input_text"],
                expert_feedback=meta["expert_feedback"],
                expert_score=meta.get("expert_score"),
                judge_output=meta.get("judge_output"),
                judge_score=meta.get("judge_score"),
                created_at=datetime.fromisoformat(meta["created_at"]),
            ))

        return examples

    def delete_example(self, example_id: str) -> bool:
        """Delete an example by ID.

        Returns:
            True if deleted, False if not found.
        """
        try:
            existing = self._episodic.get(ids=[example_id])
            if not existing["ids"]:
                return False
            self._episodic.delete(ids=[example_id])
            return True
        except Exception:
            return False

    # === Stats ===

    def get_stats(self) -> MemoryStats:
        """Get memory statistics for this judge."""
        principles = self.get_all_principles()
        examples = self.get_all_examples(limit=10000)

        p_dates = [p.created_at for p in principles] if principles else []
        e_dates = [e.created_at for e in examples] if examples else []

        return MemoryStats(
            judge_name=self._judge_name,
            total_principles=len(principles),
            total_examples=len(examples),
            oldest_principle=min(p_dates) if p_dates else None,
            newest_principle=max(p_dates) if p_dates else None,
            oldest_example=min(e_dates) if e_dates else None,
            newest_example=max(e_dates) if e_dates else None,
        )

    def delete_all(self) -> None:
        """Delete all data for this judge (both collections)."""
        self._client.delete_collection(f"{self._judge_name}_semantic")
        self._client.delete_collection(f"{self._judge_name}_episodic")
        logger.info("Deleted all memory for judge %s", self._judge_name)
