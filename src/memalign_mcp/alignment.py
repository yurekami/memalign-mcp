from __future__ import annotations

import logging
from typing import Any

from memalign_mcp.config import MemAlignConfig
from memalign_mcp.llm_client import LLMClient
from memalign_mcp.memory_store import MemoryStore
from memalign_mcp.models import AlignmentResult, Example, FeedbackInput, Principle
from memalign_mcp.prompts import (
    DEDUPLICATION_SYSTEM,
    PRINCIPLE_EXTRACTION_SYSTEM,
    format_deduplication_user,
    format_principle_extraction_user,
)

logger = logging.getLogger(__name__)


class AlignmentEngine:
    """Processes expert feedback to build semantic and episodic memory.

    The alignment flow:
    1. Store the example in episodic memory
    2. Extract generalizable principles via LLM (Haiku)
    3. Deduplicate new principles against existing ones
    4. Store unique principles in semantic memory
    """

    def __init__(
        self,
        config: MemAlignConfig,
        memory_store: MemoryStore,
        llm_client: LLMClient,
    ) -> None:
        self._config = config
        self._memory = memory_store
        self._llm = llm_client

    async def align(self, judge_criterion: str, feedback: FeedbackInput) -> AlignmentResult:
        """Process a single feedback example through the alignment pipeline.

        Args:
            judge_criterion: The judge's evaluation criterion.
            feedback: The expert feedback to process.

        Returns:
            AlignmentResult with extraction and memory stats.
        """
        # Step 1: Create and store the example in episodic memory
        example = Example(
            input_text=feedback.input_text,
            expert_feedback=feedback.expert_feedback,
            expert_score=feedback.expert_score,
            judge_output=feedback.judge_output,
            judge_score=feedback.judge_score,
        )
        self._memory.add_example(example)
        logger.info("Stored example %s in episodic memory", example.id)

        # Step 2: Extract principles via LLM
        existing_principles = self._memory.get_all_principles()
        existing_texts = [p.text for p in existing_principles]

        extracted = await self._extract_principles(
            criterion=judge_criterion,
            existing_principles=existing_texts,
            feedback=feedback,
            example_id=example.id,
        )
        logger.info("Extracted %d candidate principles", len(extracted))

        # Step 3: Deduplicate and store unique principles
        new_principles = []
        deduplicated_count = 0

        for principle in extracted:
            is_duplicate = await self._is_duplicate(principle, existing_texts)
            if is_duplicate:
                deduplicated_count += 1
                logger.debug("Filtered duplicate principle: %s", principle.text[:60])
            else:
                self._memory.add_principle(principle)
                new_principles.append(principle.text)
                existing_texts.append(principle.text)
                logger.info("Stored new principle: %s", principle.text[:60])

        # Step 4: Get final stats
        stats = self._memory.get_stats()

        return AlignmentResult(
            judge_name=stats.judge_name,
            example_id=example.id,
            principles_extracted=new_principles,
            principles_deduplicated=deduplicated_count,
            total_principles=stats.total_principles,
            total_examples=stats.total_examples,
        )

    async def _extract_principles(
        self,
        criterion: str,
        existing_principles: list[str],
        feedback: FeedbackInput,
        example_id: str,
    ) -> list[Principle]:
        """Extract generalizable principles from feedback using LLM.

        Args:
            criterion: Judge's evaluation criterion.
            existing_principles: Already stored principles.
            feedback: The feedback to extract from.
            example_id: ID of the stored example.

        Returns:
            List of candidate Principle objects.
        """
        user_prompt = format_principle_extraction_user(
            criterion=criterion,
            existing_principles=existing_principles,
            input_text=feedback.input_text,
            expert_feedback=feedback.expert_feedback,
            expert_score=feedback.expert_score,
            judge_output=feedback.judge_output,
            judge_score=feedback.judge_score,
        )

        try:
            response = await self._llm.call_json(
                system=PRINCIPLE_EXTRACTION_SYSTEM,
                user=user_prompt,
                model=self._config.extraction_model,
            )
        except ValueError as e:
            logger.warning("Failed to parse principle extraction response: %s", e)
            return []

        raw_principles = response.get("principles", [])
        principles = []
        for raw in raw_principles:
            text = raw.get("text", "").strip()
            if text:
                principles.append(Principle(
                    text=text,
                    source_example_ids=[example_id],
                ))

        return principles

    async def _is_duplicate(
        self,
        principle: Principle,
        existing_texts: list[str],
    ) -> bool:
        """Check if a principle is a duplicate using embedding similarity + LLM.

        Two-stage deduplication:
        1. Embedding similarity check (fast, cheap)
        2. LLM confirmation for borderline cases (slower, more accurate)

        Args:
            principle: The new principle to check.
            existing_texts: List of existing principle texts.

        Returns:
            True if the principle is a duplicate.
        """
        if not existing_texts:
            return False

        # Stage 1: Embedding similarity check
        similar = self._memory.find_similar_principles(
            principle.text,
            threshold=self._config.similarity_threshold,
        )

        if not similar:
            return False

        # Stage 2: LLM confirmation for high-similarity matches
        # Get the texts of similar principles for LLM check
        similar_texts = [p.text for p, _ in similar]

        try:
            user_prompt = format_deduplication_user(principle.text, similar_texts)
            response = await self._llm.call(
                system=DEDUPLICATION_SYSTEM,
                user=user_prompt,
                model=self._config.extraction_model,
            )
            return response.strip().lower() == "duplicate"
        except Exception as e:
            logger.warning("Deduplication LLM check failed: %s. Treating as unique.", e)
            return False
