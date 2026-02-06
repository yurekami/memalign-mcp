"""Judgment engine for evaluating inputs using memory-augmented prompts."""

from __future__ import annotations

import logging

from memalign_mcp.config import MemAlignConfig
from memalign_mcp.llm_client import LLMClient
from memalign_mcp.memory_store import MemoryStore
from memalign_mcp.models import JudgeConfig, JudgmentResult
from memalign_mcp.prompts import format_judgment_system, format_judgment_user

logger = logging.getLogger(__name__)


class JudgmentEngine:
    """Evaluates input using working memory constructed from semantic + episodic memory.

    The judgment flow:
    1. Load ALL semantic principles
    2. Retrieve top-k episodic examples (vector similarity to input)
    3. Construct working memory prompt
    4. Call LLM (Sonnet) for evaluation
    5. Validate and return result
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

    async def judge(
        self,
        judge_config: JudgeConfig,
        input_text: str,
        context: str | None = None,
    ) -> JudgmentResult:
        """Evaluate an input using the judge's memory-augmented prompt.

        Args:
            judge_config: The judge configuration.
            input_text: The text to evaluate.
            context: Optional additional context.

        Returns:
            JudgmentResult with score and reasoning.

        Raises:
            ValueError: If the LLM response is invalid.
        """
        # Step 1: Load ALL semantic principles
        principles = self._memory.get_all_principles()
        principle_texts = [p.text for p in principles]
        logger.info("Loaded %d principles from semantic memory", len(principle_texts))

        # Step 2: Retrieve top-k episodic examples
        examples = self._memory.retrieve_examples(input_text)
        example_dicts = [
            {
                "input": ex.input_text,
                "feedback": ex.expert_feedback,
                "score": str(ex.expert_score) if ex.expert_score is not None else None,
            }
            for ex in examples
        ]
        logger.info("Retrieved %d examples from episodic memory", len(example_dicts))

        # Step 3: Construct working memory prompt
        system_prompt = format_judgment_system(
            criterion=judge_config.criterion,
            instructions=judge_config.instructions,
            min_score=judge_config.score_range.min_score,
            max_score=judge_config.score_range.max_score,
            principles=principle_texts,
            examples=example_dicts,
        )
        user_prompt = format_judgment_user(input_text, context)

        # Step 4: Call LLM
        response = await self._llm.call_json(
            system=system_prompt,
            user=user_prompt,
            model=self._config.judgment_model,
        )

        # Step 5: Validate and return
        raw_score = response.get("score")
        reasoning = response.get("reasoning", "")

        if raw_score is None:
            raise ValueError("LLM response missing 'score' field")

        score = int(raw_score)
        min_s = judge_config.score_range.min_score
        max_s = judge_config.score_range.max_score

        if score < min_s or score > max_s:
            logger.warning(
                "Score %d outside range [%d, %d], clamping",
                score, min_s, max_s,
            )
            score = max(min_s, min(max_s, score))

        return JudgmentResult(
            score=score,
            reasoning=reasoning,
            judge_name=judge_config.name,
            principles_used=len(principle_texts),
            examples_retrieved=len(example_dicts),
        )
