"""FastMCP server exposing MemAlign tools for LLM judge alignment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from memalign_mcp.alignment import AlignmentEngine
from memalign_mcp.config import MemAlignConfig, load_config
from memalign_mcp.judge_manager import JudgeManager
from memalign_mcp.judgment import JudgmentEngine
from memalign_mcp.llm_client import LLMClient
from memalign_mcp.memory_store import MemoryStore
from memalign_mcp.models import FeedbackInput

logger = logging.getLogger(__name__)

mcp = FastMCP("memalign")

# ---------------------------------------------------------------------------
# Lazy-initialized shared state
# ---------------------------------------------------------------------------

_config: MemAlignConfig | None = None
_judge_manager: JudgeManager | None = None
_llm_client: LLMClient | None = None


def _get_config() -> MemAlignConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_judge_manager() -> JudgeManager:
    global _judge_manager
    if _judge_manager is None:
        _judge_manager = JudgeManager(_get_config())
    return _judge_manager


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(_get_config().anthropic_api_key)
    return _llm_client


def _get_memory_store(judge_name: str) -> MemoryStore:
    return MemoryStore(judge_name, _get_config())


def _get_alignment_engine(judge_name: str) -> AlignmentEngine:
    return AlignmentEngine(
        _get_config(),
        _get_memory_store(judge_name),
        _get_llm_client(),
    )


def _get_judgment_engine(judge_name: str) -> JudgmentEngine:
    return JudgmentEngine(
        _get_config(),
        _get_memory_store(judge_name),
        _get_llm_client(),
    )


# ===========================================================================
# Judge Management Tools
# ===========================================================================


@mcp.tool()
def create_judge(
    name: str,
    criterion: str,
    instructions: str,
    min_score: int = 1,
    max_score: int = 5,
) -> dict[str, Any]:
    """Create a new judge with evaluation criterion and scoring range.

    Args:
        name: Unique judge name (lowercase alphanumeric + hyphens).
        criterion: What this judge evaluates (e.g., "safety", "code quality").
        instructions: Detailed evaluation instructions for the judge.
        min_score: Minimum score (default 1).
        max_score: Maximum score (default 5).
    """
    mgr = _get_judge_manager()
    judge = mgr.create(name, criterion, instructions, min_score, max_score)
    return {
        "status": "created",
        "judge": {
            "name": judge.name,
            "criterion": judge.criterion,
            "score_range": {
                "min": judge.score_range.min_score,
                "max": judge.score_range.max_score,
            },
        },
    }


@mcp.tool()
def list_judges() -> dict[str, Any]:
    """List all configured judges with their stats."""
    mgr = _get_judge_manager()
    judges = mgr.list_judges()
    result = []
    for j in judges:
        info: dict[str, Any] = {
            "name": j.name,
            "criterion": j.criterion,
            "score_range": {
                "min": j.score_range.min_score,
                "max": j.score_range.max_score,
            },
            "created_at": j.created_at.isoformat(),
        }
        try:
            store = _get_memory_store(j.name)
            stats = store.get_stats()
            info["principles"] = stats.total_principles
            info["examples"] = stats.total_examples
        except Exception:
            info["principles"] = "unknown"
            info["examples"] = "unknown"
        result.append(info)
    return {"judges": result, "total": len(result)}


@mcp.tool()
def delete_judge(name: str) -> dict[str, Any]:
    """Delete a judge and all its memory data.

    Args:
        name: Name of the judge to delete.
    """
    mgr = _get_judge_manager()
    try:
        store = _get_memory_store(name)
        store.delete_all()
    except Exception:
        pass
    deleted = mgr.delete(name)
    return {"status": "deleted" if deleted else "not_found", "judge_name": name}


# ===========================================================================
# Alignment Tools
# ===========================================================================


@mcp.tool()
async def align(
    judge_name: str,
    input_text: str,
    expert_feedback: str,
    expert_score: int | None = None,
    judge_output: str | None = None,
    judge_score: int | None = None,
) -> dict[str, Any]:
    """Align a judge with expert feedback on a specific input.

    Stores the example in episodic memory and extracts generalizable principles
    into semantic memory.

    Args:
        judge_name: Name of the judge to align.
        input_text: The input that was evaluated.
        expert_feedback: Natural language feedback from the expert.
        expert_score: Optional numeric score from the expert.
        judge_output: Optional original judge output to compare against.
        judge_score: Optional original judge score to compare against.
    """
    mgr = _get_judge_manager()
    judge_config = mgr.get(judge_name)
    engine = _get_alignment_engine(judge_name)

    feedback = FeedbackInput(
        input_text=input_text,
        expert_feedback=expert_feedback,
        expert_score=expert_score,
        judge_output=judge_output,
        judge_score=judge_score,
    )

    result = await engine.align(judge_config.criterion, feedback)
    return {
        "status": "aligned",
        "example_id": result.example_id,
        "principles_extracted": result.principles_extracted,
        "principles_deduplicated": result.principles_deduplicated,
        "total_principles": result.total_principles,
        "total_examples": result.total_examples,
    }


@mcp.tool()
async def align_batch(judge_name: str, file_path: str) -> dict[str, Any]:
    """Bulk align a judge from a JSONL file of feedback examples.

    Each line should be a JSON object with: input_text, expert_feedback,
    and optionally expert_score, judge_output, judge_score.

    Args:
        judge_name: Name of the judge to align.
        file_path: Path to JSONL file with feedback examples.
    """
    mgr = _get_judge_manager()
    judge_config = mgr.get(judge_name)
    engine = _get_alignment_engine(judge_name)

    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines()):
        try:
            data = json.loads(line)
            feedback = FeedbackInput(**data)
            result = await engine.align(judge_config.criterion, feedback)
            results.append({"line": i + 1, "example_id": result.example_id})
        except Exception as e:
            errors.append({"line": i + 1, "error": str(e)})

    return {
        "status": "completed",
        "processed": len(results),
        "errors": len(errors),
        "error_details": errors[:10],
    }


@mcp.tool()
async def align_interactive(judge_name: str, input_text: str) -> dict[str, Any]:
    """Judge an input first, then provide feedback to align.

    Step 1: Returns the judge's current evaluation.
    Step 2: User provides feedback via the align tool.

    Args:
        judge_name: Name of the judge.
        input_text: The input to evaluate and potentially align on.
    """
    mgr = _get_judge_manager()
    judge_config = mgr.get(judge_name)
    engine = _get_judgment_engine(judge_name)

    result = await engine.judge(judge_config, input_text)
    return {
        "status": "awaiting_feedback",
        "judge_evaluation": {
            "score": result.score,
            "reasoning": result.reasoning,
            "principles_used": result.principles_used,
            "examples_retrieved": result.examples_retrieved,
        },
        "next_step": (
            f"If you disagree with this evaluation, use the 'align' tool "
            f"with judge_name='{judge_name}' to provide your expert feedback."
        ),
    }


# ===========================================================================
# Judgment Tools
# ===========================================================================


@mcp.tool()
async def judge(
    judge_name: str,
    input_text: str,
    context: str | None = None,
) -> dict[str, Any]:
    """Evaluate an input using a memory-augmented judge.

    Constructs working memory from ALL semantic principles and top-k episodic
    examples, then calls the LLM for evaluation.

    Args:
        judge_name: Name of the judge to use.
        input_text: The text to evaluate.
        context: Optional additional context for the evaluation.
    """
    mgr = _get_judge_manager()
    judge_config = mgr.get(judge_name)
    engine = _get_judgment_engine(judge_name)

    result = await engine.judge(judge_config, input_text, context)
    return {
        "score": result.score,
        "reasoning": result.reasoning,
        "judge_name": result.judge_name,
        "principles_used": result.principles_used,
        "examples_retrieved": result.examples_retrieved,
    }


@mcp.tool()
async def judge_batch(
    judge_name: str,
    file_path: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Bulk judge inputs from a JSONL file.

    Each line should be a JSON object with: input_text, and optionally context.
    Results are optionally written to an output file.

    Args:
        judge_name: Name of the judge.
        file_path: Path to JSONL file with inputs.
        output_path: Optional path to write results as JSONL.
    """
    mgr = _get_judge_manager()
    judge_config = mgr.get(judge_name)
    engine = _get_judgment_engine(judge_name)

    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines()):
        try:
            data = json.loads(line)
            result = await engine.judge(
                judge_config,
                data["input_text"],
                data.get("context"),
            )
            results.append({
                "line": i + 1,
                "score": result.score,
                "reasoning": result.reasoning,
            })
        except Exception as e:
            errors.append({"line": i + 1, "error": str(e)})

    if output_path:
        out = Path(output_path)
        out.write_text(
            "\n".join(json.dumps(r) for r in results),
            encoding="utf-8",
        )

    return {
        "status": "completed",
        "processed": len(results),
        "errors": len(errors),
        "results": results[:5],
        "output_file": output_path,
    }


# ===========================================================================
# Memory Management Tools
# ===========================================================================


@mcp.tool()
def list_principles(judge_name: str) -> dict[str, Any]:
    """List all semantic principles for a judge.

    Args:
        judge_name: Name of the judge.
    """
    store = _get_memory_store(judge_name)
    principles = store.get_all_principles()
    return {
        "judge_name": judge_name,
        "total": len(principles),
        "principles": [
            {
                "id": p.id,
                "text": p.text,
                "source_example_ids": p.source_example_ids,
                "created_at": p.created_at.isoformat(),
            }
            for p in principles
        ],
    }


@mcp.tool()
def list_examples(
    judge_name: str,
    query: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """List or search episodic examples for a judge.

    Args:
        judge_name: Name of the judge.
        query: Optional search query to find similar examples.
        limit: Maximum number of examples to return (default 10).
    """
    store = _get_memory_store(judge_name)
    if query:
        examples = store.retrieve_examples(query, k=limit)
    else:
        examples = store.get_all_examples(limit=limit)
    return {
        "judge_name": judge_name,
        "total": len(examples),
        "examples": [
            {
                "id": e.id,
                "input_text": e.input_text[:200],
                "expert_feedback": e.expert_feedback[:200],
                "expert_score": e.expert_score,
                "created_at": e.created_at.isoformat(),
            }
            for e in examples
        ],
    }


@mcp.tool()
def delete_principle(judge_name: str, principle_id: str) -> dict[str, Any]:
    """Delete a specific principle from semantic memory.

    Args:
        judge_name: Name of the judge.
        principle_id: ID of the principle to delete.
    """
    store = _get_memory_store(judge_name)
    deleted = store.delete_principle(principle_id)
    return {
        "status": "deleted" if deleted else "not_found",
        "principle_id": principle_id,
    }


@mcp.tool()
def delete_example(judge_name: str, example_id: str) -> dict[str, Any]:
    """Delete a specific example from episodic memory.

    Args:
        judge_name: Name of the judge.
        example_id: ID of the example to delete.
    """
    store = _get_memory_store(judge_name)
    deleted = store.delete_example(example_id)
    return {
        "status": "deleted" if deleted else "not_found",
        "example_id": example_id,
    }


@mcp.tool()
def update_principle(
    judge_name: str,
    principle_id: str,
    new_text: str,
) -> dict[str, Any]:
    """Update the text of a principle in semantic memory.

    Args:
        judge_name: Name of the judge.
        principle_id: ID of the principle to update.
        new_text: New text for the principle.
    """
    store = _get_memory_store(judge_name)
    updated = store.update_principle(principle_id, new_text)
    if updated:
        return {
            "status": "updated",
            "principle": {"id": updated.id, "text": updated.text},
        }
    return {"status": "not_found", "principle_id": principle_id}


@mcp.tool()
def memory_stats(judge_name: str) -> dict[str, Any]:
    """Get memory statistics for a judge.

    Args:
        judge_name: Name of the judge.
    """
    store = _get_memory_store(judge_name)
    stats = store.get_stats()
    return {
        "judge_name": stats.judge_name,
        "total_principles": stats.total_principles,
        "total_examples": stats.total_examples,
        "oldest_principle": (
            stats.oldest_principle.isoformat() if stats.oldest_principle else None
        ),
        "newest_principle": (
            stats.newest_principle.isoformat() if stats.newest_principle else None
        ),
        "oldest_example": (
            stats.oldest_example.isoformat() if stats.oldest_example else None
        ),
        "newest_example": (
            stats.newest_example.isoformat() if stats.newest_example else None
        ),
    }
