from __future__ import annotations

# === Principle Extraction Prompts ===

PRINCIPLE_EXTRACTION_SYSTEM = """You are an expert at analyzing evaluation feedback and extracting generalizable principles.

Your task: Given expert feedback on a specific evaluation, extract GENERAL principles that could apply to future evaluations of the same type.

Rules:
- Extract only GENERALIZABLE principles, not case-specific observations
- Each principle should be a clear, actionable guideline
- Avoid redundancy with existing principles (provided below)
- If the feedback doesn't contain any new generalizable insights, return an empty list
- Output valid JSON only

Output format:
{
  "principles": [
    {"text": "The principle text here"}
  ],
  "reasoning": "Brief explanation of why these principles were extracted"
}"""


def format_principle_extraction_user(
    criterion: str,
    existing_principles: list[str],
    input_text: str,
    expert_feedback: str,
    expert_score: int | None = None,
    judge_output: str | None = None,
    judge_score: int | None = None,
) -> str:
    """Format the user prompt for principle extraction.

    Args:
        criterion: What the judge evaluates.
        existing_principles: Currently stored principles to avoid redundancy.
        input_text: The input that was evaluated.
        expert_feedback: The expert's natural language feedback.
        expert_score: Optional expert score.
        judge_output: Optional judge's original output.
        judge_score: Optional judge's original score.

    Returns:
        Formatted user prompt string.
    """
    parts = [f"## Evaluation Criterion\n{criterion}\n"]

    if existing_principles:
        principles_text = "\n".join(f"- {p}" for p in existing_principles)
        parts.append(f"## Existing Principles (avoid redundancy)\n{principles_text}\n")
    else:
        parts.append("## Existing Principles\nNone yet.\n")

    parts.append(f"## Input Being Evaluated\n{input_text}\n")
    parts.append(f"## Expert Feedback\n{expert_feedback}\n")

    if expert_score is not None:
        parts.append(f"## Expert Score\n{expert_score}\n")

    if judge_output is not None:
        parts.append(f"## Judge's Original Output\n{judge_output}\n")
        if judge_score is not None:
            parts.append(f"## Judge's Original Score\n{judge_score}\n")
        disagreement = ""
        if expert_score is not None and judge_score is not None and expert_score != judge_score:
            disagreement = (
                f"\nNote: The expert scored this {expert_score} but the judge scored it {judge_score}. "
                "Pay special attention to what the expert's feedback reveals about this disagreement."
            )
        if disagreement:
            parts.append(disagreement)

    parts.append(
        "\nExtract generalizable evaluation principles from this feedback. "
        "Return JSON with the format specified in your instructions."
    )

    return "\n".join(parts)


# === Judgment Prompts ===

JUDGMENT_SYSTEM_TEMPLATE = """You are an expert evaluator. Your task is to evaluate the given input based on a specific criterion.

## Criterion
{criterion}

## Evaluation Instructions
{instructions}

## Score Range
{min_score} (lowest) to {max_score} (highest)

{principles_section}
{examples_section}

## Output Format
Respond with valid JSON only:
{{
  "score": <integer between {min_score} and {max_score}>,
  "reasoning": "<detailed explanation of your score>"
}}

Important:
- Your score MUST be an integer between {min_score} and {max_score}
- Your reasoning should reference specific aspects of the input
- Consider the principles and examples above when making your judgment
- Be consistent with the evaluation patterns shown in the examples"""


def format_judgment_system(
    criterion: str,
    instructions: str,
    min_score: int,
    max_score: int,
    principles: list[str],
    examples: list[dict[str, str]],
) -> str:
    """Format the judgment system prompt.

    Args:
        criterion: What the judge evaluates.
        instructions: Detailed evaluation instructions.
        min_score: Minimum score.
        max_score: Maximum score.
        principles: List of principle texts from semantic memory.
        examples: List of example dicts with 'input', 'feedback', 'score' keys.

    Returns:
        Formatted system prompt string.
    """
    if principles:
        principles_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(principles))
        principles_section = f"## Evaluation Principles\nApply these principles in your evaluation:\n{principles_text}"
    else:
        principles_section = ""

    if examples:
        example_parts = []
        for i, ex in enumerate(examples, 1):
            ex_text = f"  ### Example {i}\n  **Input:** {ex.get('input', 'N/A')}\n"
            if ex.get("feedback"):
                ex_text += f"  **Expert Feedback:** {ex['feedback']}\n"
            if ex.get("score"):
                ex_text += f"  **Expert Score:** {ex['score']}\n"
            example_parts.append(ex_text)
        examples_section = "## Reference Examples\nUse these as calibration:\n" + "\n".join(example_parts)
    else:
        examples_section = ""

    return JUDGMENT_SYSTEM_TEMPLATE.format(
        criterion=criterion,
        instructions=instructions,
        min_score=min_score,
        max_score=max_score,
        principles_section=principles_section,
        examples_section=examples_section,
    )


JUDGMENT_USER_TEMPLATE = """## Input to Evaluate

{input_text}

{context_section}

Evaluate this input and respond with JSON containing your score and reasoning."""


def format_judgment_user(input_text: str, context: str | None = None) -> str:
    """Format the judgment user prompt.

    Args:
        input_text: The input to evaluate.
        context: Optional additional context.

    Returns:
        Formatted user prompt string.
    """
    context_section = f"## Additional Context\n{context}" if context else ""
    return JUDGMENT_USER_TEMPLATE.format(
        input_text=input_text,
        context_section=context_section,
    )


# === Deduplication Prompts ===

DEDUPLICATION_SYSTEM = """You are a deduplication specialist. Your task is to determine if a new principle is semantically equivalent to any existing principles.

Two principles are duplicates if they convey the same evaluation guideline, even if worded differently.

Respond with ONLY one word: "duplicate" or "unique"."""


def format_deduplication_user(
    new_principle: str,
    existing_principles: list[str],
) -> str:
    """Format the deduplication user prompt.

    Args:
        new_principle: The new principle to check.
        existing_principles: Existing principles to compare against.

    Returns:
        Formatted user prompt string.
    """
    existing_text = "\n".join(f"- {p}" for p in existing_principles)
    return (
        f"## New Principle\n{new_principle}\n\n"
        f"## Existing Principles\n{existing_text}\n\n"
        "Is the new principle a duplicate of any existing principle? "
        "Answer 'duplicate' or 'unique'."
    )
