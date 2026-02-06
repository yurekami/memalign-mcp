"""Configuration management for MemAlign MCP Server."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class MemAlignConfig(BaseModel):
    """Server configuration loaded from environment variables."""

    model_config = {"frozen": True}

    anthropic_api_key: str = Field(description="Anthropic API key")
    project_dir: Path = Field(description="Project directory for .memalign storage")
    retrieval_k: int = Field(
        default=5, description="Number of episodic examples to retrieve"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="SentenceTransformer model name"
    )
    extraction_model: str = Field(
        default="claude-haiku-4-5-20251001", description="Model for principle extraction"
    )
    judgment_model: str = Field(
        default="claude-sonnet-4-5-20250929", description="Model for judgment"
    )
    similarity_threshold: float = Field(
        default=0.90, description="Cosine similarity threshold for deduplication"
    )

    @property
    def memalign_dir(self) -> Path:
        """Return the .memalign directory path within the project directory."""
        return self.project_dir / ".memalign"


def load_config(project_dir: Path | None = None) -> MemAlignConfig:
    """Load configuration from environment variables.

    Args:
        project_dir: Override for project directory. Uses MEMALIGN_PROJECT_DIR env var or cwd.

    Returns:
        Frozen MemAlignConfig instance.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Set it via: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    resolved_dir = project_dir or Path(
        os.environ.get("MEMALIGN_PROJECT_DIR", Path.cwd())
    )

    return MemAlignConfig(
        anthropic_api_key=api_key,
        project_dir=Path(resolved_dir),
        retrieval_k=int(os.environ.get("MEMALIGN_RETRIEVAL_K", "5")),
        embedding_model=os.environ.get(
            "MEMALIGN_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        ),
        extraction_model=os.environ.get(
            "MEMALIGN_EXTRACTION_MODEL", "claude-haiku-4-5-20251001"
        ),
        judgment_model=os.environ.get(
            "MEMALIGN_JUDGMENT_MODEL", "claude-sonnet-4-5-20250929"
        ),
        similarity_threshold=float(
            os.environ.get("MEMALIGN_SIMILARITY_THRESHOLD", "0.90")
        ),
    )
