from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from memalign_mcp.config import MemAlignConfig
from memalign_mcp.models import JudgeConfig, ScoreRange

logger = logging.getLogger(__name__)


class JudgeManager:
    """Manages judge lifecycle: create, list, get, delete.

    Stores judge configurations as JSON files at:
    .memalign/<judge_name>/config.json
    """

    def __init__(self, config: MemAlignConfig) -> None:
        self._config = config
        self._base_dir = config.memalign_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _judge_dir(self, name: str) -> Path:
        return self._base_dir / name

    def _config_path(self, name: str) -> Path:
        return self._judge_dir(name) / "config.json"

    def create(
        self,
        name: str,
        criterion: str,
        instructions: str,
        min_score: int = 1,
        max_score: int = 5,
    ) -> JudgeConfig:
        """Create a new judge.

        Args:
            name: Unique judge name (lowercase alphanumeric + hyphens).
            criterion: What this judge evaluates.
            instructions: Detailed evaluation instructions.
            min_score: Minimum score (inclusive).
            max_score: Maximum score (inclusive).

        Returns:
            Created JudgeConfig.

        Raises:
            ValueError: If judge already exists or name is invalid.
        """
        if self.exists(name):
            raise ValueError(f"Judge '{name}' already exists")

        judge_config = JudgeConfig(
            name=name,
            criterion=criterion,
            instructions=instructions,
            score_range=ScoreRange(min_score=min_score, max_score=max_score),
        )

        # Create directory and save config
        judge_dir = self._judge_dir(name)
        judge_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._config_path(name)
        config_path.write_text(
            judge_config.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.info("Created judge '%s'", name)
        return judge_config

    def get(self, name: str) -> JudgeConfig:
        """Get a judge configuration by name.

        Args:
            name: Judge name.

        Returns:
            JudgeConfig for the named judge.

        Raises:
            ValueError: If judge does not exist.
        """
        config_path = self._config_path(name)
        if not config_path.exists():
            raise ValueError(f"Judge '{name}' does not exist")

        data = json.loads(config_path.read_text(encoding="utf-8"))
        return JudgeConfig(**data)

    def exists(self, name: str) -> bool:
        """Check if a judge exists."""
        return self._config_path(name).exists()

    def list_judges(self) -> list[JudgeConfig]:
        """List all judges.

        Returns:
            List of JudgeConfig for all judges.
        """
        judges = []
        if not self._base_dir.exists():
            return judges

        for judge_dir in sorted(self._base_dir.iterdir()):
            if judge_dir.is_dir():
                config_path = judge_dir / "config.json"
                if config_path.exists():
                    try:
                        data = json.loads(config_path.read_text(encoding="utf-8"))
                        judges.append(JudgeConfig(**data))
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning("Skipping invalid judge config at %s: %s", config_path, e)

        return judges

    def delete(self, name: str) -> bool:
        """Delete a judge and all its data.

        Args:
            name: Judge name to delete.

        Returns:
            True if deleted, False if not found.
        """
        judge_dir = self._judge_dir(name)
        if not judge_dir.exists():
            return False

        shutil.rmtree(judge_dir, ignore_errors=True)
        logger.info("Deleted judge '%s' and all its data", name)
        return True
