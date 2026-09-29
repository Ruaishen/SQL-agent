from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SftCollectionConfig:
    student_model: Path
    teacher_model: Path
    train_data: Path
    output_dir: Path
    seed: int = 42
    target_easy: int = 256
    target_medium: int = 256
    target_hard: int = 0
    target_extra: int = 0
    bootstrap_from: Path | None = None
    max_attempts_per_task: int = 3
    max_assistant_turns: int = 10
    max_sequence_tokens: int = 16384
    max_action_tokens: int = 384
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    teacher_dtype: str = "bfloat16"

    @classmethod
    def load(cls, path: Path) -> SftCollectionConfig:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("SFT collection config must be a YAML mapping")
        for key in ("student_model", "teacher_model", "train_data", "output_dir"):
            raw[key] = Path(raw[key])
        if raw.get("bootstrap_from") is not None:
            raw["bootstrap_from"] = Path(raw["bootstrap_from"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        for path in (self.student_model, self.teacher_model, self.train_data):
            if not path.exists():
                raise FileNotFoundError(path)
        if any(target < 0 for target in self.target_counts.values()):
            raise ValueError("SFT difficulty targets must be non-negative")
        if self.target_total < 1:
            raise ValueError("at least one SFT difficulty target must be positive")
        if self.bootstrap_from is not None and not self.bootstrap_from.exists():
            raise FileNotFoundError(self.bootstrap_from)
        if self.max_attempts_per_task < 1:
            raise ValueError("max_attempts_per_task must be positive")
        if self.max_assistant_turns < 1:
            raise ValueError("max_assistant_turns must be positive")
        if not 1 <= self.max_action_tokens < self.max_sequence_tokens:
            raise ValueError("invalid SFT sequence/action token limits")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("invalid Teacher sampling parameters")
        if self.teacher_dtype != "bfloat16":
            raise ValueError("Teacher collection requires bfloat16")

    @property
    def target_total(self) -> int:
        return sum(self.target_counts.values())

    @property
    def target_counts(self) -> dict[str, int]:
        return {
            "easy": self.target_easy,
            "medium": self.target_medium,
            "hard": self.target_hard,
            "extra": self.target_extra,
        }

    @property
    def difficulties(self) -> tuple[str, ...]:
        return tuple(
            difficulty for difficulty, target in self.target_counts.items() if target > 0
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result
