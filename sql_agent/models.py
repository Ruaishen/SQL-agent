from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Difficulty = Literal["easy", "medium", "hard", "extra"]
Split = Literal["train", "internal_validation", "internal_holdout", "external_dev"]


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    source: str
    db_id: str
    db_path: str
    question: str
    reference_sql: str
    difficulty: Difficulty
    split: Split

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskRecord:
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_db_path(self, spider_root: Path) -> Path:
        path = (spider_root / self.db_path).resolve()
        root = spider_root.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Task database path escapes spider_root")
        return path


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class VerifierResult:
    reward: float
    correct: bool
    agent_sql_valid: bool
    agent_sql_executable: bool
    column_match: bool
    row_count_match: bool
    value_match: bool
    order_sensitive: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
