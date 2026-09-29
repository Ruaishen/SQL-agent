from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class EnvConfig:
    spider_root: Path = Path("/root/autodl-tmp/spider/spider_data")
    processed_data_root: Path = Path("/root/autodl-tmp/sql-agent-rl/data")
    tokenizer_path: Path = Path("/root/autodl-tmp/Qwen3-1.7B")
    split_seed: int = 42
    max_turns: int = 10
    max_context_tokens: int = 16384
    reserved_action_tokens: int = 512
    max_observation_tokens: int = 1024
    max_tables: int = 128
    max_columns: int = 128
    max_foreign_keys: int = 128
    inspect_values_max: int = 50
    execute_rows_max: int = 50
    max_cell_chars: int = 512
    max_sql_chars: int = 32768
    query_timeout_seconds: float = 2.0
    verifier_timeout_seconds: float = 5.0
    verifier_max_rows: int = 100_000
    verifier_max_bytes: int = 64 * 1024 * 1024

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnvConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        path_fields = {"spider_root", "processed_data_root", "tokenizer_path"}
        known = {field.name for field in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        values: dict[str, Any] = dict(raw)
        for name in path_fields & values.keys():
            values[name] = Path(values[name]).expanduser().resolve()
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "max_turns": self.max_turns,
            "max_context_tokens": self.max_context_tokens,
            "reserved_action_tokens": self.reserved_action_tokens,
            "max_observation_tokens": self.max_observation_tokens,
            "max_tables": self.max_tables,
            "max_columns": self.max_columns,
            "max_foreign_keys": self.max_foreign_keys,
            "inspect_values_max": self.inspect_values_max,
            "execute_rows_max": self.execute_rows_max,
            "max_cell_chars": self.max_cell_chars,
            "max_sql_chars": self.max_sql_chars,
            "query_timeout_seconds": self.query_timeout_seconds,
            "verifier_timeout_seconds": self.verifier_timeout_seconds,
            "verifier_max_rows": self.verifier_max_rows,
            "verifier_max_bytes": self.verifier_max_bytes,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {invalid}")
        if self.reserved_action_tokens >= self.max_context_tokens:
            raise ValueError("reserved_action_tokens must be smaller than max_context_tokens")
