from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from sql_agent.config import EnvConfig


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(child) for child in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "0x" + bytes(value).hex()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


class TokenCounter:
    def __init__(self, tokenizer_path: Path):
        tokenizer_file = tokenizer_path / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"Tokenizer file does not exist: {tokenizer_file}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))

    def count_text(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_json(self, value: Any) -> int:
        return self.count_text(canonical_json(value))


class ObservationLimiter:
    _TAIL_KEYS = ("rows", "values", "foreign_keys", "columns", "tables", "tools")

    def __init__(self, config: EnvConfig, token_counter: TokenCounter):
        self.config = config
        self.token_counter = token_counter

    def limit(self, observation: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(make_json_safe(observation))
        reasons = list(value.pop("truncation_reasons", []))
        text_was_truncated = self._truncate_text(value)
        if text_was_truncated:
            reasons.append("cell_length_limit")
        value["truncated"] = bool(reasons)
        value["truncation_reasons"] = self._dedupe(reasons)
        while self.token_counter.count_json(value) > self.config.max_observation_tokens:
            if not self._drop_tail(value):
                value = self._minimal_fallback(value)
                break
            if "token_limit" not in value["truncation_reasons"]:
                value["truncation_reasons"].append("token_limit")
            value["truncated"] = True
        while self.token_counter.count_json(value) > self.config.max_observation_tokens:
            message = value.get("message")
            if not isinstance(message, str) or not message:
                raise RuntimeError("Minimal observation exceeds token limit")
            value["message"] = message[:-1]
        if isinstance(value.get("values"), list) and "returned_value_count" in value:
            value["returned_value_count"] = len(value["values"])
        if isinstance(value.get("rows"), list) and "returned_row_count" in value:
            value["returned_row_count"] = len(value["rows"])
        if isinstance(value.get("tables"), list) and "returned_table_count" in value:
            value["returned_table_count"] = len(value["tables"])
        return value

    def _truncate_text(self, value: Any) -> bool:
        changed = False
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "truncation_reasons":
                    continue
                if isinstance(child, str) and len(child) > self.config.max_cell_chars:
                    value[key] = child[: self.config.max_cell_chars]
                    changed = True
                else:
                    changed = self._truncate_text(child) or changed
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, str) and len(child) > self.config.max_cell_chars:
                    value[index] = child[: self.config.max_cell_chars]
                    changed = True
                else:
                    changed = self._truncate_text(child) or changed
        return changed

    def _drop_tail(self, value: dict[str, Any]) -> bool:
        for key in self._TAIL_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, list) and candidate:
                candidate.pop()
                return True
        return False

    @staticmethod
    def _minimal_fallback(value: dict[str, Any]) -> dict[str, Any]:
        fallback = {
            "status": value.get("status", "error"),
            "error_type": value.get("error_type", "observation_too_large"),
            "message": value.get("message", "Observation exceeded the token limit"),
            "truncated": True,
            "truncation_reasons": ["token_limit"],
        }

        for key in ("max_turns", "turns_remaining", "termination_reason", "reward"):
            if key in value:
                fallback[key] = value[key]
        return fallback

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))
