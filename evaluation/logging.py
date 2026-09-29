from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        successful = sum(bool(item["success"]) for item in items)
        recovery_candidates = [
            item for item in items if item.get("first_sql_correct") is False
        ]
        return {
            "tasks": count,
            "successes": successful,
            "execution_accuracy": successful / count if count else 0.0,
            "valid_action_rate": (
                sum(bool(item["valid_action"]) for item in items) / count if count else 0.0
            ),
            "valid_sql_rate": (
                sum(bool(item["valid_sql"]) for item in items) / count if count else 0.0
            ),
            "executable_sql_rate": (
                sum(bool(item["executable_sql"]) for item in items) / count if count else 0.0
            ),
            "average_turns": sum(item["turns"] for item in items) / count if count else 0.0,
            "average_tool_calls": (
                sum(len(item["tool_calls"]) for item in items) / count if count else 0.0
            ),
            "average_tokens": sum(item["token_count"] for item in items) / count if count else 0.0,
            "average_latency_seconds": (
                sum(item["latency_seconds"] for item in items) / count if count else 0.0
            ),
            "recovery_rate": (
                sum(bool(item["recovered_after_error"]) for item in recovery_candidates)
                / len(recovery_candidates)
                if recovery_candidates
                else 0.0
            ),
            "failure_categories": dict(
                sorted(
                    Counter(
                        item["failure_category"]
                        for item in items
                        if item.get("failure_category")
                    ).items()
                )
            ),
        }

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
        by_difficulty[row["difficulty"]].append(row)
    return {
        "overall": aggregate(rows),
        "by_mode": {key: aggregate(value) for key, value in sorted(by_mode.items())},
        "by_difficulty": {
            key: aggregate(value) for key, value in sorted(by_difficulty.items())
        },
    }
