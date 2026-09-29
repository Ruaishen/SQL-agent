from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_diverse_tasks(
    tasks: list[dict[str, Any]], difficulty: str, count: int, seed: int
) -> list[dict[str, Any]]:
    candidates = [task for task in tasks if task.get("difficulty") == difficulty]
    random.Random(seed).shuffle(candidates)
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    seen_databases: set[str] = set()
    for task in candidates:
        if task["db_id"] not in seen_databases:
            selected.append(task)
            seen_databases.add(task["db_id"])
        else:
            deferred.append(task)
        if len(selected) == count:
            break
    if len(selected) < count:
        selected.extend(deferred[: count - len(selected)])
    if len(selected) != count:
        raise ValueError(
            f"not enough {difficulty} tasks: requested {count}, found {len(candidates)}"
        )
    return selected
