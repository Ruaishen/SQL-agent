from __future__ import annotations

import json
from pathlib import Path

from sql_agent.models import TaskRecord


def load_tasks(path: str | Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(TaskRecord.from_dict(json.loads(line)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid task record at {path}:{line_number}") from exc
    return records


def find_task(path: str | Path, task_id: str | None) -> TaskRecord:
    tasks = load_tasks(path)
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    if task_id is None:
        return tasks[0]
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Task {task_id!r} not found in {path}")
