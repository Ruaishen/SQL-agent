from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sft.collect import _read_tasks, diverse_task_order, should_supervise, validate_shards
from sql_agent.action_parser import parse_action
from sql_agent.models import TaskRecord


def _task(task_id: str, db_id: str, split: str = "train") -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        source="spider",
        db_id=db_id,
        db_path=f"database/{db_id}/{db_id}.sqlite",
        question="question",
        reference_sql="SELECT 1",
        difficulty="easy",
        split=split,  # type: ignore[arg-type]
    )


def test_diverse_order_places_one_task_per_database_first() -> None:
    tasks = [_task("a1", "a"), _task("a2", "a"), _task("b1", "b")]
    ordered = diverse_task_order(tasks, "easy", 42)
    assert len(ordered) == 3
    assert len({ordered[0].db_id, ordered[1].db_id}) == 2
    assert [task.task_id for task in ordered] == [
        task.task_id for task in diverse_task_order(tasks, "easy", 42)
    ]


def test_read_tasks_rejects_validation_leakage(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(_task("bad", "db", "internal_validation").to_dict()) + "\n")
    with pytest.raises(ValueError, match="non-train"):
        _read_tasks(path)


def test_supervision_requires_valid_non_error_action_and_correct_final_sql() -> None:
    inspect = parse_action('{"tool":"inspect_table","arguments":{"table_name":"t"}}')
    final_sql = parse_action('{"tool":"execute_sql","arguments":{"sql":"SELECT 1"}}')
    assert should_supervise(inspect, {"status": "success"})
    assert not should_supervise(inspect, {"status": "error"})
    assert not should_supervise(None, {"status": "success"})
    assert should_supervise(
        final_sql, {"status": "finalized", "verification": {"correct": True}}
    )
    assert not should_supervise(
        final_sql, {"status": "finalized", "verification": {"correct": False}}
    )
    assert not should_supervise(
        inspect, {"status": "finalized", "verification": {"correct": True}}
    )


def test_validate_shards_reloads_and_checks_masks(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_path = shard_dir / "easy.pt"
    torch.save(
        {
            "format_version": 1,
            "task_id": "easy-1",
            "turns": [
                {
                    "input_ids": torch.tensor([1, 2, 3, 4]),
                    "prompt_length": 2,
                    "action_mask": torch.tensor([False, True, True]),
                    "supervised": True,
                    "observation": {
                        "status": "finalized",
                        "verification": {"correct": True},
                    },
                }
            ],
        },
        shard_path,
    )
    records = [
        {
            "success": True,
            "task_id": "easy-1",
            "difficulty": "easy",
            "shard": str(shard_path),
        }
    ]
    assert validate_shards(
        tmp_path,
        records,
        target_counts={"easy": 1, "medium": 0, "hard": 0, "extra": 0},
    ) == [shard_path]
