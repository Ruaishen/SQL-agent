"""Merge original, successful retry, and deduplicated gold-guided trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sql_planner.collect import _atomic_json, _record_path, ensure_run_manifest, read_spider_train_tasks
from sql_planner.repair_tagged import validate_complete_source

MERGE_VERSION = "tagged_original_retry_gold_priority_v1"


def select_record(
    original: dict[str, Any], retries: list[dict[str, Any]],
    gold_repair: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Prefer naturally correct rollouts over gold-guided repair."""
    if original.get("correct"):
        return "original", original
    for retry in retries:
        if retry.get("correct"):
            return "retry", retry
    if gold_repair and gold_repair.get("sft_eligible") and gold_repair.get("correct"):
        return "gold_repair", gold_repair
    return None, None


def _load_record(path: Path, task_id: str, *, required: bool) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise ValueError(f"Missing required trajectory: {path}")
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("task_id", record.get("source_task_id")) != task_id:
        raise ValueError(f"Trajectory task ID mismatch: {path}")
    return record


def _unified_record(
    record: dict[str, Any], *, task_id: str, origin: str, source_path: Path,
) -> dict[str, Any]:
    result = dict(record)
    result["task_id"] = task_id
    result["merged_origin"] = origin
    result["merged_source_path"] = str(source_path.resolve())
    if origin == "gold_repair":
        result["messages"] = record["student_messages"]
        result["trainable_turn_numbers"] = record["trainable_turn_numbers"]
    else:
        result["trainable_turn_numbers"] = list(range(1, len(record["turns"]) + 1))
    if not result.get("correct") or not result.get("messages"):
        raise ValueError(f"Selected trajectory is not a usable correct example: {task_id}")
    return result


def merge(
    original_dir: Path, retry_dir: Path, gold_dir: Path, output_dir: Path,
    spider_root: Path,
) -> dict[str, int]:
    original_manifest, source_files = validate_complete_source(original_dir)
    retry_manifest = json.loads((retry_dir / "run_manifest.json").read_text(encoding="utf-8"))
    gold_manifest = json.loads((gold_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if retry_manifest.get("source_manifest_sha256") != hashlib.sha256((original_dir / "run_manifest.json").read_bytes()).hexdigest():
        raise ValueError("Retry run does not match original collection")
    if gold_manifest.get("source_manifest_sha256") != retry_manifest["source_manifest_sha256"]:
        raise ValueError("Gold repair does not match original collection")
    tasks = read_spider_train_tasks(spider_root, limit=0, seed=42)
    if len(tasks) != 7000 or hashlib.sha256((spider_root / "train_spider.json").read_bytes()).hexdigest() != original_manifest["source_sha256"]:
        raise ValueError("Spider training source does not match original collection")
    failed_count = 0
    for task in tasks:
        original = _load_record(_record_path(original_dir, task, 0), task.task_id, required=True)
        failed_count += int(not original["correct"])
    if (len(source_files) != 7000 or retry_manifest.get("source_failed_task_count") != failed_count
            or retry_manifest.get("selected_task_count") != failed_count
            or retry_manifest.get("max_attempts") != 2):
        raise ValueError("Retry run must cover all original failed tasks with at most two attempts")
    if gold_manifest.get("source_count") != 7000:
        raise ValueError("Gold repair is not from the complete original run")
    expected_gold_count = gold_manifest.get("selected_incorrect_count")
    if not isinstance(expected_gold_count, int) or len(list((gold_dir / "trajectories").glob("*.json"))) != expected_gold_count:
        raise ValueError("Gold repair batch is incomplete")
    ensure_run_manifest(output_dir, {
        "schema_version": 1, "merge_version": MERGE_VERSION,
        "original_dir": str(original_dir.resolve()),
        "retry_dir": str(retry_dir.resolve()),
        "gold_dir": str(gold_dir.resolve()),
        "original_manifest_sha256": retry_manifest["source_manifest_sha256"],
        "retry_manifest_sha256": hashlib.sha256((retry_dir / "run_manifest.json").read_bytes()).hexdigest(),
        "gold_manifest_sha256": hashlib.sha256((gold_dir / "run_manifest.json").read_bytes()).hexdigest(),
        "priority": ["original", "retry", "gold_repair"],
    })
    counts = {
        "original_correct": 0, "retry_rescued": 0, "gold_repair_total_candidates": 0,
        "gold_repair_duplicates_removed": 0, "gold_repair_unique": 0,
        "merged_unique_tasks": 0, "unsolved_tasks": 0,
    }
    index_path = output_dir / "selection_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as index_file:
        for task in tasks:
            original_path = _record_path(original_dir, task, 0)
            original = _load_record(original_path, task.task_id, required=True)
            retries: list[dict[str, Any]] = []
            retry_paths: list[Path] = []
            if not original["correct"]:
                first_path = _record_path(retry_dir, task, 1)
                first = _load_record(first_path, task.task_id, required=True)
                retries.append(first)
                retry_paths.append(first_path)
                if not first["correct"]:
                    second_path = _record_path(retry_dir, task, 2)
                    second = _load_record(second_path, task.task_id, required=True)
                    retries.append(second)
                    retry_paths.append(second_path)
            gold_path = _record_path(gold_dir, task, 0)
            gold = _load_record(gold_path, task.task_id, required=False)
            if gold and gold.get("sft_eligible") and gold.get("correct"):
                counts["gold_repair_total_candidates"] += 1
            origin, selected = select_record(original, retries, gold)
            if original["correct"]:
                counts["original_correct"] += 1
            if origin == "retry":
                counts["retry_rescued"] += 1
            if gold and gold.get("sft_eligible") and gold.get("correct"):
                if origin != "gold_repair":
                    counts["gold_repair_duplicates_removed"] += 1
                else:
                    counts["gold_repair_unique"] += 1
                    _atomic_json(
                        _record_path(output_dir / "gold_repair_unique", task, 0),
                        _unified_record(gold, task_id=task.task_id, origin="gold_repair", source_path=gold_path),
                    )
            selected_path = (
                original_path if origin == "original" else
                retry_paths[next(i for i, row in enumerate(retries) if row.get("correct"))] if origin == "retry" else
                gold_path if origin == "gold_repair" else None
            )
            if selected is None:
                counts["unsolved_tasks"] += 1
            else:
                assert selected_path is not None
                _atomic_json(
                    _record_path(output_dir, task, 0),
                    _unified_record(selected, task_id=task.task_id, origin=origin, source_path=selected_path),
                )
                counts["merged_unique_tasks"] += 1
            index_file.write(json.dumps({
                "task_id": task.task_id, "origin": origin,
                "source_path": str(selected_path.resolve()) if selected_path else None,
            }, ensure_ascii=False) + "\n")
    if counts["gold_repair_total_candidates"] != counts["gold_repair_duplicates_removed"] + counts["gold_repair_unique"]:
        raise AssertionError("Gold deduplication counts do not reconcile")
    if counts["merged_unique_tasks"] + counts["unsolved_tasks"] != 7000:
        raise AssertionError("Merged task counts do not reconcile")
    _atomic_json(output_dir / "merge_report.json", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge correct tagged trajectories by original > retry > gold repair")
    parser.add_argument("--original-dir", type=Path, default=Path("artifacts/sql_planner/spider_train_7000_reasoning_tool_v4"))
    parser.add_argument("--retry-dir", type=Path, default=Path("artifacts/sql_planner/spider_train_7000_reasoning_tool_retry2_v1"))
    parser.add_argument("--gold-dir", type=Path, default=Path("artifacts/sql_planner/spider_train_7000_reasoning_tool_gold_repair_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sql_planner/spider_train_7000_reasoning_tool_sft_merged_v1"))
    parser.add_argument("--spider-root", type=Path, default=Path("../datasets/spider/spider_data"))
    args = parser.parse_args()
    print(json.dumps(merge(
        args.original_dir.expanduser().resolve(), args.retry_dir.expanduser().resolve(),
        args.gold_dir.expanduser().resolve(), args.output_dir.expanduser().resolve(),
        args.spider_root.expanduser().resolve(),
    ), sort_keys=True))


if __name__ == "__main__":
    main()
