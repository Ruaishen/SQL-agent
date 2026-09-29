"""Retry failed tagged Spider tasks with independent full rollouts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from sql_agent.config import EnvConfig
from sql_agent.models import TaskRecord
from sql_planner.collect import _atomic_json, _record_path, ensure_run_manifest, read_spider_train_tasks
from sql_planner.collect_tagged import collect_trajectory
from sql_planner.deepseek import DeepSeekClient
from sql_planner.repair_tagged import validate_complete_source

RETRY_VERSION = "tagged_failed_task_retry_v1"


def failed_task_ids(source_files: list[Path]) -> set[str]:
    failed: set[str] = set()
    seen: set[str] = set()
    for path in source_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        task_id = record["task_id"]
        if task_id in seen:
            raise ValueError(f"Duplicate source task: {task_id}")
        seen.add(task_id)
        if not record.get("correct"):
            failed.add(task_id)
    if len(seen) != 7000:
        raise ValueError("Source must contain 7000 unique tasks")
    return failed


def retry_task(
    task: TaskRecord, config: EnvConfig, client: DeepSeekClient, output_dir: Path,
    *, max_attempts: int, temperature: float, max_tokens: int,
) -> dict[str, Any]:
    """Resume existing attempts; stop at the first verified-correct rollout."""
    attempted = 0
    created = 0
    correct = False
    for attempt in range(1, max_attempts + 1):
        path = _record_path(output_dir, task, attempt)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("task_id") != task.task_id or record.get("sample") != attempt:
                raise ValueError(f"Existing retry file does not match task/attempt: {path}")
        else:
            record = collect_trajectory(
                task, config, client, sample=attempt,
                temperature=temperature, max_tokens=max_tokens,
            )
            _atomic_json(path, record)
            created += 1
        attempted += 1
        if record.get("correct"):
            correct = True
            break
    return {"task_id": task.task_id, "attempted": attempted, "created": created, "rescued": correct}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry each failed tagged Spider task until correct or attempt limit")
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/sql_planner/spider_train_7000_reasoning_tool_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sql_planner/spider_train_7000_reasoning_tool_retry2_v1"))
    parser.add_argument("--spider-root", type=Path, default=Path("../datasets/spider/spider_data"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env_sql_planner_local_multi_table_v11.yaml"))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Optional pilot task count after source completion")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.max_attempts < 1 or args.max_attempts > 99 or args.limit < 0 or args.workers < 1 or args.max_tokens < 1:
        parser.error("Invalid numeric argument")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY in the environment")
    source_dir = args.source_dir.expanduser().resolve()
    source_manifest, source_files = validate_complete_source(source_dir)
    spider_root = args.spider_root.expanduser().resolve()
    raw_spider_path = spider_root / "train_spider.json"
    if hashlib.sha256(raw_spider_path.read_bytes()).hexdigest() != source_manifest["source_sha256"]:
        parser.error("Spider source does not match original collection")
    if hashlib.sha256(args.env_config.read_bytes()).hexdigest() != source_manifest["env_config_sha256"]:
        parser.error("Environment config does not match original collection")
    failed = failed_task_ids(source_files)
    tasks = [task for task in read_spider_train_tasks(spider_root, limit=0, seed=42) if task.task_id in failed]
    if args.limit:
        tasks = tasks[:args.limit]
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=spider_root)
    ensure_run_manifest(args.output_dir, {
        "schema_version": 1, "retry_version": RETRY_VERSION,
        "source_dir": str(source_dir),
        "source_manifest_sha256": hashlib.sha256((source_dir / "run_manifest.json").read_bytes()).hexdigest(),
        "source_failed_task_count": len(failed), "selected_task_count": len(tasks),
        "spider_source_sha256": source_manifest["source_sha256"],
        "env_config_sha256": source_manifest["env_config_sha256"],
        "model": args.model, "base_url": args.base_url,
        "max_attempts": args.max_attempts, "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    })
    client = DeepSeekClient(api_key, model=args.model, base_url=args.base_url)

    def run(task: TaskRecord) -> dict[str, Any]:
        return retry_task(
            task, config, client, args.output_dir, max_attempts=args.max_attempts,
            temperature=args.temperature, max_tokens=args.max_tokens,
        )

    counts = {"tasks": len(tasks), "processed": 0, "created": 0, "attempts_total": 0, "rescued": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run, tasks):
            counts["processed"] += 1
            counts["created"] += result["created"]
            counts["attempts_total"] += result["attempted"]
            counts["rescued"] += int(result["rescued"])
            if counts["processed"] % 25 == 0:
                print(json.dumps(counts, sort_keys=True), file=sys.stderr, flush=True)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
