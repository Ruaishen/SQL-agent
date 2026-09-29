from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from sqlglot import parse_one
from sqlglot.errors import ParseError, TokenError

from grpo.config import GrpoConfig
from grpo.selection import _read_jsonl, select_diverse_tasks
from sql_agent.tokenizer_check import assert_tokenizers_identical

DIFFICULTIES = ("easy", "medium", "hard", "extra")
SOURCES = ("zero", "boundary", "exploration", "mastered")


def _normalize_sql(sql: str) -> str:
    try:
        return parse_one(sql, read="sqlite").sql(dialect="sqlite", pretty=False, normalize=True)
    except (ParseError, TokenError):
        return " ".join(sql.lower().split())


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = json.dumps(
        [
            task["db_id"],
            " ".join(task["question"].lower().split()),
            _normalize_sql(task["reference_sql"]),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _excluded_task_ids(paths: tuple[Path, ...]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            excluded.update(value.get("task_ids", ()))
            continue
        excluded.update(value["task_id"] for value in _read_jsonl(path) if value.get("task_id"))
    return excluded


def _source_for_count(correct_count: int | None, rollout_count: int) -> str:
    if correct_count is None:
        return "exploration"
    if correct_count == 0:
        return "zero"
    if correct_count == rollout_count:
        return "mastered"
    return "boundary"


def build_curriculum_pool(
    config: GrpoConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if config.candidate_tasks is None or config.ability_map is None:
        raise ValueError("curriculum paths are required")
    all_training_tasks = _read_jsonl(config.train_data)
    task_lookup = {task["task_id"]: task for task in all_training_tasks}
    candidates = _read_jsonl(config.candidate_tasks)
    ability = {value["task_id"]: value for value in _read_jsonl(config.ability_map)}
    excluded_ids = _excluded_task_ids(config.exclusion_files)
    excluded_fingerprints = {
        task_fingerprint(task_lookup[task_id]) for task_id in excluded_ids if task_id in task_lookup
    }

    eligible: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCES}
    seen_fingerprints: set[str] = set()
    for task in sorted(candidates, key=lambda value: value["task_id"]):
        fingerprint = task_fingerprint(task)
        if task["task_id"] in excluded_ids or fingerprint in excluded_fingerprints:
            continue
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        mapped = ability.get(task["task_id"])
        correct_count = None if mapped is None else int(mapped["correct_count"])
        if correct_count is not None and not 0 <= correct_count <= config.rollouts_per_prompt:
            raise ValueError(f"invalid ability bucket for {task['task_id']}")
        source = _source_for_count(correct_count, config.rollouts_per_prompt)
        eligible[source].append(
            {
                "task": task,
                "task_fingerprint": fingerprint,
                "selection_source": source,
                "prior_correct_count": correct_count,
            }
        )

    selected: list[dict[str, Any]] = []
    for source_index, source in enumerate(SOURCES):
        source_records = eligible[source]
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            count = config.source_difficulty_quotas[source][difficulty]
            candidates_for_cell = [
                record for record in source_records if record["task"]["difficulty"] == difficulty
            ]
            rng = random.Random(config.seed + source_index * 101 + difficulty_index)
            rng.shuffle(candidates_for_cell)
            unique_db: list[dict[str, Any]] = []
            repeated_db: list[dict[str, Any]] = []
            seen_databases: set[str] = set()
            for record in candidates_for_cell:
                db_id = record["task"]["db_id"]
                target = unique_db if db_id not in seen_databases else repeated_db
                target.append(record)
                seen_databases.add(db_id)
            chosen = (unique_db + repeated_db)[:count]
            if len(chosen) != count:
                raise ValueError(
                    f"not enough {source}/{difficulty} tasks: requested {count}, "
                    f"found {len(candidates_for_cell)}"
                )
            selected.extend(chosen)

    random.Random(config.seed).shuffle(selected)
    tasks = [record["task"] for record in selected]
    metadata = [
        {
            "task_id": record["task"]["task_id"],
            **{key: value for key, value in record.items() if key != "task"},
        }
        for record in selected
    ]
    selected_ids = {task["task_id"] for task in tasks}
    selected_fingerprints = {record["task_fingerprint"] for record in metadata}
    if len(selected_ids) != config.task_count or len(selected_fingerprints) != config.task_count:
        raise ValueError("curriculum selection contains duplicate tasks or fingerprints")
    id_intersection = selected_ids & excluded_ids
    fingerprint_intersection = selected_fingerprints & excluded_fingerprints
    if id_intersection or fingerprint_intersection:
        raise ValueError("curriculum selection overlaps excluded SFT tasks")
    diagnostics = {
        "excluded_task_ids": len(excluded_ids),
        "excluded_fingerprints": len(excluded_fingerprints),
        "selected_exclusion_id_intersection": len(id_intersection),
        "selected_exclusion_fingerprint_intersection": len(fingerprint_intersection),
        "eligible_source_counts": {source: len(records) for source, records in eligible.items()},
    }
    return tasks, metadata, diagnostics


def prepare(config: GrpoConfig) -> dict:
    tokenizer_hashes = assert_tokenizers_identical(config.student_model, config.reference_model)
    metadata: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    if config.reuse_pool_dir is not None:
        source_pool = config.reuse_pool_dir / "task_pool.jsonl"
        source_metadata = config.reuse_pool_dir / "task_metadata.jsonl"
        if not source_pool.is_file() or not source_metadata.is_file():
            raise FileNotFoundError("reuse_pool_dir must contain task_pool/task_metadata JSONL")
        tasks = _read_jsonl(source_pool)
        metadata = _read_jsonl(source_metadata)
        if len(tasks) != config.task_count or len(metadata) != config.task_count:
            raise ValueError("reused GRPO pool size does not match config")
        if [task["task_id"] for task in tasks] != [record["task_id"] for record in metadata]:
            raise ValueError("reused GRPO pool and metadata order differ")
        diagnostics = {
            "reused_pool_dir": str(config.reuse_pool_dir),
            "reused_pool_sha256": hashlib.sha256(source_pool.read_bytes()).hexdigest(),
            "reused_metadata_sha256": hashlib.sha256(source_metadata.read_bytes()).hexdigest(),
        }
    elif config.source_difficulty_quotas:
        tasks, metadata, diagnostics = build_curriculum_pool(config)
    else:
        source = _read_jsonl(config.train_data)
        tasks = []
        for index, difficulty in enumerate(DIFFICULTIES):
            count = getattr(config, f"{difficulty}_tasks")
            if count:
                tasks.extend(select_diverse_tasks(source, difficulty, count, config.seed + index))
        random.Random(config.seed).shuffle(tasks)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = config.output_dir / "task_pool.jsonl"
    if config.reuse_pool_dir is not None:
        shutil.copyfile(config.reuse_pool_dir / "task_pool.jsonl", pool_path)
        with (config.output_dir / "task_metadata.jsonl").open("w", encoding="utf-8") as handle:
            for record in metadata:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    else:
        with pool_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    if metadata and config.reuse_pool_dir is None:
        with (config.output_dir / "task_metadata.jsonl").open("w", encoding="utf-8") as handle:
            for record in metadata:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "prepared",
        "formal_training_ready": True,
        "purpose": (
            "schema_grounded_hierarchical_grpo_v2"
            if config.reward_mode == "schema_hierarchical_v2"
            else "schema_grounded_hierarchical_grpo"
            if config.reward_mode == "schema_hierarchical"
            else "difficulty_adaptive_turn_grpo"
            if config.reward_mode == "difficulty_adaptive_turn"
            else "curriculum_binary_grpo"
            if metadata
            else "bounded_binary_grpo"
        ),
        "student_initialization": str(config.student_model),
        "reference_model": str(config.reference_model),
        "student_update": "full",
        "reward": config.reward_mode,
        "config": config.to_dict(),
        "task_count": len(tasks),
        "difficulty_counts": {
            difficulty: sum(task["difficulty"] == difficulty for task in tasks)
            for difficulty in DIFFICULTIES
        },
        "selection_source_counts": {
            source: sum(record.get("selection_source") == source for record in metadata)
            for source in SOURCES
        },
        "unique_databases": len({task["db_id"] for task in tasks}),
        "tokenizer_sha256": tokenizer_hashes,
        **diagnostics,
    }
    (config.output_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the bounded SFT-v2 GRPO smoke run")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(GrpoConfig.load(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
