from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from sql_agent.tokenizer_check import assert_tokenizers_identical
from sft.collect import (
    _atomic_json,
    _load_records,
    _save_shard,
    _summary,
    _write_progress,
    bootstrap_records,
    diverse_task_order,
    rebuild_jsonl_views,
    should_supervise,
    validate_shards,
)
from sft.config import SftCollectionConfig
from sql_agent.action_parser import ActionParseError, parse_action
from sql_agent.models import TaskRecord


def _record_files(output_dir: Path):
    for path in sorted((output_dir / "attempt_records").glob("*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def _diverse_records(records: list[tuple[Path, dict[str, Any]]], count: int, seed: int):
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    first: list[tuple[Path, dict[str, Any]]] = []
    rest: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in ordered:
        db_id = item[1]["db_id"]
        if db_id not in seen:
            first.append(item)
            seen.add(db_id)
        else:
            rest.append(item)
    return (first + rest)[:count]


def trim_successes(config: SftCollectionConfig) -> dict[str, int]:
    by_difficulty: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        difficulty: [] for difficulty in config.target_counts
    }
    for path, record in _record_files(config.output_dir):
        if record["success"]:
            by_difficulty[record["difficulty"]].append((path, record))
    for offset, (difficulty, records) in enumerate(by_difficulty.items()):
        target = config.target_counts[difficulty]
        if len(records) <= target:
            continue
        keep = {
            path
            for path, _ in _diverse_records(records, target, config.seed + offset)
        }
        for path, record in records:
            if path in keep:
                continue
            shard = Path(record["shard"])
            if shard.parent != config.output_dir / "shards":
                raise ValueError(f"refusing to delete shard outside v2 output: {shard}")
            shard.unlink()
            path.unlink()
    return {
        difficulty: len(records)
        for difficulty, records in _successes_by_difficulty(config.output_dir).items()
    }


def _successes_by_difficulty(output_dir: Path):
    result: dict[str, list[dict[str, Any]]] = {
        difficulty: [] for difficulty in ("easy", "medium", "hard", "extra")
    }
    for record in _load_records(output_dir):
        if record["success"]:
            result[record["difficulty"]].append(record)
    return result


def prepare_candidates(config: SftCollectionConfig, destination: Path) -> dict[str, int]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_records(config)
    trim_successes(config)
    records = rebuild_jsonl_views(config.output_dir)
    successful = {record["task_id"] for record in records if record["success"]}
    with config.train_data.open(encoding="utf-8") as handle:
        tasks = [TaskRecord.from_dict(json.loads(line)) for line in handle if line.strip()]
    selected: list[TaskRecord] = []
    counts: dict[str, int] = {}
    factors = {"easy": 2, "medium": 3, "hard": 4, "extra": 8}
    minimums = {"easy": 32, "medium": 64, "hard": 700, "extra": 1200}
    current = _summary(records)
    for offset, difficulty in enumerate(config.difficulties):
        deficit = config.target_counts[difficulty] - current[f"{difficulty}_successes"]
        if deficit <= 0:
            continue
        candidates = [
            task
            for task in diverse_task_order(tasks, difficulty, config.seed + offset)
            if task.task_id not in successful
        ]
        requested = max(minimums[difficulty], deficit * factors[difficulty])
        chosen = candidates[: min(len(candidates), requested)]
        selected.extend(chosen)
        counts[difficulty] = len(chosen)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in selected:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    _write_progress(config, records, 0.0, status="collecting")
    return counts


def _turn_tensors(tokenizer, step: dict[str, Any]):
    prompt_ids = tokenizer.apply_chat_template(
        step["messages"],
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )[0]
    response_ids = torch.tensor(
        tokenizer.encode(step["response"], add_special_tokens=False), dtype=torch.long
    )
    input_ids = torch.cat((prompt_ids, response_ids))
    action_mask = torch.zeros(input_ids.numel() - 1, dtype=torch.bool)
    action_mask[prompt_ids.numel() - 1 :] = True
    targets = input_ids[1:]
    for token_id in tokenizer.all_special_ids:
        action_mask &= targets.ne(token_id)
    action = None
    with contextlib.suppress(ActionParseError):
        action = parse_action(step["response"])
    supervised = should_supervise(action, step["observation"])
    if not supervised:
        action_mask.zero_()
    return input_ids, action_mask, supervised, prompt_ids.numel()


def import_successes(
    config: SftCollectionConfig, trajectories: Path, *, seed: int
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)
    records = rebuild_jsonl_views(config.output_dir)
    successful_ids = {record["task_id"] for record in records if record["success"]}
    current = _summary(records)
    values = [json.loads(line) for line in trajectories.read_text().splitlines() if line]
    accepted = [
        value
        for value in values
        if value["success"] and value["task_id"] not in successful_ids
    ]
    accepted.sort(key=lambda value: (value["difficulty"], value["db_id"], value["task_id"]))
    db_counts = Counter(record["db_id"] for record in records if record["success"])
    accepted.sort(key=lambda value: db_counts[value["db_id"]])
    imported = Counter()
    for value in accepted:
        difficulty = value["difficulty"]
        if current[f"{difficulty}_successes"] >= config.target_counts[difficulty]:
            continue
        shard_turns = []
        turn_metadata = []
        for number, step in enumerate(value["steps"], 1):
            input_ids, action_mask, supervised, prompt_length = _turn_tensors(tokenizer, step)
            metadata = {
                "turn": number,
                "response": step["response"],
                "observation": step["observation"],
                "parse_error": None,
                "supervised": supervised,
                "action_tokens": int(action_mask.sum()),
                "prompt_length": prompt_length,
                "sequence_length": input_ids.numel(),
            }
            turn_metadata.append(metadata)
            shard_turns.append(
                {
                    "turn": number,
                    "input_ids": input_ids,
                    "prompt_length": prompt_length,
                    "action_mask": action_mask,
                    "response": step["response"],
                    "observation": step["observation"],
                    "supervised": supervised,
                }
            )
        tokens = sum(turn["action_tokens"] for turn in turn_metadata)
        supervised_turns = sum(turn["supervised"] for turn in turn_metadata)
        if tokens == 0 or not turn_metadata[-1]["supervised"]:
            continue
        digest = hashlib.sha256(value["task_id"].encode()).hexdigest()[:12]
        name = f"{difficulty}_{digest}_api_seed_{seed}"
        shard_path = config.output_dir / "shards" / f"{name}.pt"
        _save_shard(
            shard_path,
            {
                "format_version": 1,
                "task_id": value["task_id"],
                "db_id": value["db_id"],
                "difficulty": difficulty,
                "attempt": seed,
                "turns": shard_turns,
            },
        )
        trajectory = {
            "task_id": value["task_id"],
            "db_id": value["db_id"],
            "difficulty": difficulty,
            "attempt": seed,
            "success": True,
            "submitted": True,
            "turns": turn_metadata,
            "supervised_turns": supervised_turns,
            "supervised_action_tokens": tokens,
        }
        _atomic_json(
            config.output_dir / "attempt_records" / f"{name}.json",
            {
                "format_version": 1,
                "task_id": value["task_id"],
                "db_id": value["db_id"],
                "difficulty": difficulty,
                "split": "train",
                "attempt": seed,
                "success": True,
                "failure_reason": None,
                "attempt_seconds": value["latency_seconds"],
                "shard": str(shard_path),
                "collection_backend": "vllm_openai_import",
                "trajectory": trajectory,
            },
        )
        current[f"{difficulty}_successes"] += 1
        imported[difficulty] += 1
    records = rebuild_jsonl_views(config.output_dir)
    progress = _write_progress(config, records, 0.0, status="collecting")
    return {"imported": dict(imported), "progress": progress}


def finalize(config: SftCollectionConfig) -> dict[str, Any]:
    records = rebuild_jsonl_views(config.output_dir)
    stats = _summary(records)
    for difficulty, target in config.target_counts.items():
        if stats[f"{difficulty}_successes"] != target:
            raise ValueError(f"cannot finalize incomplete {difficulty} quota: {stats}")
    shards = validate_shards(
        config.output_dir, records, target_counts=config.target_counts
    )
    tokenizer_hashes = assert_tokenizers_identical(
        config.student_model, config.teacher_model
    )
    bootstrap_path = config.output_dir / "bootstrap.json"
    bootstrap = (
        json.loads(bootstrap_path.read_text(encoding="utf-8"))
        if bootstrap_path.exists()
        else {}
    )
    elapsed = float(bootstrap.get("source_elapsed_seconds", 0.0))
    manifest = {
        "status": "completed",
        "format_version": 1,
        "config": config.to_dict(),
        "tokenizer_sha256": tokenizer_hashes,
        "source_split": "train",
        "validation_data_used": False,
        "collection_backends": ["transformers", "vllm_openai_import"],
        "statistics": stats,
        "elapsed_seconds": elapsed,
        "shards": [str(path) for path in shards],
    }
    _atomic_json(config.output_dir / "manifest.json", manifest)
    _write_progress(config, records, elapsed, status="completed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/import concurrent Teacher SFT data")
    parser.add_argument("command", choices=("prepare", "import", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = SftCollectionConfig.load(args.config)
    if args.command == "finalize":
        result = finalize(config)
    elif args.path is None:
        parser.error("--path is required for prepare/import")
    elif args.command == "prepare":
        result = prepare_candidates(config, args.path)
    else:
        result = import_successes(config, args.path, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
