from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.runner import extract_sql
from sql_agent.tokenizer_check import assert_tokenizers_identical
from sft.config import SftCollectionConfig
from sql_agent.action_parser import ActionParseError, ExecuteSQLAction, parse_action
from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.models import TaskRecord
from sql_agent.prompts import render_agent_messages


def _read_tasks(path: Path) -> list[TaskRecord]:
    with path.open(encoding="utf-8") as handle:
        tasks = [TaskRecord.from_dict(json.loads(line)) for line in handle if line.strip()]
    invalid = [task.task_id for task in tasks if task.split != "train"]
    if invalid:
        raise ValueError(f"SFT collector received non-train tasks: {invalid[:3]}")
    return tasks


def diverse_task_order(
    tasks: list[TaskRecord], difficulty: str, seed: int
) -> list[TaskRecord]:
    candidates = [task for task in tasks if task.difficulty == difficulty]
    random.Random(seed).shuffle(candidates)
    first_by_database: list[TaskRecord] = []
    remaining: list[TaskRecord] = []
    seen: set[str] = set()
    for task in candidates:
        if task.db_id not in seen:
            first_by_database.append(task)
            seen.add(task.db_id)
        else:
            remaining.append(task)
    return first_by_database + remaining


def should_supervise(action: Any, observation: dict[str, Any]) -> bool:
    if action is None or observation.get("status") == "error":
        return False
    if "verification" in observation:
        return isinstance(action, ExecuteSQLAction) and bool(
            observation.get("verification", {}).get("correct", False)
        )
    return True


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_name(task: TaskRecord, attempt: int) -> str:
    digest = hashlib.sha256(task.task_id.encode()).hexdigest()[:12]
    return f"{task.difficulty}_{digest}_attempt_{attempt:02d}"


def _load_records(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((output_dir / "attempt_records").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def rebuild_jsonl_views(output_dir: Path) -> list[dict[str, Any]]:
    records = _load_records(output_dir)
    attempts_tmp = output_dir / "attempts.jsonl.tmp"
    successes_tmp = output_dir / "successful_trajectories.jsonl.tmp"
    with attempts_tmp.open("w", encoding="utf-8") as attempts, successes_tmp.open(
        "w", encoding="utf-8"
    ) as successes:
        for record in records:
            attempts.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if record["success"]:
                successes.write(
                    json.dumps(record["trajectory"], ensure_ascii=False, sort_keys=True) + "\n"
                )
    attempts_tmp.replace(output_dir / "attempts.jsonl")
    successes_tmp.replace(output_dir / "successful_trajectories.jsonl")
    return records


def bootstrap_records(config: SftCollectionConfig) -> float:
    source = config.bootstrap_from
    if source is None:
        return 0.0
    bootstrap_path = config.output_dir / "bootstrap.json"
    if any((config.output_dir / "attempt_records").glob("*.json")):
        if bootstrap_path.exists():
            value = json.loads(bootstrap_path.read_text(encoding="utf-8"))
            return float(value.get("source_elapsed_seconds", 0.0))
        return 0.0
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("source_split") != "train":
        raise ValueError("SFT bootstrap dataset must be completed and train-only")
    for source_path in sorted((source / "attempt_records").glob("*.json")):
        record = json.loads(source_path.read_text(encoding="utf-8"))
        if record["success"]:
            source_shard = Path(record["shard"])
            destination = config.output_dir / "shards" / source_shard.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_shard, destination)
            record["shard"] = str(destination)
        _atomic_json(config.output_dir / "attempt_records" / source_path.name, record)
    _atomic_json(
        bootstrap_path,
        {
            "source": str(source),
            "source_manifest": str(source / "manifest.json"),
            "source_successes": manifest["statistics"]["successes"],
            "source_elapsed_seconds": manifest.get("elapsed_seconds", 0.0),
        },
    )
    return float(manifest.get("elapsed_seconds", 0.0))


def _tokenize_prompt(tokenizer, messages: list[dict[str, str]]) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")


def _generation_seed(config: SftCollectionConfig, task_id: str, attempt: int, turn: int) -> int:
    task_seed = int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big")
    return config.seed + task_seed + attempt * 1_000_003 + turn


def collect_attempt(
    teacher,
    tokenizer,
    task: TaskRecord,
    attempt: int,
    env_config: EnvConfig,
    config: SftCollectionConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    env = SQLAgentEnv(env_config)
    special_ids = set(tokenizer.all_special_ids)
    turns: list[dict[str, Any]] = []
    shard_turns: list[dict[str, Any]] = []
    success = submitted = False
    try:
        env.reset(task)
        while not env.done and len(turns) < config.max_assistant_turns:
            prompt_ids = _tokenize_prompt(tokenizer, render_agent_messages(env.history))
            if prompt_ids.shape[1] + config.max_action_tokens > config.max_sequence_tokens:
                raise ValueError(f"task {task.task_id} prompt exceeds SFT sequence budget")
            prompt_mask = torch.ones_like(prompt_ids)
            seed = _generation_seed(config, task.task_id, attempt, len(turns))
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sequence = teacher.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        do_sample=True,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        max_new_tokens=config.max_action_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            prompt_length = prompt_ids.shape[1]
            generated_ids = sequence[0, prompt_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not response:
                response = extract_sql(tokenizer.decode(generated_ids).strip())
            action = None
            parse_error = None
            try:
                action = parse_action(response)
            except ActionParseError as exc:
                parse_error = exc.error_type
            observation, done = env.step(response)
            if done and "verification" in observation:
                submitted = True
                success = bool(observation.get("verification", {}).get("correct", False))

            target_mask = torch.zeros(sequence.shape[1] - 1, dtype=torch.bool, device="cuda")
            target_mask[prompt_length - 1 :] = True
            targets = sequence[0, 1:]
            for token_id in special_ids:
                target_mask &= targets.ne(token_id)
            supervised = should_supervise(action, observation)
            if not supervised:
                target_mask.zero_()
            turn_number = len(turns) + 1
            turn_metadata = {
                "turn": turn_number,
                "response": response,
                "observation": observation,
                "parse_error": parse_error,
                "supervised": supervised,
                "action_tokens": int(target_mask.sum().item()),
                "prompt_length": prompt_length,
                "sequence_length": int(sequence.shape[1]),
            }
            turns.append(turn_metadata)
            shard_turns.append(
                {
                    "turn": turn_number,
                    "input_ids": sequence[0].detach().cpu(),
                    "prompt_length": prompt_length,
                    "action_mask": target_mask.detach().cpu(),
                    "response": response,
                    "observation": observation,
                    "supervised": supervised,
                }
            )
            del sequence, prompt_ids, prompt_mask, generated_ids, targets, target_mask
            if done:
                break
    finally:
        env.close()

    supervised_turns = sum(turn["supervised"] for turn in turns)
    supervised_tokens = sum(turn["action_tokens"] for turn in turns)
    accepted = success and turns[-1]["supervised"] and supervised_tokens > 0
    trajectory = {
        "task_id": task.task_id,
        "db_id": task.db_id,
        "difficulty": task.difficulty,
        "attempt": attempt,
        "success": accepted,
        "submitted": submitted,
        "turns": turns,
        "supervised_turns": supervised_turns,
        "supervised_action_tokens": supervised_tokens,
    }
    shard = None
    if accepted:
        shard = {
            "format_version": 1,
            "task_id": task.task_id,
            "db_id": task.db_id,
            "difficulty": task.difficulty,
            "attempt": attempt,
            "turns": shard_turns,
        }
    return trajectory, shard


def _failure_reason(trajectory: dict[str, Any]) -> str | None:
    if trajectory["success"]:
        return None
    if not trajectory["submitted"]:
        return "no_final_sql"
    return "incorrect_final_sql"


def _save_shard(path: Path, shard: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(shard, temporary)
    temporary.replace(path)


def validate_shards(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    target_counts: dict[str, int],
) -> list[Path]:
    successes = [record for record in records if record["success"]]
    expected_total = sum(target_counts.values())
    if len(successes) != expected_total:
        raise ValueError(f"expected {expected_total} successful records, found {len(successes)}")
    task_ids = [record["task_id"] for record in successes]
    if len(set(task_ids)) != expected_total:
        raise ValueError("successful SFT trajectories contain duplicate tasks")
    counts = {
        difficulty: sum(record["difficulty"] == difficulty for record in successes)
        for difficulty in target_counts
    }
    if counts != target_counts:
        raise ValueError(f"unexpected successful difficulty counts: {counts}")

    referenced = {Path(record["shard"]) for record in successes}
    shards = set((output_dir / "shards").glob("*.pt"))
    if shards != referenced:
        raise ValueError("successful records and shard files do not match exactly")
    for path in sorted(shards):
        shard = torch.load(path, map_location="cpu", weights_only=True)
        if shard.get("format_version") != 1 or shard.get("task_id") not in task_ids:
            raise ValueError(f"invalid shard identity: {path}")
        turns = shard.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"shard has no turns: {path}")
        for turn in turns:
            input_ids = turn.get("input_ids")
            action_mask = turn.get("action_mask")
            prompt_length = turn.get("prompt_length")
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.ndim != 1
                or not isinstance(action_mask, torch.Tensor)
                or action_mask.dtype != torch.bool
                or action_mask.ndim != 1
                or action_mask.numel() != input_ids.numel() - 1
                or not isinstance(prompt_length, int)
                or not 1 <= prompt_length <= input_ids.numel()
                or action_mask[: prompt_length - 1].any()
                or bool(action_mask.any()) != bool(turn.get("supervised"))
            ):
                raise ValueError(f"invalid token mask in shard: {path}")
        final_turn = turns[-1]
        verification = final_turn.get("observation", {}).get("verification", {})
        if not final_turn.get("supervised") or not verification.get("correct", False):
            raise ValueError(f"shard does not end in a supervised correct final SQL: {path}")
    return sorted(shards)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record["success"]]
    return {
        "attempts": len(records),
        "successes": len(successes),
        "easy_successes": sum(record["difficulty"] == "easy" for record in successes),
        "medium_successes": sum(record["difficulty"] == "medium" for record in successes),
        "hard_successes": sum(record["difficulty"] == "hard" for record in successes),
        "extra_successes": sum(record["difficulty"] == "extra" for record in successes),
        "unique_tasks": len({record["task_id"] for record in successes}),
        "unique_databases": len({record["db_id"] for record in successes}),
        "turns": sum(record["trajectory"]["supervised_turns"] for record in successes),
        "action_tokens": sum(
            record["trajectory"]["supervised_action_tokens"] for record in successes
        ),
        "masked_error_turns": sum(
            not turn["supervised"]
            for record in successes
            for turn in record["trajectory"]["turns"]
        ),
    }


def _write_progress(
    config: SftCollectionConfig,
    records: list[dict[str, Any]],
    elapsed_seconds: float,
    *,
    status: str,
) -> dict[str, Any]:
    stats = _summary(records)
    success_rate = stats["successes"] / max(1, stats["attempts"])
    remaining = config.target_total - stats["successes"]
    average_attempt = elapsed_seconds / max(1, stats["attempts"])
    eta = remaining / max(success_rate, 0.01) * average_attempt
    progress = {
        "status": status,
        "target_total": config.target_total,
        "target_easy": config.target_easy,
        "target_medium": config.target_medium,
        "target_hard": config.target_hard,
        "target_extra": config.target_extra,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": 0.0 if status == "completed" else eta,
        "estimated_finish_utc": (
            datetime.now(UTC) + timedelta(seconds=0.0 if status == "completed" else eta)
        ).isoformat(timespec="seconds"),
        "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        **stats,
    }
    _atomic_json(config.output_dir / "progress.json", progress)
    return progress


def collect(config: SftCollectionConfig, env_config: EnvConfig) -> dict[str, Any]:
    config.validate()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    bootstrap_elapsed = bootstrap_records(config)
    tokenizer_hashes = assert_tokenizers_identical(config.student_model, config.teacher_model)
    tasks = _read_tasks(config.train_data)
    counts = {
        difficulty: sum(task.difficulty == difficulty for task in tasks)
        for difficulty in config.difficulties
    }
    if any(counts[difficulty] < config.target_counts[difficulty] for difficulty in counts):
        raise ValueError(f"insufficient train tasks for SFT targets: {counts}")

    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    teacher.requires_grad_(False)
    teacher.eval()
    records = rebuild_jsonl_views(output_dir)
    previous_elapsed = bootstrap_elapsed
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        previous_elapsed = float(
            json.loads(progress_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0)
        )
    run_started = time.monotonic()

    for attempt in range(config.max_attempts_per_task):
        orders = {
            difficulty: diverse_task_order(
                tasks,
                difficulty,
                config.seed + attempt * 10_007 + (0 if difficulty == "easy" else 1),
            )
            for difficulty in config.difficulties
        }
        indices = dict.fromkeys(config.difficulties, 0)
        while True:
            stats = _summary(records)
            needs = {
                difficulty: stats[f"{difficulty}_successes"]
                < config.target_counts[difficulty]
                for difficulty in config.difficulties
            }
            if not any(needs.values()):
                break
            made_progress = False
            successful_tasks = {record["task_id"] for record in records if record["success"]}
            recorded = {
                (record["task_id"], int(record["attempt"])) for record in records
            }
            for difficulty in config.difficulties:
                if not needs[difficulty]:
                    continue
                order = orders[difficulty]
                while indices[difficulty] < len(order):
                    task = order[indices[difficulty]]
                    indices[difficulty] += 1
                    if task.task_id in successful_tasks or (task.task_id, attempt) in recorded:
                        continue
                    started = time.monotonic()
                    trajectory, shard = collect_attempt(
                        teacher, tokenizer, task, attempt, env_config, config
                    )
                    name = _record_name(task, attempt)
                    shard_path = None
                    if shard is not None:
                        shard_path = output_dir / "shards" / f"{name}.pt"
                        _save_shard(shard_path, shard)
                    record = {
                        "format_version": 1,
                        "task_id": task.task_id,
                        "db_id": task.db_id,
                        "difficulty": task.difficulty,
                        "split": task.split,
                        "attempt": attempt,
                        "success": trajectory["success"],
                        "failure_reason": _failure_reason(trajectory),
                        "attempt_seconds": time.monotonic() - started,
                        "shard": None if shard_path is None else str(shard_path),
                        "trajectory": trajectory,
                    }
                    _atomic_json(output_dir / "attempt_records" / f"{name}.json", record)
                    records = rebuild_jsonl_views(output_dir)
                    elapsed = previous_elapsed + time.monotonic() - run_started
                    progress = _write_progress(config, records, elapsed, status="collecting")
                    print(json.dumps(progress, sort_keys=True), flush=True)
                    made_progress = True
                    break
            if not made_progress:
                break
        stats = _summary(records)
        if all(
            stats[f"{difficulty}_successes"] >= config.target_counts[difficulty]
            for difficulty in config.difficulties
        ):
            break

    elapsed = previous_elapsed + time.monotonic() - run_started
    stats = _summary(records)
    if any(
        stats[f"{difficulty}_successes"] != config.target_counts[difficulty]
        for difficulty in config.target_counts
    ):
        _write_progress(config, records, elapsed, status="failed")
        raise RuntimeError(f"Teacher collection did not reach exact targets: {stats}")
    shards = validate_shards(
        output_dir,
        records,
        target_counts=config.target_counts,
    )
    manifest = {
        "status": "completed",
        "format_version": 1,
        "config": config.to_dict(),
        "tokenizer_sha256": tokenizer_hashes,
        "source_split": "train",
        "validation_data_used": False,
        "statistics": stats,
        "elapsed_seconds": elapsed,
        "shards": [str(path) for path in shards],
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    _write_progress(config, records, elapsed, status="completed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect successful Teacher SFT trajectories")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-per-difficulty", type=int)
    args = parser.parse_args()
    config = SftCollectionConfig.load(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.target_per_difficulty is not None:
        if args.target_per_difficulty < 1:
            raise ValueError("--target-per-difficulty must be positive")
        config = replace(
            config,
            **{
                f"target_{difficulty}": args.target_per_difficulty
                for difficulty in config.difficulties
            },
        )
    result = collect(config, EnvConfig.from_yaml(args.env_config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
