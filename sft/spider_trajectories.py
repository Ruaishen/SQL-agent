from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from sql_agent.tokenizer_check import tokenizer_fingerprint
from sql_planner.collect import api_tools, build_prompt

SOURCE = Path("artifacts/sql_planner/spider_filtered_correct/filtered_correct_trajectories.jsonl")
MODEL = Path("/root/autodl-tmp/Qwen2.5-Coder-3B-Instruct")
OUTPUT = Path("artifacts/sql_planner/spider_filtered_correct_sft")
PROMPT_VERSION = "sql_planner_multi_table_ordered_sft_v1"
SAMPLE_COUNTS = {"easy": 500, "medium": 1000, "hard": 500, "extra": 500}


def build_sft_prompt(max_turns: int) -> str:
    prompt = build_prompt(max_turns)
    old_order = (
        "Choose tools freely based on the information you have; "
        "there is no required tool order, and you may skip or revisit any tool. "
        "Decide independently whether and when schema inspection, value inspection, "
        "or SQL execution is useful."
    )
    new_order = (
        "First call list_tables, then call inspect_tables to inspect relevant tables. "
        "After these two calls, decide independently whether and when further schema "
        "inspection, value inspection, or SQL execution is useful."
    )
    old_freedom = "Otherwise, choose tools freely. "
    new_freedom = "Otherwise, choose tools freely after the required first two calls. "
    old_ending = "Tool outputs can be truncated."
    new_ending = "Tool outputs can be truncated. Every trajectory must end with submit_sql."
    for old, new in (
        (old_order, new_order),
        (old_freedom, new_freedom),
        (old_ending, new_ending),
    ):
        if prompt.count(old) != 1:
            raise ValueError(f"Unexpected original prompt fragment: {old}")
        prompt = prompt.replace(old, new, 1)
    return prompt


def _normalize_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [dict(message) for message in record["messages"]]
    if messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError(f"Unexpected message prefix: {record['task_id']}")
    messages[0]["content"] = build_sft_prompt(int(record["max_turns"]))
    for message in messages:
        if message["role"] != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            raise ValueError(f"Expected one tool call: {record['task_id']}")
        call = dict(calls[0])
        function = dict(call["function"])
        if isinstance(function["arguments"], str):
            function["arguments"] = json.loads(function["arguments"])
        call["function"] = function
        message["tool_calls"] = [call]
    return messages


def tokenize_trajectory(record: dict[str, Any], tokenizer, max_sequence_tokens: int):
    messages = _normalize_messages(record)
    tools = api_tools()
    turns = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(
            messages[:index], tools=tools, tokenize=False, add_generation_prompt=True
        )
        rendered = tokenizer.apply_chat_template(messages[: index + 1], tools=tools, tokenize=False)
        if not rendered.startswith(prefix):
            raise ValueError(f"Assistant prefix mismatch: {record['task_id']}")
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        input_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        if input_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(f"Tokenizer prefix mismatch: {record['task_id']}")
        if len(input_ids) > max_sequence_tokens:
            raise ValueError(
                f"Trajectory turn exceeds {max_sequence_tokens} tokens: {record['task_id']}"
            )
        action_mask = torch.zeros(len(input_ids) - 1, dtype=torch.bool)
        action_mask[len(prefix_ids) - 1 :] = True
        turns.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "action_mask": action_mask,
                "supervised": True,
                "turn": len(turns) + 1,
            }
        )
    if len(turns) != len(record["tool_sequence"]):
        raise ValueError(f"Tool/message count mismatch: {record['task_id']}")
    return turns


def sample_trajectories(source: Path, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {difficulty: [] for difficulty in SAMPLE_COUNTS}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("correct") is not True or record.get("split") != "train":
                raise ValueError(f"Expected a correct train trajectory at line {line_number}")
            difficulty = record["difficulty"]
            if difficulty not in grouped:
                raise ValueError(f"Unexpected difficulty at line {line_number}: {difficulty}")
            grouped[difficulty].append(record)
    available = {difficulty: len(records) for difficulty, records in grouped.items()}
    for difficulty, count in SAMPLE_COUNTS.items():
        if available[difficulty] < count:
            raise ValueError(
                f"Not enough {difficulty} trajectories: {available[difficulty]} < {count}"
            )
    rng = random.Random(seed)
    selected = [
        record
        for difficulty, count in SAMPLE_COUNTS.items()
        for record in rng.sample(grouped[difficulty], count)
    ]
    rng.shuffle(selected)
    return selected, available


def prepare(
    source: Path, model: Path, output: Path, max_sequence_tokens: int, seed: int = 42
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    selected, available = sample_trajectories(source, seed)
    tokenizer = AutoTokenizer.from_pretrained(model)
    output.mkdir(parents=True, exist_ok=True)
    shards_dir = output / "shards"
    shards_dir.mkdir()
    shards = []
    action_tokens = 0
    total_turns = 0
    for index, record in enumerate(selected):
        turns = tokenize_trajectory(record, tokenizer, max_sequence_tokens)
        shard_path = (shards_dir / f"{index:05d}.pt").resolve()
        torch.save({"task_id": record["task_id"], "turns": turns}, shard_path)
        shards.append(str(shard_path))
        total_turns += len(turns)
        action_tokens += sum(int(turn["action_mask"].sum()) for turn in turns)
    manifest = {
        "status": "completed",
        "source_split": "train",
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "prompt": build_sft_prompt(10),
        "sampling": {
            "seed": seed,
            "available_by_difficulty": available,
            "selected_by_difficulty": SAMPLE_COUNTS,
            "selected_task_ids": [record["task_id"] for record in selected],
        },
        "tokenizer_sha256": tokenizer_fingerprint(model),
        "shards": shards,
        "statistics": {
            "trajectories": len(shards),
            "turns": total_turns,
            "action_tokens": action_tokens,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unchanged Spider trajectories for SFT")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.source, args.model, args.output, args.max_sequence_tokens, args.seed)
    print(json.dumps({**result["statistics"], **result["sampling"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
