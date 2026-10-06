from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from sql_agent.tokenizer_check import tokenizer_fingerprint


def _trainable_turns(record: dict, assistant_count: int, path: Path) -> set[int]:
    if record.get("merged_origin") == "gold_repair":
        cutoff = record.get("cutoff_turn")
        if type(cutoff) is not int or not 1 <= cutoff <= assistant_count:
            raise ValueError(f"Gold repair has an invalid cutoff_turn: {path}")
        return set(range(cutoff, assistant_count + 1))
    return set(record["trainable_turn_numbers"])


def prepare(
    source: Path, model: Path, output: Path, max_sequence_tokens: int,
    selection: Path | None = None,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    if selection is None:
        paths = sorted((source / "trajectories").glob("*.json"))
    else:
        selected = json.loads(selection.read_text(encoding="utf-8"))
        paths = [source / "trajectories" / name for name in selected["selected_files"]]
        if len(paths) != len(set(paths)) or any(not path.is_file() for path in paths):
            raise ValueError("Selection has duplicate or missing trajectory files")
    if not paths:
        raise ValueError("No trajectories found")
    tokenizer = AutoTokenizer.from_pretrained(model)
    output.mkdir(parents=True, exist_ok=True)
    shards_dir = output / "shards"
    shards_dir.mkdir()
    prompt = None
    shards = []
    turns_total = 0
    tokens_total = 0
    max_length = 0
    for index, path in enumerate(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        train_split = record.get("split") == "train" or (
            record.get("split") is None
            and record.get("merged_origin") == "gold_repair"
            and str(record.get("task_id", "")).startswith("spider_train_")
        )
        if record.get("correct") is not True or not train_split:
            raise ValueError(f"Expected a correct training trajectory: {path}")
        messages = record["messages"]
        if messages[0]["role"] != "system" or messages[1]["role"] != "user":
            raise ValueError(f"Unexpected message prefix: {path}")
        if prompt is None:
            prompt = messages[0]["content"]
        elif messages[0]["content"] != prompt:
            raise ValueError(f"Mixed system prompts: {path}")
        assistant_count = sum(message["role"] == "assistant" for message in messages)
        trainable = _trainable_turns(record, assistant_count, path)
        if not trainable or not trainable.issubset(set(range(1, assistant_count + 1))):
            raise ValueError(f"Invalid trainable turn numbers: {path}")
        turns = []
        turn_number = 0
        for message_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            turn_number += 1
            if turn_number not in trainable:
                continue
            prefix = tokenizer.apply_chat_template(
                messages[:message_index], tokenize=False, add_generation_prompt=True
            )
            rendered = tokenizer.apply_chat_template(
                messages[: message_index + 1], tokenize=False, add_generation_prompt=False
            )
            if not rendered.startswith(prefix):
                raise ValueError(f"Assistant prefix mismatch: {path}, turn {turn_number}")
            prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
            input_ids = tokenizer(rendered, add_special_tokens=False).input_ids
            if input_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError(f"Tokenizer prefix mismatch: {path}, turn {turn_number}")
            if len(input_ids) > max_sequence_tokens:
                raise ValueError(f"Sequence exceeds {max_sequence_tokens} tokens: {path}, turn {turn_number}")
            action_mask = torch.zeros(len(input_ids) - 1, dtype=torch.bool)
            action_mask[len(prefix_ids) - 1 :] = True
            turns.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "action_mask": action_mask,
                "supervised": True,
                "turn": turn_number,
            })
            tokens_total += int(action_mask.sum())
            max_length = max(max_length, len(input_ids))
        if len(turns) != len(trainable):
            raise ValueError(f"Trainable turn mismatch: {path}")
        shard_path = (shards_dir / f"{index:05d}.pt").resolve()
        torch.save({"task_id": record["task_id"], "turns": turns}, shard_path)
        shards.append(str(shard_path))
        turns_total += len(turns)
        if (index + 1) % 500 == 0:
            print(f"Prepared {index + 1}/{len(paths)} trajectories", flush=True)
    manifest = {
        "status": "completed",
        "source_split": "train",
        "source_path": str(source.resolve()),
        "selection_path": None if selection is None else str(selection.resolve()),
        "selection_sha256": None if selection is None else hashlib.sha256(selection.read_bytes()).hexdigest(),
        "prompt_version": "reasoning_tool_observation_v4",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "tokenizer_sha256": tokenizer_fingerprint(model),
        "shards": shards,
        "statistics": {
            "trajectories": len(paths),
            "turns": turns_total,
            "action_tokens": tokens_total,
            "max_sequence_tokens": max_length,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest["statistics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.source, args.model, args.output, args.max_sequence_tokens, args.selection
    )))


if __name__ == "__main__":
    main()
