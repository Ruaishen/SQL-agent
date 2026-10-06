"""Tokenize completed DPO pairs using the SFT model's chat template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from sql_agent.tokenizer_check import tokenizer_fingerprint
from dpo.dataset import load_audited_pairs, sha256, validate_pair


def tokenize_branch(tokenizer, messages: list[dict], fork_turn: int,
                    max_sequence_tokens: int) -> dict:
    assistant_seen = 0
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(ids) > max_sequence_tokens:
        raise ValueError(f"Branch is {len(ids)} tokens, over {max_sequence_tokens}")
    mask = torch.zeros(len(ids) - 1, dtype=torch.bool)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        assistant_seen += 1
        if assistant_seen < fork_turn:
            continue
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=False,
                                                add_generation_prompt=True)
        response = tokenizer.apply_chat_template(messages[:index + 1], tokenize=False,
                                                  add_generation_prompt=False)
        if not response.startswith(prefix) or not rendered.startswith(response):
            raise ValueError("Chat template does not preserve assistant turn prefixes")
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        response_ids = tokenizer(response, add_special_tokens=False).input_ids
        if ids[:len(response_ids)] != response_ids or response_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Tokenizer changes prefix tokens at the fork")
        if len(response_ids) <= len(prefix_ids):
            raise ValueError("Empty assistant response")
        mask[len(prefix_ids) - 1:len(response_ids) - 1] = True
    if not mask.any():
        raise ValueError("Branch has no trainable assistant tokens")
    return {"input_ids": torch.tensor(ids, dtype=torch.long), "loss_mask": mask,
            "assistant_turns": assistant_seen, "loss_tokens": int(mask.sum())}


def prepare(pairs_dir: Path, model: Path, output_dir: Path,
            max_sequence_tokens: int) -> dict:
    report_path = pairs_dir.parent / "generation_report.json"
    selection_path = pairs_dir.parent / "selection.json"
    if not report_path.is_file() or not selection_path.is_file():
        raise ValueError("Missing DPO selection or generation report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (report.get("status") != "completed"
            or report.get("eligible_pairs") != selection.get("eligible_count")):
        raise ValueError("DPO generation is incomplete")
    paths = sorted(pairs_dir.glob("*.json"))
    expected_files = {entry["file"] for entry in selection["entries"] if entry["eligible"]}
    if {path.name for path in paths} != expected_files:
        raise ValueError("DPO pair files do not match the eligible selection")
    pairs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return prepare_pairs(pairs, model, output_dir, max_sequence_tokens,
                         {"source_kind": "legacy_directory", "pairs_dir": str(pairs_dir.resolve())})


def prepare_audited(pairs_jsonl: Path, audit_roots: list[Path], model: Path,
                    output_dir: Path, max_sequence_tokens: int) -> dict:
    pairs, source = load_audited_pairs(pairs_jsonl, audit_roots)
    return prepare_pairs(pairs, model, output_dir, max_sequence_tokens, source)


def prepare_pairs(pairs: list[dict], model: Path, output_dir: Path,
                  max_sequence_tokens: int, source: dict) -> dict:
    if not pairs or max_sequence_tokens < 1:
        raise ValueError("Empty dataset or invalid sequence limit")
    seen = set()
    for pair in pairs:
        validate_pair(pair)
        if pair["task_id"] in seen:
            raise ValueError(f"Duplicate training pair: {pair['task_id']}")
        seen.add(pair["task_id"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir()
    shards = []
    max_length = 0
    total_tokens = 0
    shard_hashes = {}
    pair_stats = []
    for index, pair in enumerate(pairs):
        task_id = pair["task_id"]
        fork = pair["fork_turn"]
        chosen = tokenize_branch(tokenizer, pair["chosen_messages"], fork, max_sequence_tokens)
        rejected = tokenize_branch(tokenizer, pair["rejected_messages"], fork, max_sequence_tokens)
        if (pair["chosen_messages"][:2 * fork] != pair["prompt_messages"]
                or pair["rejected_messages"][:2 * fork] != pair["prompt_messages"]):
            raise ValueError(f"Shared prompt changed: {task_id}")
        shard = shards_dir / f"{index:05d}.pt"
        torch.save({"task_id": task_id, "fork_turn": fork,
                    "chosen": chosen, "rejected": rejected}, shard)
        relative = shard.relative_to(output_dir).as_posix()
        shards.append(relative)
        shard_hashes[relative] = sha256(shard)
        pair_stats.append({"task_id": task_id, "fork_turn": fork,
                           "fork_reason": pair.get("fork_reason"),
                           "chosen_tokens": chosen["input_ids"].numel(),
                           "rejected_tokens": rejected["input_ids"].numel(),
                           "chosen_loss_tokens": chosen["loss_tokens"],
                           "rejected_loss_tokens": rejected["loss_tokens"]})
        max_length = max(max_length, chosen["input_ids"].numel(), rejected["input_ids"].numel())
        total_tokens += chosen["loss_tokens"] + rejected["loss_tokens"]
    manifest = {"status": "completed", "source_split": "train", **source,
                "tokenizer_sha256": tokenizer_fingerprint(model),
                "model": str(model.resolve()), "pair_count": len(shards),
                "max_sequence_tokens": max_length, "loss_tokens": total_tokens,
                "sequence_limit": max_sequence_tokens,
                "shards": shards, "shard_sha256": shard_hashes, "pair_stats": pair_stats}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs-dir", type=Path)
    group.add_argument("--pairs-jsonl", type=Path)
    parser.add_argument("--audit-roots", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    args = parser.parse_args()
    if args.pairs_jsonl:
        if not args.audit_roots:
            parser.error("--pairs-jsonl requires --audit-roots")
        result = prepare_audited(args.pairs_jsonl, args.audit_roots, args.model,
                                 args.output_dir, args.max_sequence_tokens)
    else:
        if args.audit_roots:
            parser.error("--audit-roots requires --pairs-jsonl")
        result = prepare(args.pairs_dir, args.model, args.output_dir, args.max_sequence_tokens)
    print(json.dumps(result["pair_count"]))


if __name__ == "__main__":
    main()
