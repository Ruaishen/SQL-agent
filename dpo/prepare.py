"""Tokenize full branches; only assistant tokens contribute to DPO scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def tokenize_branch(tokenizer, messages, max_tokens):
    import torch
    turns = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
        rendered = tokenizer.apply_chat_template(messages[:index + 1], tokenize=False, add_generation_prompt=False)
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        ids = tokenizer(rendered, add_special_tokens=False).input_ids
        if not rendered.startswith(prefix) or ids[:len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids) or not prefix_ids:
            raise ValueError("Chat template/tokenizer does not preserve assistant prefix")
        if len(ids) > max_tokens:
            raise ValueError("Branch exceeds context limit; no silent truncation")
        mask = torch.zeros(len(ids) - 1, dtype=torch.bool)
        mask[len(prefix_ids) - 1:] = True
        turns.append({"input_ids": torch.tensor(ids, dtype=torch.long), "action_mask": mask})
    if not turns:
        raise ValueError("Branch has no assistant tokens")
    return turns


def prepare(pairs, model, output, max_tokens):
    import torch
    from transformers import AutoTokenizer
    from dpo.pairs import public_messages
    from sql_agent.tokenizer_check import tokenizer_fingerprint
    if output.exists():
        raise FileExistsError(output)
    if max_tokens < 2:
        raise ValueError("Invalid context limit")
    records = [json.loads(line) for line in pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({p["task_id"] for p in records}) != len(records):
        raise ValueError("Empty/duplicate pairs")
    tokenizer = AutoTokenizer.from_pretrained(model)
    output.mkdir(parents=True)
    shards = {}
    for index, pair in enumerate(records):
        if pair.get("split") != "train" or pair.get("leak_audit") != "accept" or pair.get("chosen_correct") is not True or pair.get("rejected_correct") is not False:
            raise ValueError("Pair is not a verified/audited training pair")
        branches = {key: public_messages({"messages": pair[key + "_messages"]}) for key in ("chosen", "rejected")}
        if branches["chosen"][:2] != branches["rejected"][:2] or branches["chosen"][:2] != pair["prompt_messages"] or branches["chosen"] == branches["rejected"]:
            raise ValueError("Pair prompt or branches changed")
        shard = {"task_id": pair["task_id"], **{key: tokenize_branch(tokenizer, messages, max_tokens) for key, messages in branches.items()}}
        path = output / f"{index:05d}.pt"
        torch.save(shard, path)
        shards[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"status": "completed", "pair_count": len(records), "shards": shards,
                "pairs_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest(),
                "tokenizer_sha256": tokenizer_fingerprint(model), "model": str(model.resolve()),
                "max_sequence_tokens": max_tokens, "score_mode": "sum_all_assistant_tokens"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()
    print(json.dumps(prepare(args.pairs, args.model, args.output, args.max_tokens)))


if __name__ == "__main__":
    main()
