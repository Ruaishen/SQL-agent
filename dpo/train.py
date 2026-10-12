"""DPO with cached frozen SFT reference scores and full assistant trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_identity(path):
    # Cached scores are valid only while this local checkpoint is immutable.
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Empty checkpoint")
    return {"path": str(path.resolve()), "files": [
        [p.relative_to(path).as_posix(), p.stat().st_size, p.stat().st_mtime_ns] for p in files]}


def dataset_manifest(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("score_mode") != "sum_all_assistant_tokens" or not manifest.get("shards") or manifest["pair_count"] != len(manifest["shards"]):
        raise ValueError("Incomplete or incompatible DPO dataset")
    for name, digest in manifest["shards"].items():
        path = (dataset / name).resolve()
        if not path.is_relative_to(dataset.resolve()) or sha256(path) != digest:
            raise ValueError(f"Shard changed: {name}")
    return manifest


def load_model(checkpoint, device, gradient_checkpointing=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(checkpoint)
    for name in ("attention_dropout", "hidden_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob", "resid_pdrop", "embd_pdrop", "attn_pdrop"):
        if hasattr(config, name):
            setattr(config, name, 0.0)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, config=config, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model.to(device)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def turn_logp(model, turn, device):
    import torch
    import torch.nn.functional as F
    ids = turn["input_ids"].to(device)
    mask = turn["action_mask"].to(device)
    if ids.ndim != 1 or mask.numel() != ids.numel() - 1 or mask.dtype != torch.bool or not mask.any():
        raise ValueError("Invalid assistant mask")
    logits = model(input_ids=ids.unsqueeze(0)).logits[0, :-1][mask].float()
    return -F.cross_entropy(logits, ids[1:][mask], reduction="sum")


def branch_score(model, turns, device):
    import torch
    with torch.no_grad():
        return sum(float(turn_logp(model, turn, device)) for turn in turns)


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    import torch.nn.functional as F
    return F.softplus(-beta * (chosen - rejected - reference_chosen + reference_rejected))


def reference(args):
    import torch
    from sql_agent.tokenizer_check import tokenizer_fingerprint
    manifest = dataset_manifest(args.dataset)
    if tokenizer_fingerprint(args.checkpoint) != manifest["tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizer differs from prepared data")
    if args.reference_dir.exists():
        raise FileExistsError("Use a new reference directory")
    identity = {"checkpoint": checkpoint_identity(args.checkpoint), "dataset_sha256": sha256(args.dataset / "manifest.json")}
    model = load_model(args.checkpoint, args.device)
    model.eval()
    model.requires_grad_(False)
    args.reference_dir.mkdir(parents=True)
    scores = {}
    for name in manifest["shards"]:
        pair = torch.load(args.dataset / name, map_location="cpu", weights_only=True)
        scores[name] = {"task_id": pair["task_id"], "chosen": branch_score(model, pair["chosen"], args.device), "rejected": branch_score(model, pair["rejected"], args.device)}
    (args.reference_dir / "scores.json").write_text(json.dumps({"status": "completed", **identity, "scores": scores}), encoding="utf-8")
    print(json.dumps({"reference_pairs": len(scores)}))


def train(args):
    import math
    import torch
    from sql_agent.tokenizer_check import tokenizer_fingerprint
    manifest = dataset_manifest(args.dataset)
    ref_path = args.reference_dir / "scores.json"
    reference_data = json.loads(ref_path.read_text(encoding="utf-8"))
    if reference_data.get("status") != "completed" or reference_data.get("checkpoint") != checkpoint_identity(args.checkpoint) or reference_data.get("dataset_sha256") != sha256(args.dataset / "manifest.json") or set(reference_data["scores"]) != set(manifest["shards"]):
        raise ValueError("Cached reference differs from checkpoint/dataset")
    if tokenizer_fingerprint(args.checkpoint) != manifest["tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizer differs from prepared data")
    if args.output.exists():
        raise FileExistsError("Use a new training output directory")
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, args.device, args.gradient_checkpointing)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True)
    names = list(manifest["shards"])
    step = 0
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(names)
        for start in range(0, len(names), args.batch_size):
            batch = names[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for name in batch:
                pair = torch.load(args.dataset / name, map_location="cpu", weights_only=True)
                ref = reference_data["scores"][name]
                if ref["task_id"] != pair["task_id"]:
                    raise ValueError("Reference task mismatch")
                win = branch_score(model, pair["chosen"], args.device)
                lose = branch_score(model, pair["rejected"], args.device)
                margin = win - lose - ref["chosen"] + ref["rejected"]
                if not math.isfinite(margin):
                    raise ValueError("Nonfinite DPO margin")
                # Exact DPO gradient; recompute one turn at a time to save memory.
                coefficient = args.beta * torch.sigmoid(torch.tensor(-args.beta * margin, dtype=torch.float64)).item()
                for sign, key in ((-1, "chosen"), (1, "rejected")):
                    for turn in pair[key]:
                        (sign * coefficient * turn_logp(model, turn, args.device) / len(batch)).backward()
                losses.append(float(dpo_loss(torch.tensor(win, dtype=torch.float64), torch.tensor(lose, dtype=torch.float64), ref["chosen"], ref["rejected"], args.beta)))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            metrics = {"epoch": epoch + 1, "step": step, "loss": sum(losses) / len(losses), "grad_norm": float(norm)}
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
    from transformers import AutoTokenizer
    model.save_pretrained(args.output / "checkpoint")
    AutoTokenizer.from_pretrained(args.checkpoint).save_pretrained(args.output / "checkpoint")
    (args.output / "manifest.json").write_text(json.dumps({"status": "completed", "steps": step, "pairs": len(names), "beta": args.beta, "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "seed": args.seed, "checkpoint_identity": reference_data["checkpoint"], "dataset_sha256": reference_data["dataset_sha256"], "reference_sha256": sha256(ref_path)}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "train"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if min(args.beta, args.learning_rate, args.max_grad_norm) <= 0 or min(args.epochs, args.batch_size) < 1:
        parser.error("DPO parameters must be positive")
    if args.mode == "train" and args.output is None:
        parser.error("Training requires --output")
    (reference if args.mode == "reference" else train)(args)


if __name__ == "__main__":
    main()
