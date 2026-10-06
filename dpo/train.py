"""Full-parameter DPO with reference scores cached before policy training.

Run ``score-reference`` once, then ``train``. The reference and policy use the
same SFT checkpoint; only masked assistant tokens from the fork onward count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sql_agent.tokenizer_check import tokenizer_fingerprint
from dpo.dataset import sha256


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("sft_checkpoint", "dataset_dir", "output_dir", "epochs",
                "effective_batch_size", "learning_rate", "beta"):
        if key not in config:
            raise ValueError(f"Missing DPO config: {key}")
    if config["epochs"] < 1 or config["effective_batch_size"] < 1:
        raise ValueError("Invalid training duration or batch size")
    if config["learning_rate"] <= 0 or config["beta"] <= 0:
        raise ValueError("Learning rate and beta must be positive")
    return config


def load_dataset(config: dict) -> tuple[dict, list[Path]]:
    dataset = Path(config["dataset_dir"])
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("source_split") != "train":
        raise ValueError("DPO dataset must be complete and train-only")
    if manifest["tokenizer_sha256"] != tokenizer_fingerprint(Path(config["sft_checkpoint"])):
        raise ValueError("DPO tokenizer does not match the SFT checkpoint")
    paths = [Path(value) if Path(value).is_absolute() else dataset / value
             for value in manifest["shards"]]
    if len(paths) != manifest["pair_count"] or not paths or any(not p.is_file() for p in paths):
        raise ValueError("Missing or inconsistent DPO shards")
    if "shard_sha256" in manifest:
        if set(manifest["shard_sha256"]) != set(manifest["shards"]):
            raise ValueError("Incomplete shard integrity manifest")
        if any(sha256(path) != manifest["shard_sha256"][key]
               for key, path in zip(manifest["shards"], paths)):
            raise ValueError("DPO shard changed after preparation")
    return manifest, paths


def branch_logp(model, branch: dict, *, gradient: bool) -> torch.Tensor:
    ids = branch["input_ids"].to("cuda")
    mask = branch["loss_mask"].to("cuda")
    if mask.numel() != ids.numel() - 1 or not mask.any():
        raise ValueError("Invalid DPO branch mask")
    with torch.set_grad_enabled(gradient), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids.unsqueeze(0)).logits[0, :-1]
        selected = logits[mask].float()
        targets = ids[1:][mask]
        result = (selected.gather(-1, targets[:, None]).squeeze(-1)
                  - torch.logsumexp(selected, dim=-1)).sum()
    return result


def _load_model(path: str, *, train: bool, gradient_checkpointing: bool):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                low_cpu_mem_usage=True).to("cuda")
    model.config.use_cache = False
    # Separate margin and gradient forwards must score the same deterministic policy.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for name in ("attention_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob"):
        if hasattr(model.config, name):
            setattr(model.config, name, 0.0)
    for module in model.modules():
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    if train:
        model.train()
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
    else:
        model.eval()
        model.requires_grad_(False)
    return model


def score_reference(config: dict) -> dict:
    dataset_manifest, paths = load_dataset(config)
    output = Path(config["output_dir"])
    score_dir = output / "reference_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    identity = {"sft_checkpoint": str(Path(config["sft_checkpoint"]).resolve()),
                "dataset_manifest_sha256": hashlib.sha256(
                    (Path(config["dataset_dir"]) / "manifest.json").read_bytes()).hexdigest(),
                "pair_count": dataset_manifest["pair_count"]}
    identity_path = score_dir / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Reference score cache belongs to a different model or dataset")
    else:
        if any(score_dir.glob("*.json")):
            raise ValueError("Reference scores exist without an identity file")
        identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    model = _load_model(config["sft_checkpoint"], train=False,
                        gradient_checkpointing=False)
    created = skipped = 0
    try:
        for path in paths:
            dest = score_dir / (path.stem + ".json")
            if dest.exists():
                skipped += 1
                continue
            pair = torch.load(path, map_location="cpu", weights_only=True)
            chosen = float(branch_logp(model, pair["chosen"], gradient=False).cpu())
            rejected = float(branch_logp(model, pair["rejected"], gradient=False).cpu())
            temporary = dest.with_suffix(".tmp")
            temporary.write_text(json.dumps({"task_id": pair["task_id"],
                                             "chosen_logp": chosen,
                                             "rejected_logp": rejected}) + "\n",
                                 encoding="utf-8")
            temporary.replace(dest)
            created += 1
    finally:
        del model
        torch.cuda.empty_cache()
    return {"created": created, "skipped": skipped, "total": len(paths)}


def train(config: dict) -> dict:
    dataset_manifest, paths = load_dataset(config)
    output = Path(config["output_dir"])
    score_dir = output / "reference_scores"
    identity_path = score_dir / "identity.json"
    expected_identity = {"sft_checkpoint": str(Path(config["sft_checkpoint"]).resolve()),
                         "dataset_manifest_sha256": hashlib.sha256(
                             (Path(config["dataset_dir"]) / "manifest.json").read_bytes()).hexdigest(),
                         "pair_count": dataset_manifest["pair_count"]}
    if (not identity_path.is_file()
            or json.loads(identity_path.read_text(encoding="utf-8")) != expected_identity):
        raise ValueError("Reference scores do not match this model and dataset")
    if any(not (score_dir / (p.stem + ".json")).is_file() for p in paths):
        raise ValueError("Run score-reference before DPO training")
    if (output / "train_manifest.json").exists():
        raise FileExistsError("DPO training already completed in this output directory")
    torch.manual_seed(int(config.get("seed", 42)))
    model = _load_model(config["sft_checkpoint"], train=True,
                        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                   betas=(0.9, 0.95), weight_decay=0)
    beta = float(config["beta"])
    batch_size = int(config["effective_batch_size"])
    epochs = int(config["epochs"])
    max_norm = float(config.get("gradient_clip_norm", 1.0))
    if max_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    rng = random.Random(int(config.get("seed", 42)))
    metrics_path = output / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError("Existing DPO metrics; choose a fresh output directory")
    steps = 0
    for epoch in range(1, epochs + 1):
        ordered = list(paths)
        rng.shuffle(ordered)
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            margins = []
            for path in chunk:
                pair = torch.load(path, map_location="cpu", weights_only=True)
                ref = json.loads((score_dir / (path.stem + ".json")).read_text(encoding="utf-8"))
                if ref["task_id"] != pair["task_id"]:
                    raise ValueError(f"Reference score task mismatch: {path}")
                reference_margin = ref["chosen_logp"] - ref["rejected_logp"]
                # Evaluate the margin without a graph, then backpropagate each
                # branch separately. This avoids retaining two 16k-token graphs.
                win_value = branch_logp(model, pair["chosen"], gradient=False)
                lose_value = branch_logp(model, pair["rejected"], gradient=False)
                margin_value = float((win_value - lose_value).cpu()) - reference_margin
                coefficient = beta * torch.sigmoid(torch.tensor(-beta * margin_value)).item()
                win = branch_logp(model, pair["chosen"], gradient=True)
                (-coefficient * win / len(chunk)).backward()
                lose = branch_logp(model, pair["rejected"], gradient=True)
                (coefficient * lose / len(chunk)).backward()
                losses.append(float(F.softplus(torch.tensor(-beta * margin_value))))
                margins.append(margin_value)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite DPO gradient")
            optimizer.step()
            steps += 1
            metric = {"epoch": epoch, "step": steps, "pairs": len(chunk),
                      "loss": sum(losses) / len(chunk),
                      "mean_logp_margin_over_reference": sum(margins) / len(chunk),
                      "grad_norm": float(grad_norm.detach().cpu())}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric) + "\n")
            if steps % 10 == 0:
                print(json.dumps(metric), flush=True)
        checkpoint = output / f"checkpoint_epoch_{epoch}"
        checkpoint.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
        AutoTokenizer.from_pretrained(config["sft_checkpoint"]).save_pretrained(checkpoint)
    manifest = {"status": "completed", "base_checkpoint": config["sft_checkpoint"],
                "dataset_dir": config["dataset_dir"], "epochs": epochs, "steps": steps,
                "pairs": len(paths), "beta": beta,
                "checkpoint": str(checkpoint.resolve())}
    (output / "train_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("score-reference", "train"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(score_reference(config) if args.mode == "score-reference" else train(config)))


if __name__ == "__main__":
    main()
