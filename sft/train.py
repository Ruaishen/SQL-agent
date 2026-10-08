from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sql_agent.tokenizer_check import tokenizer_fingerprint
from sft.train_config import SftTrainConfig


def load_sft_samples(dataset_dir: Path, max_sequence_tokens: int) -> list[dict[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("source_split") != "train":
        raise ValueError("SFT dataset is incomplete or is not train-only")
    samples: list[dict[str, Any]] = []
    for path_value in manifest["shards"]:
        shard = torch.load(Path(path_value), map_location="cpu", weights_only=True)
        trajectory_tokens = sum(
            int(turn["action_mask"].sum())
            for turn in shard["turns"]
            if turn["supervised"]
        )
        if trajectory_tokens < 1:
            raise ValueError(f"SFT trajectory has no supervised tokens: {path_value}")
        for turn in shard["turns"]:
            if not turn["supervised"]:
                continue
            input_ids = turn["input_ids"].to(dtype=torch.long)
            action_mask = turn["action_mask"].to(dtype=torch.bool)
            if input_ids.numel() > max_sequence_tokens:
                raise ValueError(f"SFT sequence exceeds configured limit: {path_value}")
            if action_mask.numel() != input_ids.numel() - 1 or not action_mask.any():
                raise ValueError(f"invalid supervised SFT mask: {path_value}")
            samples.append(
                {
                    "input_ids": input_ids,
                    "action_mask": action_mask,
                    "task_id": shard["task_id"],
                    "turn": turn["turn"],
                    "loss_weight": 1.0 / trajectory_tokens,
                }
            )
    expected_turns = int(manifest["statistics"]["turns"])
    expected_tokens = int(manifest["statistics"]["action_tokens"])
    if len(samples) != expected_turns:
        raise ValueError("SFT sample count differs from collection manifest")
    if sum(int(sample["action_mask"].sum()) for sample in samples) != expected_tokens:
        raise ValueError("SFT action-token count differs from collection manifest")
    return samples


def epoch_order(samples: list[dict[str, Any]], seed: int, bucket_size: int = 256):
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    result: list[dict[str, Any]] = []
    for start in range(0, len(ordered), bucket_size):
        bucket = ordered[start : start + bucket_size]
        bucket.sort(key=lambda sample: sample["input_ids"].numel())
        if (start // bucket_size) % 2:
            bucket.reverse()
        result.extend(bucket)
    return result


def collate_samples(samples: list[dict[str, Any]], pad_token_id: int, device: str):
    max_length = max(sample["input_ids"].numel() for sample in samples)
    input_ids = torch.full(
        (len(samples), max_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    action_mask = torch.zeros(
        (len(samples), max_length - 1), dtype=torch.bool, device=device
    )
    for row, sample in enumerate(samples):
        length = sample["input_ids"].numel()
        input_ids[row, :length] = sample["input_ids"].to(device)
        attention_mask[row, :length] = 1
        action_mask[row, : length - 1] = sample["action_mask"].to(device)
    return input_ids, attention_mask, action_mask


def masked_ce_sum(logits: torch.Tensor, input_ids: torch.Tensor, action_mask: torch.Tensor):
    token_losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view_as(action_mask)
    return (token_losses * action_mask).sum()


def learning_rate_at_step(config: SftTrainConfig, step: int, total_steps: int) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    decay_steps = max(1, total_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def update_batch(
    student,
    optimizer,
    samples: list[dict[str, Any]],
    pad_token_id: int,
    config: SftTrainConfig,
    micro_batch_size: int,
) -> dict[str, Any]:
    total_tokens = sum(int(sample["action_mask"].sum()) for sample in samples)
    total_weight = sum(
        int(sample["action_mask"].sum()) * float(sample.get("loss_weight", 1.0))
        for sample in samples
    )
    optimizer.zero_grad(set_to_none=True)
    loss_sum_value = 0.0
    for start in range(0, len(samples), micro_batch_size):
        chunk = samples[start : start + micro_batch_size]
        input_ids, attention_mask, action_mask = collate_samples(chunk, pad_token_id, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
            token_losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                input_ids[:, 1:].reshape(-1),
                reduction="none",
            ).view_as(action_mask)
            row_weights = torch.tensor(
                [sample.get("loss_weight", 1.0) for sample in chunk],
                device="cuda",
            ).unsqueeze(1)
            loss_sum = (token_losses * action_mask * row_weights).sum()
            loss = loss_sum / total_weight
        loss.backward()
        loss_sum_value += float(loss_sum.detach().cpu())
        del (
            input_ids,
            attention_mask,
            action_mask,
            logits,
            token_losses,
            row_weights,
            loss_sum,
            loss,
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("non-finite full-parameter SFT gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    grad_value = float(grad_norm.detach().cpu())
    return {
        "loss": loss_sum_value / total_weight,
        "loss_normalization": "trajectory_equal",
        "action_tokens": total_tokens,
        "samples": len(samples),
        "grad_norm_before_clip": grad_value,
        "gradient_clip_norm": config.gradient_clip_norm,
        "gradient_clip_scale": min(1.0, config.gradient_clip_norm / (grad_value + 1e-12)),
        "micro_batch_size": micro_batch_size,
    }


def update_with_oom_backoff(student, optimizer, samples, pad_token_id, config, micro_batch):
    attempted = min(micro_batch, len(samples))
    while attempted >= 1:
        try:
            return update_batch(
                student, optimizer, samples, pad_token_id, config, attempted
            )
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if attempted == 1:
                raise
            attempted = max(1, attempted // 2)
    raise AssertionError("unreachable")


def micro_batch_for_samples(samples, config: SftTrainConfig, previous: int) -> int:
    if not config.micro_batch_max_tokens:
        return previous
    longest = max(sample["input_ids"].numel() for sample in samples)
    return min(config.micro_batch_size, max(1, config.micro_batch_max_tokens // longest))


def train(
    config: SftTrainConfig, *, max_steps: int | None, save: bool,
    epoch_callback: Callable[[int, Path, Any, Any], None] | None = None,
) -> dict[str, Any]:
    config.validate()
    manifest = json.loads((config.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if tokenizer_fingerprint(config.student_model) != manifest["tokenizer_sha256"]:
        raise ValueError("Student tokenizer differs from tokenized SFT dataset")
    samples = load_sft_samples(config.dataset_dir, config.max_sequence_tokens)
    steps_per_epoch = math.ceil(len(samples) / config.effective_batch_size)
    configured_steps = steps_per_epoch * config.epochs
    total_steps = configured_steps if max_steps is None else min(max_steps, configured_steps)
    if total_steps < 1:
        raise ValueError("SFT training requires at least one step")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to append to existing SFT run: {metrics_path}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.student_model)
    student = AutoModelForCausalLM.from_pretrained(
        config.student_model, dtype=torch.float32, device_map={"": "cuda:0"}
    )
    student.requires_grad_(True)
    student.train()
    student.config.use_cache = False
    if config.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=True,
    )
    started = time.monotonic()
    step = 0
    last_metric: dict[str, Any] | None = None
    stop = False
    current_micro_batch = config.micro_batch_size
    for epoch in range(1, config.epochs + 1):
        ordered = epoch_order(samples, config.seed + epoch)
        for start in range(0, len(ordered), config.effective_batch_size):
            if step >= total_steps:
                stop = True
                break
            step += 1
            step_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            lr = learning_rate_at_step(config, step, configured_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = ordered[start : start + config.effective_batch_size]
            requested_micro_batch = micro_batch_for_samples(batch, config, current_micro_batch)
            update = update_with_oom_backoff(
                student,
                optimizer,
                batch,
                tokenizer.pad_token_id,
                config,
                requested_micro_batch,
            )
            current_micro_batch = int(update["micro_batch_size"])
            elapsed = time.monotonic() - started
            eta = elapsed / step * (total_steps - step)
            last_metric = {
                "step": step,
                "total_steps": total_steps,
                "configured_steps": configured_steps,
                "epoch": epoch,
                "progress_percent": 100.0 * step / total_steps,
                "learning_rate": lr,
                "step_seconds": time.monotonic() - step_started,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "estimated_finish_utc": (
                    datetime.now(UTC) + timedelta(seconds=eta)
                ).isoformat(timespec="seconds"),
                "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "requested_micro_batch_size": requested_micro_batch,
                **update,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_metric, sort_keys=True) + "\n")
            print(json.dumps(last_metric, sort_keys=True), flush=True)
        if stop:
            break
        if save and config.save_every_epoch:
            epoch_checkpoint = config.output_dir / f"checkpoint_epoch_{epoch}"
            student.save_pretrained(
                epoch_checkpoint, safe_serialization=True, max_shard_size="2GB"
            )
            tokenizer.save_pretrained(epoch_checkpoint)
            print(f"Saved {epoch_checkpoint}", flush=True)
            if epoch_callback is not None:
                epoch_callback(epoch, epoch_checkpoint, student, optimizer)

    checkpoint = None
    if save:
        checkpoint = config.output_dir / (
            f"checkpoint_epoch_{epoch}" if config.save_every_epoch else "checkpoint_final"
        )
        if not config.save_every_epoch:
            student.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
            tokenizer.save_pretrained(checkpoint)
    if last_metric is None:
        raise AssertionError("SFT produced no metrics")
    summary = {
        "status": "completed",
        "student_initialization": "base_model",
        "student_update": "full",
        "dataset_manifest": str(config.dataset_dir / "manifest.json"),
        "dataset_samples": len(samples),
        "dataset_action_tokens": sum(int(x["action_mask"].sum()) for x in samples),
        "completed_steps": step,
        "configured_steps": configured_steps,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "config": config.to_dict(),
        "final_metric": last_metric,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-parameter action-masked SFT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = SftTrainConfig.load(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    result = train(config, max_steps=args.max_steps, save=not args.no_save)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
