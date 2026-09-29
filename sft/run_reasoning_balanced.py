from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch

from sft.train import train
from sft.train_config import SftTrainConfig

SPIDER_ROOT = Path("/root/autodl-tmp/sqlagent/datasets/spider/spider_data")


def move_optimizer(optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = SftTrainConfig.load(args.config)
    if config.epochs != 2 or not config.save_every_epoch:
        raise ValueError("Balanced run requires two epochs and per-epoch checkpoints")
    def evaluate_epoch(epoch: int, checkpoint: Path, student, optimizer) -> None:
        student.to("cpu")
        move_optimizer(optimizer, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.memory_allocated()
        print(f"Epoch {epoch}: GPU allocations after CPU offload: {free} bytes", flush=True)
        output = config.output_dir / f"eval_epoch_{epoch}"
        command = [
            sys.executable, "-u", "-m", "evaluation.run_reasoning_sft",
            "--model", str(checkpoint),
            "--dataset", str(config.dataset_dir),
            "--tasks", "data/external_dev.jsonl",
            "--env-config", "configs/env_sql_planner_qwen25_coder_3b.yaml",
            "--spider-root", str(SPIDER_ROOT),
            "--output-dir", str(output),
            "--batch-size", "16",
            "--max-tokens", "512",
        ]
        subprocess.run(command, check=True)
        result = json.loads((output / "summary.json").read_text())
        if result["completed"] != result["total"]:
            raise RuntimeError(f"Incomplete epoch {epoch} evaluation: {result}")
        report = {
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "correct": result["correct"],
            "total": result["total"],
            "execution_accuracy": result["execution_accuracy"],
        }
        with (config.output_dir / "epoch_accuracy.jsonl").open("a") as handle:
            handle.write(json.dumps(report) + "\n")
        print(f"EPOCH {epoch} ACCURACY: {result['correct']}/{result['total']} = {result['execution_accuracy']:.4%}", flush=True)
        if epoch < config.epochs:
            student.to("cuda:0")
            move_optimizer(optimizer, "cuda:0")
            student.train()
            torch.cuda.empty_cache()
    train(config, max_steps=None, save=True, epoch_callback=evaluate_epoch)


if __name__ == "__main__":
    main()
