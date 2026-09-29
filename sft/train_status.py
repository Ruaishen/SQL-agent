from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sft.status import _duration
from sft.train_config import SftTrainConfig


def read_status(config: SftTrainConfig) -> str:
    metrics_path = config.output_dir / "metrics.jsonl"
    summary_path = config.output_dir / "run_summary.json"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return "waiting | SFT metrics not written yet"
    metric = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])
    state = "completed" if summary_path.exists() else "running"
    return (
        f"{state} | step {metric['step']}/{metric['total_steps']} "
        f"({metric['progress_percent']:.1f}%) | epoch {metric['epoch']} | "
        f"elapsed {_duration(metric['elapsed_seconds'])} | ETA {_duration(metric['eta_seconds'])} "
        f"| finish {metric['estimated_finish_utc']} | loss {metric['loss']:.4f} | "
        f"grad {metric['grad_norm_before_clip']:.2f} | micro {metric['micro_batch_size']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show SFT training progress and ETA")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--watch", type=float, default=0.0, metavar="SECONDS")
    args = parser.parse_args()
    config = SftTrainConfig.load(args.config)
    while True:
        print(read_status(config), flush=True)
        if args.watch <= 0 or (config.output_dir / "run_summary.json").exists():
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
