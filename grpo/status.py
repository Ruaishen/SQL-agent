from __future__ import annotations

import argparse
import json
from pathlib import Path

from grpo.config import GrpoConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Show bounded GRPO progress")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = GrpoConfig.load(args.config)
    metrics_path = config.output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        print("prepared | no training metrics")
        return
    with metrics_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    latest = records[-1]
    print(
        f"step {latest['step']}/{latest['total_steps']} "
        f"({latest['progress_percent']:.1f}%) | reward {latest['mean_reward']:.3f} | "
        f"mixed {latest['mixed_group_ratio']:.1%} | all-zero "
        f"{latest['all_zero_group_ratio']:.1%} | ETA {latest['eta_seconds']:.0f}s"
    )


if __name__ == "__main__":
    main()
