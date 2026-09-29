from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sft.config import SftCollectionConfig


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_status(config: SftCollectionConfig) -> str:
    progress_path = config.output_dir / "progress.json"
    if not progress_path.exists():
        return f"waiting | 0/{config.target_total} | progress not written yet"
    value = json.loads(progress_path.read_text(encoding="utf-8"))
    difficulty_status = ", ".join(
        f"{difficulty} {value[f'{difficulty}_successes']}/"
        f"{value[f'target_{difficulty}']}"
        for difficulty in config.difficulties
    )
    return (
        f"{value['status']} | {value['successes']}/{value['target_total']} "
        f"({difficulty_status}) | "
        f"attempts {value['attempts']} | elapsed {_duration(value['elapsed_seconds'])} | "
        f"ETA {_duration(value['eta_seconds'])} | finish {value['estimated_finish_utc']} | "
        f"GPU {value['gpu_allocated_gib']:.1f}/{value['gpu_reserved_gib']:.1f} GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Teacher SFT collection progress")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--watch", type=float, default=0.0, metavar="SECONDS")
    args = parser.parse_args()
    config = SftCollectionConfig.load(args.config)
    while True:
        print(read_status(config), flush=True)
        if args.watch <= 0 or (config.output_dir / "manifest.json").exists():
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
