from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DIFFICULTIES = ("easy", "medium", "hard", "extra")
TARGETS = {
    "original": dict(zip(DIFFICULTIES, (400, 800, 400, 400), strict=True)),
    "gold_repair": dict(zip(DIFFICULTIES, (100, 200, 100, 100), strict=True)),
}
ALLOWED_SPLITS = {
    "original": {"train"},
    "gold_repair": {"train", "internal_holdout"},
}


def select(source: Path, data_dir: Path, output: Path, seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(output)
    task_info = {}
    for split in ("train", "internal_holdout"):
        for line in (data_dir / f"{split}.jsonl").read_text().splitlines():
            task = json.loads(line)
            if task["task_id"] in task_info:
                raise ValueError(f"Task appears in multiple splits: {task['task_id']}")
            task_info[task["task_id"]] = (split, task["difficulty"])
    pools = {(origin, difficulty): [] for origin in TARGETS for difficulty in DIFFICULTIES}
    for path in sorted((source / "trajectories").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        origin = record.get("merged_origin")
        if origin not in TARGETS:
            continue
        task_id = record["task_id"]
        if task_id not in task_info:
            continue
        split, difficulty = task_info[task_id]
        if split not in ALLOWED_SPLITS[origin]:
            continue
        if record.get("correct") is not True or not record.get("trainable_turn_numbers"):
            continue
        if record.get("difficulty") not in (None, difficulty):
            raise ValueError(f"Difficulty mismatch: {path}")
        pools[origin, difficulty].append((path.name, task_id, split))
    available = {
        origin: {difficulty: len(pools[origin, difficulty]) for difficulty in DIFFICULTIES}
        for origin in TARGETS
    }
    shortages = {
        f"{origin}/{difficulty}": target - available[origin][difficulty]
        for origin, targets in TARGETS.items()
        for difficulty, target in targets.items()
        if available[origin][difficulty] < target
    }
    if shortages:
        raise ValueError(f"Insufficient trajectories: {shortages}")
    rng = random.Random(seed)
    selected = []
    for origin, targets in TARGETS.items():
        for difficulty, target in targets.items():
            train_pool = [item for item in pools[origin, difficulty] if item[2] == "train"]
            holdout_pool = [item for item in pools[origin, difficulty] if item[2] == "internal_holdout"]
            from_train = min(target, len(train_pool))
            selected.extend(rng.sample(train_pool, from_train))
            if from_train < target:
                selected.extend(rng.sample(holdout_pool, target - from_train))
    rng.shuffle(selected)
    ids = [item[1] for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected trajectories contain duplicate tasks")
    splits = Counter(item[2] for item in selected)
    result = {
        "seed": seed,
        "source": str(source.resolve()),
        "official_source": "Spider train_spider.json",
        "local_partitions": dict(splits),
        "evaluation_partition": "internal_validation",
        "available_by_origin_and_difficulty": available,
        "selected_by_origin_and_difficulty": TARGETS,
        "selected_files": [item[0] for item in selected],
        "selected_task_ids": ids,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = select(args.source, args.data_dir, args.output, args.seed)
    print(json.dumps({
        "available": result["available_by_origin_and_difficulty"],
        "selected": result["selected_by_origin_and_difficulty"],
        "partitions": result["local_partitions"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
