from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DIFFICULTIES = ("easy", "medium", "hard", "extra")
BASE_TARGETS = {"easy": 400, "medium": 800, "hard": 400, "extra": 400}
EXTRA_TARGETS = {"easy": 100, "medium": 200, "hard": 100, "extra": 100}
FINAL_TARGETS = {key: BASE_TARGETS[key] + EXTRA_TARGETS[key] for key in DIFFICULTIES}


def select(source: Path, base_selection: Path, data_dir: Path, output: Path, seed: int = 43) -> dict:
    if output.exists():
        raise FileExistsError(output)
    root = source / "trajectories"
    base = json.loads(base_selection.read_text())
    original_base = []
    for name in base["selected_files"]:
        record = json.loads((root / name).read_text())
        if record["merged_origin"] == "original":
            original_base.append(name)
    if len(original_base) != 2000:
        raise ValueError("Expected exactly 2,000 original correct trajectories in the base selection")
    task_info = {}
    for split in ("train", "internal_holdout"):
        for line in (data_dir / f"{split}.jsonl").read_text().splitlines():
            task = json.loads(line)
            task_info[task["task_id"]] = (split, task["difficulty"])
    base_counts = Counter(json.loads((root / name).read_text())["difficulty"] for name in original_base)
    if dict(base_counts) != BASE_TARGETS:
        raise ValueError(f"Unexpected base difficulty distribution: {base_counts}")
    used = set(original_base)
    pools = {(split, diff): [] for split in ("train", "internal_holdout") for diff in DIFFICULTIES}
    for path in sorted(root.glob("*.json")):
        if path.name in used:
            continue
        record = json.loads(path.read_text())
        if record.get("merged_origin") != "original" or record.get("correct") is not True:
            continue
        info = task_info.get(record["task_id"])
        if info is None:
            continue
        split, difficulty = info
        if record.get("difficulty") != difficulty or not record.get("trainable_turn_numbers"):
            raise ValueError(f"Invalid original trajectory: {path}")
        pools[split, difficulty].append(path.name)
    available = {
        split: {diff: len(pools[split, diff]) for diff in DIFFICULTIES}
        for split in ("train", "internal_holdout")
    }
    rng = random.Random(seed)
    extras = []
    for diff, target in EXTRA_TARGETS.items():
        take_train = min(target, len(pools["train", diff]))
        extras.extend(rng.sample(pools["train", diff], take_train))
        remainder = target - take_train
        if len(pools["internal_holdout", diff]) < remainder:
            raise ValueError(f"Not enough {diff} trajectories to fill {remainder} slots")
        if remainder:
            extras.extend(rng.sample(pools["internal_holdout", diff], remainder))
    selected = original_base + extras
    rng.shuffle(selected)
    ids = [json.loads((root / name).read_text())["task_id"] for name in selected]
    if len(ids) != 2500 or len(ids) != len(set(ids)):
        raise ValueError("Selection contains duplicate tasks or the wrong count")
    external = [json.loads(line) for line in (data_dir / "external_dev.jsonl").read_text().splitlines()]
    external_ids = {task["task_id"] for task in external}
    external_dbs = {task["db_id"] for task in external}
    selected_dbs = {json.loads((root / name).read_text())["db_id"] for name in selected}
    if set(ids) & external_ids or selected_dbs & external_dbs:
        raise ValueError("Selected training tasks overlap external_dev")
    counts = Counter(task_info[task_id][0] for task_id in ids)
    result = {
        "seed": seed,
        "source": str(source.resolve()),
        "base_selection": str(base_selection.resolve()),
        "official_source": "Spider train_spider.json",
        "local_partitions": dict(counts),
        "evaluation_partition": "external_dev",
        "base_correct_trajectories": 2000,
        "additional_correct_trajectories": 500,
        "gold_repair_trajectories": 0,
        "available_additional_by_partition_and_difficulty": available,
        "selected_by_difficulty": FINAL_TARGETS,
        "selected_files": selected,
        "selected_task_ids": ids,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    result = select(args.source, args.base_selection, args.data_dir, args.output, args.seed)
    print(json.dumps({
        "selected": result["selected_by_difficulty"],
        "partitions": result["local_partitions"],
        "external_dev_overlap": 0,
    }))


if __name__ == "__main__":
    main()
