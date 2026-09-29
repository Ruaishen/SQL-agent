from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from sql_agent.config import EnvConfig
from sql_agent.models import TaskRecord
from sql_agent.sandbox import connect_readonly
from third_party.spider_eval.hardness import eval_hardness

TRAIN_FILES = ("train_spider.json", "train_others.json")
VALID_DIFFICULTIES = {"easy", "medium", "hard", "extra"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values
    )
    _atomic_write_text(path, content)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        try:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        finally:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def _validate_gold(db_path: Path, sql: str, timeout_seconds: float) -> str | None:
    connection = connect_readonly(db_path)
    deadline = time.monotonic() + timeout_seconds
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1_000)
    try:
        connection.execute(sql).fetchall()
        return None
    except sqlite3.DatabaseError as exc:
        message = str(exc).replace(str(db_path), "<database>")
        return message[:512]
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def _split_databases(db_ids: set[str], seed: int) -> dict[str, list[str]]:
    shuffled = sorted(db_ids)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * 0.70)
    validation_count = (len(shuffled) - train_count) // 2
    result = {
        "train": sorted(shuffled[:train_count]),
        "internal_validation": sorted(shuffled[train_count : train_count + validation_count]),
        "internal_holdout": sorted(shuffled[train_count + validation_count :]),
    }
    sets = [set(value) for value in result.values()]
    if set.union(*sets) != db_ids or any(
        sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)
    ):
        raise AssertionError("Database split is not a partition")
    return result


def build_dataset(config: EnvConfig, *, validate_gold: bool = True) -> dict[str, Any]:
    source_root = config.spider_root
    output_root = config.processed_data_root
    train_rows: list[dict[str, Any]] = []
    for filename in TRAIN_FILES:
        train_rows.extend(_read_json(source_root / filename))
    dev_rows: list[dict[str, Any]] = _read_json(source_root / "dev.json")
    train_db_ids = {row["db_id"] for row in train_rows}
    dev_db_ids = {row["db_id"] for row in dev_rows}
    if train_db_ids & dev_db_ids:
        raise ValueError("Spider train and dev databases overlap")
    split_manifest = _split_databases(train_db_ids, config.split_seed)
    db_to_split = {
        db_id: split_name for split_name, db_ids in split_manifest.items() for db_id in db_ids
    }
    tasks_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_tasks: list[dict[str, Any]] = []
    all_task_ids: set[str] = set()

    def consume(rows: list[dict[str, Any]], prefix: str, external: bool) -> None:
        for index, row in enumerate(rows):
            task_id = f"spider_{prefix}_{index:05d}"
            if task_id in all_task_ids:
                raise ValueError(f"Duplicate task ID: {task_id}")
            all_task_ids.add(task_id)
            db_id = row["db_id"]
            db_relative = Path("database") / db_id / f"{db_id}.sqlite"
            db_path = source_root / db_relative
            if not db_path.is_file():
                invalid_tasks.append(
                    {"task_id": task_id, "db_id": db_id, "reason": "missing_database"}
                )
                continue
            try:
                difficulty = eval_hardness(row["sql"])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                invalid_tasks.append(
                    {
                        "task_id": task_id,
                        "db_id": db_id,
                        "reason": "hardness_error",
                        "detail": str(exc)[:512],
                    }
                )
                continue
            if difficulty not in VALID_DIFFICULTIES:
                raise AssertionError(f"Unexpected difficulty: {difficulty}")
            error = (
                _validate_gold(db_path, row["query"], config.verifier_timeout_seconds)
                if validate_gold
                else None
            )
            if error:
                invalid_tasks.append(
                    {
                        "task_id": task_id,
                        "db_id": db_id,
                        "reason": "gold_execution_error",
                        "detail": error,
                    }
                )
                continue
            split = "external_dev" if external else db_to_split[db_id]
            task = TaskRecord(
                task_id=task_id,
                source="spider",
                db_id=db_id,
                db_path=db_relative.as_posix(),
                question=row["question"],
                reference_sql=row["query"],
                difficulty=difficulty,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
            )
            tasks_by_split[split].append(asdict(task))

    consume(train_rows, "train", False)
    consume(dev_rows, "dev", True)
    for split in ("train", "internal_validation", "internal_holdout", "external_dev"):
        _write_jsonl(output_root / f"{split}.jsonl", tasks_by_split[split])
    _write_jsonl(output_root / "invalid_tasks.jsonl", invalid_tasks)
    _write_json(
        output_root / "splits.json",
        {
            "seed": config.split_seed,
            "source": "Spider 1.0 train databases",
            "train": split_manifest["train"],
            "internal_validation": split_manifest["internal_validation"],
            "internal_holdout": split_manifest["internal_holdout"],
            "external_dev": sorted(dev_db_ids),
        },
    )

    database_manifest = []
    for db_id in sorted(train_db_ids | dev_db_ids):
        path = source_root / "database" / db_id / f"{db_id}.sqlite"
        database_manifest.append(
            {
                "db_id": db_id,
                "relative_path": path.relative_to(source_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(output_root / "database_manifest.json", database_manifest)

    statistics: dict[str, Any] = {
        "source_task_count": len(train_rows) + len(dev_rows),
        "valid_task_count": sum(len(values) for values in tasks_by_split.values()),
        "invalid_task_count": len(invalid_tasks),
        "splits": {},
    }
    for split, tasks in sorted(tasks_by_split.items()):
        statistics["splits"][split] = {
            "tasks": len(tasks),
            "databases": len({task["db_id"] for task in tasks}),
            "difficulty": dict(sorted(Counter(task["difficulty"] for task in tasks).items())),
        }
    _write_json(output_root / "statistics.json", statistics)
    return statistics


def main(
    config_path: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = Path("configs/env.yaml"),
    validate_gold: Annotated[
        bool, typer.Option(help="Execute every Gold SQL on its read-only database.")
    ] = True,
) -> None:
    config = EnvConfig.from_yaml(config_path)
    statistics = build_dataset(config, validate_gold=validate_gold)
    typer.echo(json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True))


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
