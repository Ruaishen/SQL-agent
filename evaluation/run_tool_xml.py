from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Any

import typer

from evaluation.schema import SCHEMA_FORMAT
from evaluation.tool_xml import PROMPT_VERSION, ToolXmlEvaluator, build_system_prompt
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.model_adapter import OpenAICompatibleAdapter
from sql_agent.models import TaskRecord


def load_initial_sql(
    directory: Path, tasks: list[TaskRecord], dataset_sha256: str
) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Baseline dataset hash does not match the evaluation dataset")
    candidates = {}
    for task in tasks:
        record = json.loads((directory / "trajectories" / f"{task.task_id}.json").read_text())
        sql = record.get("sql")
        if record.get("task_id") != task.task_id or not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"Missing or invalid baseline SQL for {task.task_id}")
        candidates[task.task_id] = sql
    return candidates, {
        "directory": str(directory.resolve()),
        "manifest": manifest,
        "sql_sha256": hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest(),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def summarize(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    correct = sum(bool(record["correct"]) for record in records)
    by_difficulty: dict[str, dict[str, int]] = {}
    for difficulty in ("easy", "medium", "hard", "extra"):
        group = [record for record in records if record["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "count": len(group),
            "correct": sum(bool(record["correct"]) for record in group),
        }
    return {
        "total": total,
        "completed": len(records),
        "complete": len(records) == total,
        "correct": correct,
        "execution_accuracy": correct / len(records) if records else 0.0,
        "by_difficulty": by_difficulty,
        "status_counts": dict(Counter(record["status"] for record in records)),
    }


def main(
    endpoint: Annotated[str, typer.Option(help="OpenAI-compatible vLLM endpoint.")],
    model: Annotated[str, typer.Option(help="Served model name.")],
    task_file: Annotated[Path, typer.Option(exists=True)] = Path("data/external_dev.jsonl"),
    env_config: Annotated[Path, typer.Option(exists=True)] = Path(
        "configs/env_sql_planner_qwen25_coder_3b.yaml"
    ),
    spider_root: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path, typer.Option()] = Path(
        "artifacts/sql_planner/qwen25_coder_3b_base_tool_xml_forced_submit_v4_eval"
    ),
    workers: Annotated[int, typer.Option(min=1)] = 8,
    max_tokens: Annotated[int, typer.Option(min=1)] = 512,
    initial_sql_dir: Annotated[
        Path | None, typer.Option(exists=True, help="Replay saved direct SQL as first tool call.")
    ] = None,
    limit: Annotated[int, typer.Option(min=0)] = 0,
    single_turn: Annotated[
        bool, typer.Option(help="One response, no replay or tool execution.")
    ] = False,
) -> None:
    config = EnvConfig.from_yaml(env_config)
    if spider_root is not None:
        config = replace(config, spider_root=spider_root.expanduser().resolve())
    task_file = task_file.resolve()
    tasks = load_tasks(task_file)
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise typer.BadParameter("Task file contains no tasks")
    if single_turn and initial_sql_dir is not None:
        raise typer.BadParameter("Single-turn mode cannot replay baseline SQL")
    prompt = build_system_prompt(0 if single_turn else config.max_turns)
    dataset_sha256 = hashlib.sha256(task_file.read_bytes()).hexdigest()
    initial_sql: dict[str, str] = {}
    baseline = None
    if initial_sql_dir is not None:
        try:
            initial_sql, baseline = load_initial_sql(initial_sql_dir, tasks, dataset_sha256)
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    manifest = {
        "protocol": PROMPT_VERSION,
        "schema_format": SCHEMA_FORMAT,
        "model": model,
        "dataset": str(task_file),
        "dataset_sha256": dataset_sha256,
        "spider_root": str(config.spider_root),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "max_exploratory_calls": 0 if single_turn else config.max_turns,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 20,
        "min_p": 0.0,
        "seed": config.split_seed,
        "enable_thinking": False,
        "workers": workers,
        "environment": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "baseline_replay": baseline,
        "stop": ["</tool>"],
        "limit": limit,
        "scorer": "ExecutionVerifier",
    }
    if single_turn:
        manifest["single_turn"] = True
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise typer.BadParameter(
                "Existing output has a different manifest; use a new output directory"
            )
    else:
        write_json(manifest_path, manifest)
        (output_dir / "prompt_template.txt").write_text(prompt + "\n")
    records: dict[str, dict[str, Any]] = {}
    pending = []
    for task in tasks:
        path = trajectory_dir / f"{task.task_id}.json"
        if path.exists():
            records[task.task_id] = json.loads(path.read_text())
        else:
            pending.append(task)
    adapter = OpenAICompatibleAdapter(
        endpoint, model, api_key=os.getenv("OPENAI_API_KEY"), stop=["</tool>"]
    )
    evaluator = ToolXmlEvaluator(config, adapter, max_tokens=max_tokens)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            (
                executor.submit(evaluator.evaluate_single, task)
                if single_turn
                else executor.submit(evaluator.evaluate, task, initial_sql.get(task.task_id))
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            task = futures[future]
            record = future.result()
            write_json(trajectory_dir / f"{task.task_id}.json", record)
            records[task.task_id] = record
            if len(records) % 25 == 0 or len(records) == len(tasks):
                summary = summarize(list(records.values()), len(tasks))
                write_json(output_dir / "summary.json", summary)
                typer.echo(json.dumps(summary, ensure_ascii=False))
    if not pending:
        summary = summarize(list(records.values()), len(tasks))
        write_json(output_dir / "summary.json", summary)
        typer.echo(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    typer.run(main)
