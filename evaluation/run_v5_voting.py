from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer

from evaluation.run_tool_xml import write_json
from evaluation.v5_voting import V5VotingEvaluator, vote_sql
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.model_adapter import OpenAICompatibleAdapter
from sql_agent.models import TaskRecord
from sql_agent.verifier import ExecutionVerifier

SOURCE_DIR = Path(
    "artifacts/sql_planner/qwen25_coder_3b_base_direct_seeded_evidence_columns_v5_eval"
)


def vote_task(
    task: TaskRecord, sample_dir: Path, output_dir: Path, samples: int, config: EnvConfig
) -> dict[str, Any]:
    candidates = [
        json.loads((sample_dir / f"{task.task_id}_{index:02d}.json").read_text())
        for index in range(samples)
    ]
    voted = vote_sql(task, candidates, config)
    final_sql = voted["final_sql"]
    verification = (
        ExecutionVerifier(task.resolve_db_path(config.spider_root), task.reference_sql, config)
        .verify(final_sql)
        .to_dict()
        if final_sql is not None
        else None
    )
    record = {
        "task_id": task.task_id,
        "db_id": task.db_id,
        "difficulty": task.difficulty,
        "status": "voted" if final_sql is not None else "all_invalid",
        "correct": verification["correct"] if verification else False,
        "final_sql": final_sql,
        "verification": verification,
        "selected_sample": voted["selected_sample"],
        "votes": voted["votes"],
        "valid_candidates": voted["valid"],
        "candidate_sqls": [candidate["final_sql"] for candidate in candidates],
        "candidate_correct": [candidate["correct"] for candidate in candidates],
        "candidate_statuses": [candidate["status"] for candidate in candidates],
    }
    write_json(output_dir / f"{task.task_id}.json", record)
    return record


def summarize(records: list[dict[str, Any]], total: int, samples: int) -> dict[str, Any]:
    by_difficulty = {}
    for difficulty in ("easy", "medium", "hard", "extra"):
        group = [record for record in records if record["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "count": len(group),
            "correct": sum(record["correct"] for record in group),
        }
    return {
        "total": total,
        "completed": len(records),
        "complete": len(records) == total,
        "correct": sum(record["correct"] for record in records),
        "execution_accuracy": sum(record["correct"] for record in records) / len(records)
        if records
        else 0.0,
        "samples_per_task": samples,
        "first_sample_correct": sum(record["candidate_correct"][0] for record in records),
        "any_sample_correct_oracle": sum(any(record["candidate_correct"]) for record in records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "by_difficulty": by_difficulty,
    }


def main(
    endpoint: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    task_file: Annotated[Path, typer.Option(exists=True)] = Path("data/external_dev.jsonl"),
    env_config: Annotated[Path, typer.Option(exists=True)] = Path(
        "configs/env_sql_planner_qwen25_coder_3b.yaml"
    ),
    spider_root: Annotated[Path | None, typer.Option()] = None,
    source_dir: Annotated[Path, typer.Option(exists=True)] = SOURCE_DIR,
    output_dir: Annotated[Path, typer.Option()] = Path(
        "artifacts/sql_planner/qwen25_coder_3b_base_v5_eight_vote_eval"
    ),
    samples: Annotated[int, typer.Option(min=2)] = 8,
    workers: Annotated[int, typer.Option(min=1)] = 16,
    limit: Annotated[int, typer.Option(min=0)] = 0,
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
    source_dir = source_dir.resolve()
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if source_manifest["protocol"] != "tool_xml_direct_aligned_v5_evidence_columns":
        raise typer.BadParameter("Source run is not the retained v5 evaluation")
    dataset_hash = hashlib.sha256(task_file.read_bytes()).hexdigest()
    if source_manifest["dataset_sha256"] != dataset_hash:
        raise typer.BadParameter("Source evaluation uses a different task file")
    prompt = (source_dir / "prompt_template.txt").read_text().rstrip("\n")
    if hashlib.sha256(prompt.encode()).hexdigest() != source_manifest["prompt_sha256"]:
        raise typer.BadParameter("Source prompt does not match its manifest")
    source_records = {}
    for task in tasks:
        record = json.loads((source_dir / "trajectories" / f"{task.task_id}.json").read_text())
        if record["task_id"] != task.task_id or not record["initial_sql"]:
            raise typer.BadParameter(f"Invalid source trajectory: {task.task_id}")
        if record["messages"][0]["content"] != prompt:
            raise typer.BadParameter(f"Source prompt mismatch: {task.task_id}")
        source_records[task.task_id] = record
    manifest = {
        "protocol": "v5_multiturn_execution_result_vote_v1",
        "source_run": str(source_dir),
        "source_prompt_sha256": source_manifest["prompt_sha256"],
        "dataset": str(task_file),
        "dataset_sha256": dataset_hash,
        "model": model,
        "samples_per_task": samples,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 512,
        "seed_per_sample": [config.split_seed + index for index in range(samples)],
        "max_exploratory_calls": config.max_turns,
        "workers": workers,
        "spider_root": str(config.spider_root),
        "vote": (
            "largest group of equal frozenset(full execution rows); "
            "earliest sample breaks ties"
        ),
        "scorer": "ExecutionVerifier",
        "limit": limit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    trajectory_dir = output_dir / "trajectories"
    sample_dir.mkdir(exist_ok=True)
    trajectory_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise typer.BadParameter("Existing output has a different manifest")
    else:
        write_json(manifest_path, manifest)
        (output_dir / "prompt_template.txt").write_text(prompt + "\n")
    adapter = OpenAICompatibleAdapter(
        endpoint, model, api_key=os.getenv("OPENAI_API_KEY"), stop=["</tool>"]
    )
    evaluator = V5VotingEvaluator(config, adapter)
    pending = [
        (task, index)
        for task in tasks
        for index in range(samples)
        if not (sample_dir / f"{task.task_id}_{index:02d}.json").exists()
    ]
    typer.echo(f"Generating {len(pending)} of {len(tasks) * samples} candidates")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluator.evaluate,
                task,
                source_records[task.task_id]["messages"][:2],
                source_records[task.task_id]["initial_sql"],
                index,
            ): (task, index)
            for task, index in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            task, index = futures[future]
            write_json(sample_dir / f"{task.task_id}_{index:02d}.json", future.result())
            if completed % 100 == 0 or completed == len(pending):
                typer.echo(f"Candidates: {completed}/{len(pending)}")
    records = {}
    to_vote = []
    for task in tasks:
        path = trajectory_dir / f"{task.task_id}.json"
        if path.exists():
            records[task.task_id] = json.loads(path.read_text())
        else:
            to_vote.append(task)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(vote_task, task, sample_dir, trajectory_dir, samples, config): task
            for task in to_vote
        }
        for future in as_completed(futures):
            task = futures[future]
            records[task.task_id] = future.result()
            if len(records) % 25 == 0 or len(records) == len(tasks):
                summary = summarize(list(records.values()), len(tasks), samples)
                write_json(output_dir / "summary.json", summary)
                typer.echo(json.dumps(summary, ensure_ascii=False))
    if not to_vote:
        summary = summarize(list(records.values()), len(tasks), samples)
        write_json(output_dir / "summary.json", summary)
        typer.echo(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    typer.run(main)
