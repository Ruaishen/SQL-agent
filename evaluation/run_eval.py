from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Annotated

import typer

from evaluation.runner import BaselineMode, EvaluationRunner, RunnerConfig
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.model_adapter import OpenAICompatibleAdapter
from sql_agent.prompts import PROMPT_VERSION
from sql_agent.truncation import canonical_json


def select_tasks(tasks, *, limit: int, per_difficulty: int, seed: int):
    if limit and per_difficulty:
        raise ValueError("--limit and --per-difficulty cannot be used together")
    if per_difficulty:
        rng = random.Random(seed)
        selected = []
        for difficulty in ("easy", "medium", "hard", "extra"):
            candidates = [task for task in tasks if task.difficulty == difficulty]
            if len(candidates) < per_difficulty:
                raise ValueError(
                    f"Split has only {len(candidates)} {difficulty} tasks; "
                    f"requested {per_difficulty}"
                )
            selected.extend(rng.sample(candidates, per_difficulty))
        return selected
    return tasks[:limit] if limit else tasks


def main(
    endpoint: Annotated[str, typer.Option(help="vLLM/SGLang OpenAI-compatible endpoint.")],
    model: Annotated[str, typer.Option(help="Served Qwen model name.")],
    split: Annotated[str, typer.Option()] = "internal_validation",
    task_file: Annotated[
        Path | None, typer.Option(exists=True, help="Evaluate this JSONL instead of a split.")
    ] = None,
    modes: Annotated[str, typer.Option(help="Comma-separated baseline modes.")] = ",".join(
        mode.value for mode in BaselineMode
    ),
    limit: Annotated[int, typer.Option(min=0, help="Zero evaluates the full split.")] = 0,
    per_difficulty: Annotated[
        int, typer.Option(min=0, help="Deterministically sample N tasks per difficulty.")
    ] = 0,
    output_dir: Annotated[Path, typer.Option()] = Path("/root/autodl-tmp/sql-agent-rl/evaluation"),
    env_config_path: Annotated[Path, typer.Option(exists=True)] = Path("configs/env.yaml"),
    api_key_env: Annotated[str, typer.Option()] = "OPENAI_API_KEY",
    seed: Annotated[int, typer.Option()] = 42,
    temperature: Annotated[float, typer.Option(min=0.0)] = 0.7,
    top_p: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    top_k: Annotated[int, typer.Option(min=0)] = 20,
    min_p: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    max_tokens: Annotated[int, typer.Option(min=1)] = 512,
    workers: Annotated[int, typer.Option(min=1, help="Concurrent evaluation episodes.")] = 1,
    prompt_version: Annotated[
        str, typer.Option(help="Multi-turn prompt version.")
    ] = PROMPT_VERSION,
) -> None:
    env_config = EnvConfig.from_yaml(env_config_path)
    tasks = load_tasks(
        task_file if task_file is not None else env_config.processed_data_root / f"{split}.jsonl"
    )
    tasks = select_tasks(tasks, limit=limit, per_difficulty=per_difficulty, seed=seed)
    selected_modes = [BaselineMode(value.strip()) for value in modes.split(",") if value.strip()]
    adapter = OpenAICompatibleAdapter(endpoint, model, api_key=os.getenv(api_key_env))
    runner = EvaluationRunner(
        env_config,
        adapter,
        RunnerConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            seed=seed,
            enable_thinking=False,
            workers=workers,
            prompt_version=prompt_version,
        ),
    )
    summary = runner.run(
        tasks,
        selected_modes,
        log_path=output_dir / "trajectories.jsonl",
        summary_path=output_dir / "summary.json",
    )
    typer.echo(canonical_json(summary))


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
