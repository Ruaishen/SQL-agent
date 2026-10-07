from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from experience_memory.evolve import ensure_manifest, read_json, write_json
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.deepseek import Completion, DeepSeekAPIError, DeepSeekClient
from sql_agent.env import SQLAgentEnv
from sql_agent.models import TaskRecord
from sql_agent.protocol import PROMPT_VERSION, build_prompt
from sql_agent.protocol import parse_tagged_response as parse_response
from sql_agent.truncation import canonical_json

RESPONSE_RETRIES = 2


class TextClient(Protocol):
    model: str

    def complete_text(
        self, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int
    ) -> Completion: ...


def collect_trajectory(
    task: TaskRecord,
    config: EnvConfig,
    client: TextClient,
    *,
    sample: int = 0,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    response_retries: int = RESPONSE_RETRIES,
) -> dict[str, Any]:
    if response_retries < 0:
        raise ValueError("response_retries must be non-negative")
    prompt = build_prompt(config.max_turns)
    env = SQLAgentEnv(config, reserve_final_submission=True, system_prompt=prompt)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": task.question},
    ]
    turns: list[dict[str, Any]] = []
    rejected_responses: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    response_models: list[str] = []
    response_index = 0
    status = "incomplete"
    error: str | None = None
    submitted_sql: str | None = None
    verification: dict[str, Any] | None = None
    try:
        env.reset(task)
        while not env.done:
            for retry in range(response_retries + 1):
                completion = client.complete_text(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                response_index += 1
                for key in usage:
                    usage[key] += int(completion.usage.get(key, 0))
                if completion.model and completion.model not in response_models:
                    response_models.append(completion.model)
                content = completion.message.get("content") or ""
                try:
                    if completion.message.get("tool_calls"):
                        raise ValueError("Native tool calls are not part of the tagged protocol")
                    if completion.finish_reason == "length":
                        raise ValueError("Response was truncated")
                    reasoning, action = parse_response(content)
                    if env.turn >= config.max_turns and action["tool"] != "submit_sql":
                        raise ValueError("Only submit_sql is allowed after the exploration budget")
                    issue = None
                except ValueError as exc:
                    issue = str(exc)
                if issue is None or retry == response_retries:
                    break
                rejected_responses.append(
                    {
                        "response_index": response_index,
                        "reason": issue,
                        "request_id": completion.request_id,
                        "finish_reason": completion.finish_reason,
                        "content": content,
                    }
                )
            if issue is not None:
                status = "invalid_format"
                error = issue
                messages.append({"role": "assistant", "content": content})
                break
            messages.append({"role": "assistant", "content": content})
            if action["tool"] == "submit_sql":
                observation, _ = env.submit_sql(action["arguments"])
                submitted_sql = action["arguments"]["sql"]
                verification = observation.get("verification")
                status = "submitted_sql" if verification is not None else "invalid_submission"
            else:
                observation, _ = env.step(action)
                if env.done and observation.get("termination_reason") == "context_limit":
                    status = "context_limit"
            turns.append(
                {
                    "turn": len(turns) + 1,
                    "response_index": response_index,
                    "reasoning": reasoning,
                    "response": content,
                    "tool": action["tool"],
                    "arguments": action["arguments"],
                    "observation": observation,
                    "request_id": completion.request_id,
                    "finish_reason": completion.finish_reason,
                }
            )
            if env.done:
                break
            messages.append(
                {
                    "role": "user",
                    "content": f"<observation>{canonical_json(observation)}</observation>",
                }
            )
    except (DeepSeekAPIError, ValueError, OSError) as exc:
        status = "collection_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "model": client.model,
        "response_models": response_models,
        "task_id": task.task_id,
        "db_id": task.db_id,
        "difficulty": task.difficulty,
        "split": task.split,
        "question": task.question,
        "sample": sample,
        "status": status,
        "error": error,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_turns": config.max_turns,
        "response_retries": response_retries,
        "exploration_turns": env.turn,
        "tool_sequence": [turn["tool"] for turn in turns],
        "turns": turns,
        "rejected_responses": rejected_responses,
        "final_sql": submitted_sql,
        "verification": verification,
        "correct": bool(verification and verification.get("correct")),
        "usage": usage,
        "messages": messages,
        "trainable_turn_numbers": list(range(1, len(turns) + 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect original, gold-free reasoning SFT trajectories"
    )
    parser.add_argument("--tasks", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sft/original"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.samples_per_task < 1 or args.limit < 0 or args.max_tokens < 1:
        parser.error("Invalid collection limits")
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks or any(task.split != "train" for task in tasks):
        parser.error("Use a nonempty train-only task file")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        parser.error("Set DEEPSEEK_API_KEY in the environment")
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=args.spider_root.resolve())
    ensure_manifest(
        args.output_dir / "manifest.json",
        {
            "model": args.model,
            "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
            "env_sha256": hashlib.sha256(args.env_config.read_bytes()).hexdigest(),
            "spider_root": str(config.spider_root),
            "limit": args.limit,
            "samples_per_task": args.samples_per_task,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt": build_prompt(config.max_turns),
        },
    )
    client = DeepSeekClient(key, model=args.model)
    counts = {"completed": 0, "correct": 0}
    for task in tasks:
        for sample in range(args.samples_per_task):
            identifier = hashlib.sha256(f"{task.task_id}:{sample}".encode()).hexdigest()
            path = args.output_dir / "trajectories" / f"{identifier}.json"
            record = (
                read_json(path)
                if path.exists()
                else collect_trajectory(
                    task,
                    config,
                    client,
                    sample=sample,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            )
            write_json(path, record)
            counts["completed"] += 1
            counts["correct"] += int(record["correct"])
            if record["status"] == "collection_error":
                raise RuntimeError(
                    "Collection failed; inspect trajectory and use a new output directory"
                )
    write_json(args.output_dir / "summary.json", counts)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
