from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

from sql_agent.action_parser import TOOL_DEFINITIONS
from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.models import TaskRecord
from sql_agent.truncation import canonical_json
from sql_planner.deepseek import Completion, DeepSeekAPIError, DeepSeekClient
from third_party.spider_eval.hardness import eval_hardness

PROMPT_VERSION = "sql_planner_multi_table_single_call_v11"


class ToolCallingClient(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion: ...


def build_prompt(max_turns: int) -> str:
    return (
        "You are a read-only SQLite agent. Answer the user's question using the "
        "available tools. Choose tools freely based on the information you have; "
        "there is no required tool order, and you may skip or revisit any tool. "
        "Decide independently whether and when schema inspection, value inspection, "
        "or SQL execution is useful. On every response, output exactly one native "
        "tool call with a valid JSON arguments object containing all required "
        "parameters. Output no prose, explanation, reasoning, chain of thought, "
        "Markdown, or standalone JSON text alongside the tool call. SQL queries "
        "must be read-only SELECT or WITH ... SELECT statements. You have at most "
        f"{max_turns} exploratory tool calls, excluding submit_sql. Use inspect_tables "
        "with table_names to inspect up to 8 relevant tables in one call. "
        "If a multi-table observation is truncated, inspect any omitted tables "
        "in a later call. Use execute_sql with a sql argument to explore or test a query. "
        "Use submit_sql with a sql argument to submit your final query and end "
        "the task; this one final call does not consume the exploration budget. "
        f"After {max_turns} exploratory calls, your next response must call "
        "submit_sql, with no more exploration calls. You may submit earlier "
        "only if the immediately preceding "
        "tool call was a "
        "successful execute_sql in a previous response. When submitting early, "
        "preferably use the same SQL as in that successful execute_sql call. "
        "Otherwise, choose tools freely. "
        "Wait for that tool's observation before choosing your next tool. "
        "Never return multiple tool calls in one response. Tool outputs can be truncated."
    )


def api_tools(*, submit_only: bool = False) -> list[dict[str, Any]]:
    definitions = [
        definition for definition in TOOL_DEFINITIONS
        if definition["name"] != "inspect_table" and not submit_only
    ]
    definitions.append(
        {
            "name": "submit_sql",
            "description": "Execute and submit one final read-only SQLite query, ending the task.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        }
    )
    return [{"type": "function", "function": definition} for definition in definitions]


def read_tasks(path: Path, *, limit: int, seed: int) -> list[TaskRecord]:
    with path.open(encoding="utf-8") as handle:
        tasks = [TaskRecord.from_dict(json.loads(line)) for line in handle if line.strip()]
    if any(task.split != "train" for task in tasks):
        raise ValueError("Free-order collection requires a train-only task file")
    random.Random(seed).shuffle(tasks)
    return tasks[:limit] if limit else tasks


def read_spider_train_tasks(
    spider_root: Path, *, limit: int, seed: int
) -> list[TaskRecord]:
    """Load only the 7,000-example Spider-authored training split."""
    source_path = spider_root / "train_spider.json"
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("train_spider.json must contain a JSON array")
    tasks = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid Spider row at index {index}")
        db_id = row["db_id"]
        tasks.append(
            TaskRecord(
                task_id=f"spider_train_{index:05d}",
                source="spider",
                db_id=db_id,
                db_path=(Path("database") / db_id / f"{db_id}.sqlite").as_posix(),
                question=row["question"],
                reference_sql=row["query"],
                difficulty=eval_hardness(row["sql"]),
                split="train",
            )
        )
    random.Random(seed).shuffle(tasks)
    return tasks[:limit] if limit else tasks


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _record_path(output_dir: Path, task: TaskRecord, sample: int) -> Path:
    digest = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:16]
    return output_dir / "trajectories" / f"{digest}_sample_{sample:02d}.json"


def ensure_run_manifest(output_dir: Path, expected: dict[str, Any]) -> None:
    path = output_dir / "run_manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError("Output directory contains trajectories from different settings")
    elif any((output_dir / "trajectories").glob("*.json")):
        raise ValueError("Existing trajectories have no run manifest; use a fresh output directory")
    else:
        _atomic_json(path, expected)


def _tool_action(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    raw_arguments = function.get("arguments", "{}")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        arguments = raw_arguments
    return {"tool": function.get("name"), "arguments": arguments}


def collect_trajectory(
    task: TaskRecord,
    env_config: EnvConfig,
    client: ToolCallingClient,
    *,
    sample: int = 0,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> dict[str, Any]:
    env = SQLAgentEnv(
        env_config,
        reserve_final_submission=True,
        system_prompt=build_prompt(env_config.max_turns),
        tool_definitions=tuple(
            tool["function"] for tool in api_tools()
            if tool["function"]["name"] != "submit_sql"
        ),
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_prompt(env_config.max_turns)},
        {"role": "user", "content": task.question},
    ]
    turns: list[dict[str, Any]] = []
    response_index = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    response_models: list[str] = []
    status = "max_turns"
    verification: dict[str, Any] | None = None
    submitted_sql: str | None = None
    error: str | None = None
    batch_violation = False
    try:
        env.reset(task)
        while not env.done:
            submit_only = env.turn >= env_config.max_turns
            completion = client.complete(
                messages,
                api_tools(submit_only=submit_only),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_index += 1
            for key in usage:
                usage[key] += int(completion.usage.get(key, 0))
            if completion.model and completion.model not in response_models:
                response_models.append(completion.model)
            raw_calls = completion.message.get("tool_calls") or []
            if not raw_calls:
                content = completion.message.get("content") or ""
                if completion.finish_reason == "length":
                    status = "incomplete_response"
                elif submit_only:
                    status = "max_turns_without_submit"
                else:
                    status = "unexpected_text"
                messages.append({"role": "assistant", "content": content})
                break
            if not isinstance(raw_calls, list):
                status = "invalid_api_response"
                break
            if (completion.message.get("content") or "").strip():
                status = "unexpected_text"
                messages.append({"role": "assistant", "content": completion.message["content"]})
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.message.get("content"),
                    "tool_calls": raw_calls,
                }
            )
            if len(raw_calls) != 1:
                batch_violation = True
                status = "multiple_tool_calls"
                error = f"Expected one tool call per response; received {len(raw_calls)}"
                break
            finalized = False
            for call_index, call in enumerate(raw_calls):
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    status = "invalid_api_response"
                    break
                action = _tool_action(call)
                is_submit = action["tool"] == "submit_sql"
                if env.done or finalized:
                    observation = {
                        "status": "error",
                        "error_type": "turn_budget",
                        "message": "Tool call was not executed because the episode ended",
                    }
                else:
                    if is_submit:
                        observation, _ = env.submit_sql(action["arguments"])
                        arguments = action["arguments"]
                        sql = arguments.get("sql") if isinstance(arguments, dict) else None
                        if observation.get("termination_reason") == "context_limit":
                            status = "context_limit"
                        elif observation.get("error_type") == "invalid_arguments":
                            observation = {
                                key: value for key, value in observation.items()
                                if key not in {"verification", "reward"}
                            }
                            status = "invalid_submission"
                        elif isinstance(sql, str) and sql.strip():
                            submitted_sql = sql
                            verification = (
                                observation.get("verification")
                                or env.verify(sql).to_dict()
                            )
                            observation = {
                                **observation,
                                "verification": verification,
                                "reward": verification["reward"],
                                "termination_reason": "submit_sql",
                            }
                            status = "submitted_sql"
                        else:
                            observation = {
                                key: value for key, value in observation.items()
                                if key not in {"verification", "reward"}
                            }
                            status = "invalid_submission"
                        finalized = True
                    elif env.turn >= env_config.max_turns:
                        observation = {
                            "status": "error",
                            "error_type": "exploration_budget_exhausted",
                            "message": "Only submit_sql is allowed after the exploration budget",
                            "termination_reason": "max_turns_without_submit",
                            "turns_remaining": 0,
                            "max_turns": env_config.max_turns,
                        }
                        status = "max_turns_without_submit"
                        finalized = True
                    elif action["tool"] == "inspect_table":
                        observation = {
                            "status": "error",
                            "error_type": "unoffered_tool",
                            "message": "Use inspect_tables in this collection version",
                        }
                        status = "invalid_api_response"
                        finalized = True
                    else:
                        observation, _ = env.step(action)
                        if env.done and observation.get("termination_reason") == "context_limit":
                            status = "context_limit"
                    turns.append(
                        {
                            "turn": len(turns) + 1,
                            "response_index": response_index,
                            "call_index": call_index,
                            "batch_size": len(raw_calls),
                            "tool": action["tool"],
                            "arguments": action["arguments"],
                            "observation": observation,
                            "request_id": completion.request_id,
                            "finish_reason": completion.finish_reason,
                        }
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": canonical_json(observation),
                    }
                )
            if status in {"invalid_api_response", "invalid_submission", "submitted_sql", "context_limit"} or env.done:
                break
    except (DeepSeekAPIError, ValueError, OSError) as exc:
        status = "collection_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return {
        "schema_version": 2,
        "prompt_version": PROMPT_VERSION,
        "model": client.model,
        "response_models": response_models,
        "task_id": task.task_id,
        "db_id": task.db_id,
        "difficulty": task.difficulty,
        "split": task.split,
        "question": task.question,
        "sample": sample,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_turns": env_config.max_turns,
        "submission_budget": 1,
        "exploration_turns": env.turn,
        "max_tool_calls": env_config.max_turns + 1,
        "status": status,
        "error": error,
        "tool_sequence": [turn["tool"] for turn in turns],
        "has_batched_calls": batch_violation,
        "turns": turns,
        "final_sql": submitted_sql,
        "verification": verification,
        "correct": bool(verification and verification.get("correct")),
        "usage": usage,
        "messages": messages,
    }


def collect(
    tasks: list[TaskRecord],
    env_config: EnvConfig,
    client: ToolCallingClient,
    *,
    output_dir: Path,
    samples_per_task: int,
    temperature: float,
    max_tokens: int,
    workers: int = 1,
    progress_every: int = 0,
) -> dict[str, int]:
    if workers < 1 or progress_every < 0:
        raise ValueError("workers must be positive and progress_every non-negative")
    counts = {"created": 0, "skipped": 0, "correct": 0, "errors": 0}
    pending: list[tuple[TaskRecord, int, Path]] = []
    for task in tasks:
        for sample in range(samples_per_task):
            path = _record_path(output_dir, task, sample)
            if path.exists():
                counts["skipped"] += 1
                continue
            pending.append((task, sample, path))

    def collect_one(job: tuple[TaskRecord, int, Path]) -> dict[str, Any]:
        task, sample, path = job
        record = collect_trajectory(
            task,
            env_config,
            client,
            sample=sample,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        _atomic_json(path, record)
        return record

    if workers == 1:
        records = map(collect_one, pending)
        executor = None
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        records = (
            future.result()
            for future in concurrent.futures.as_completed(
                executor.submit(collect_one, job) for job in pending
            )
        )
    try:
        for record in records:
            counts["created"] += 1
            counts["correct"] += int(record["correct"])
            counts["errors"] += int(record["status"] == "collection_error")
            if progress_every and counts["created"] % progress_every == 0:
                print(json.dumps(counts, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect free-order DeepSeek SQL tool trajectories"
    )
    parser.add_argument("--tasks", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument(
        "--env-config", type=Path, default=Path("configs/env_sql_planner_qwen25_coder_3b.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sql_planner/multi_table_v11"))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--limit", type=int, default=0, help="0 collects all train tasks")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.limit < 0 or args.samples_per_task < 1 or not 0 <= args.temperature <= 2:
        parser.error("Invalid limit, sample count, or temperature")
    if args.max_tokens < 1 or args.workers < 1 or args.progress_every < 0:
        parser.error("max-tokens/workers must be positive; progress-every must be non-negative")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY in the environment")
    env_config = EnvConfig.from_yaml(args.env_config)
    tasks = read_tasks(args.tasks, limit=args.limit, seed=args.seed)
    ensure_run_manifest(args.output_dir, {
        "schema_version": 2,
        "prompt_version": PROMPT_VERSION,
        "exploration_budget": env_config.max_turns,
        "submission_budget": 1,
        "tool_calls_per_response": 1,
        "schema_tool": "inspect_tables",
        "task_file_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "env_config_sha256": hashlib.sha256(args.env_config.read_bytes()).hexdigest(),
        "tools_sha256": hashlib.sha256(canonical_json(api_tools()).encode()).hexdigest(),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "samples_per_task": args.samples_per_task,
        "seed": args.seed,
    })
    client = DeepSeekClient(api_key, model=args.model, base_url=args.base_url)
    counts = collect(
        tasks, env_config, client, output_dir=args.output_dir,
        samples_per_task=args.samples_per_task,
        temperature=args.temperature, max_tokens=args.max_tokens,
        workers=args.workers, progress_every=args.progress_every,
    )
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
