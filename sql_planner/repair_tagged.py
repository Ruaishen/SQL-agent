"""Build gold-guided, execution-grounded suffixes for failed tagged trajectories.

The batch entry point refuses an incomplete source run. It never changes source files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.models import TaskRecord
from sql_agent.truncation import canonical_json
from sql_planner.collect import _atomic_json, _record_path, ensure_run_manifest, read_spider_train_tasks
from sql_planner.collect_tagged import (
    PROMPT_VERSION as SOURCE_PROMPT_VERSION,
    RESPONSE_RETRIES,
    TextClient,
    build_prompt,
    parse_response,
)
from sql_planner.deepseek import DeepSeekAPIError, DeepSeekClient

REPAIR_VERSION = "gold_guided_tagged_suffix_v2"
GOLD_MENTION = re.compile(r"\b(?:gold|reference|ground.?truth)\s*(?:sql|query|answer)?\b", re.I)


def choose_cutoff(turns: list[dict[str, Any]], *, max_prefix_turns: int = 6) -> tuple[int, str]:
    """Return the one-based first turn to regenerate.

    A repeated final wrong SQL is an observable error. Other earlier semantic
    mistakes cannot be established mechanically, so a six-turn cap is used.
    """
    if max_prefix_turns < 0:
        raise ValueError("max_prefix_turns must be non-negative")
    if not turns or turns[-1].get("tool") != "submit_sql":
        raise ValueError("Only trajectories ending with submit_sql can be repaired")
    final_sql = turns[-1].get("arguments", {}).get("sql")
    cutoff = min(len(turns), max_prefix_turns + 1)
    if isinstance(final_sql, str):
        for index, turn in enumerate(turns[:cutoff - 1], start=1):
            if (turn.get("tool") == "execute_sql"
                    and turn.get("arguments", {}).get("sql", "").strip() == final_sql.strip()):
                return index, "repeated_final_wrong_sql"
    if cutoff == len(turns):
        return cutoff, "replace_submission"
    return cutoff, "prefix_turn_cap"


def teacher_prompt(max_turns: int) -> str:
    base = build_prompt(max_turns)
    return base.replace(
        "Never use or request the gold SQL or gold result.",
        "A separate training-only hint may supply the target SQL. Use it to plan a valid continuation, "
        "but do not mention the hint, gold SQL, or privileged knowledge in reasoning. "
        "Every claimed observation must come from an actual tool result. "
        "Before submit_sql, call execute_sql with the exact SQL you intend to submit, "
        "inspect its real observation, and correct the SQL if necessary.",
    )


def _hint(gold_sql: str) -> str:
    return (
        "Training-only target SQL (never quote or mention this hint in your reasoning):\n"
        + gold_sql
        + "\nContinue from the current observation. Gather any missing schema evidence. "
        "Call execute_sql with the exact final SQL and inspect its real result before calling submit_sql. "
        "Use one <tool> block per turn."
    )


def repair_trajectory(
    source: dict[str, Any], task: TaskRecord, config: EnvConfig, client: TextClient,
    *, temperature: float = 0.3, max_tokens: int = 2048,
    max_prefix_turns: int = 6, response_retries: int = RESPONSE_RETRIES,
) -> dict[str, Any]:
    """Replay a trusted prefix, then generate and execute a gold-guided suffix."""
    if source.get("task_id") != task.task_id or source.get("prompt_version") != SOURCE_PROMPT_VERSION:
        raise ValueError("Source task or tagged prompt version does not match")
    if source.get("correct") or source.get("status") != "submitted_sql":
        raise ValueError("Source must be an incorrect submitted trajectory")
    original = source.get("turns")
    if not isinstance(original, list):
        raise ValueError("Source turns must be a list")
    cutoff, cut_reason = choose_cutoff(original, max_prefix_turns=max_prefix_turns)
    student_prompt = build_prompt(config.max_turns)
    env = SQLAgentEnv(config, reserve_final_submission=True, system_prompt=student_prompt)
    student_messages: list[dict[str, str]] = [
        {"role": "system", "content": student_prompt},
        {"role": "user", "content": task.question},
    ]
    turns: list[dict[str, Any]] = []
    rejected_responses: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status = "incomplete"
    error: str | None = None
    verification: dict[str, Any] | None = None
    final_sql: str | None = None
    gold_check: dict[str, Any] | None = None
    response_index = 0
    try:
        env.reset(task)
        gold_check = env.verify(task.reference_sql).to_dict()
        if not (gold_check["agent_sql_valid"] and gold_check["agent_sql_executable"]
                and gold_check["correct"]):
            status = "gold_not_executable"
        else:
            for original_turn in original[:cutoff - 1]:
                response = original_turn.get("response")
                if not isinstance(response, str):
                    raise ValueError("Prefix turn has no tagged response")
                reasoning, action = parse_response(response)
                if action["tool"] != original_turn.get("tool") or action["arguments"] != original_turn.get("arguments"):
                    raise ValueError("Prefix action does not match source record")
                if action["tool"] == "submit_sql":
                    raise ValueError("Prefix contains final submission")
                observation, _ = env.step(action)
                if canonical_json(observation) != canonical_json(original_turn.get("observation")):
                    raise ValueError("Prefix observation differs when replayed")
                turns.append({**original_turn, "reasoning": reasoning, "origin": "source_prefix"})
                student_messages.append({"role": "assistant", "content": response})
                student_messages.append({"role": "user", "content": f"<observation>{canonical_json(observation)}</observation>"})

            teacher_messages = [{**student_messages[0], "content": teacher_prompt(config.max_turns)}, *student_messages[1:]]
            teacher_messages.append({"role": "user", "content": _hint(task.reference_sql)})
            while not env.done:
                issue: str | None = None
                for retry in range(response_retries + 1):
                    completion = client.complete_text(
                        teacher_messages, temperature=temperature, max_tokens=max_tokens
                    )
                    response_index += 1
                    for key in usage:
                        usage[key] += int(completion.usage.get(key, 0))
                    content = completion.message.get("content") or ""
                    try:
                        if completion.message.get("tool_calls") or completion.finish_reason == "length":
                            raise ValueError("Native or truncated tool response")
                        reasoning, action = parse_response(content)
                        if env.turn >= config.max_turns and action["tool"] != "submit_sql":
                            raise ValueError("Only submit_sql is allowed after exploration budget")
                        if GOLD_MENTION.search(reasoning):
                            raise ValueError("Reasoning mentions privileged gold hint")
                        issue = None
                    except ValueError as exc:
                        issue = str(exc)
                    if issue is None or retry == response_retries:
                        break
                    rejected_responses.append({
                        "response_index": response_index, "reason": issue,
                        "request_id": completion.request_id, "finish_reason": completion.finish_reason,
                        "content": content,
                    })
                if issue is not None:
                    status, error = "invalid_generated_suffix", issue
                    break
                teacher_messages.append({"role": "assistant", "content": content})
                student_messages.append({"role": "assistant", "content": content})
                if action["tool"] == "submit_sql":
                    observation, _ = env.submit_sql(action["arguments"])
                    final_sql = action["arguments"].get("sql")
                    verification = observation.get("verification")
                    tested = any(
                        turn["origin"] == "gold_guided_suffix"
                        and turn["tool"] == "execute_sql"
                        and turn["arguments"].get("sql") == final_sql
                        and turn["observation"].get("status") == "success"
                        for turn in turns
                    )
                    status = (
                        "repaired" if verification and verification.get("correct") and tested
                        else "untested_submission" if verification and verification.get("correct")
                        else "wrong_final_sql"
                    )
                else:
                    observation, _ = env.step(action)
                    if env.done:
                        status = "environment_terminated"
                turns.append({
                    "turn": len(turns) + 1, "response_index": response_index,
                    "reasoning": reasoning, "response": content, "tool": action["tool"],
                    "arguments": action["arguments"], "observation": observation,
                    "request_id": completion.request_id, "finish_reason": completion.finish_reason,
                    "origin": "gold_guided_suffix",
                })
                if env.done:
                    break
                observation_message = {"role": "user", "content": f"<observation>{canonical_json(observation)}</observation>"}
                student_messages.append(observation_message)
                teacher_messages.append(observation_message)
    except (DeepSeekAPIError, ValueError, OSError) as exc:
        status, error = "repair_error", f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return {
        "schema_version": 1, "repair_version": REPAIR_VERSION,
        "source_task_id": task.task_id, "db_id": task.db_id,
        "source_sample": source.get("sample", 0),
        "source_final_sql": source.get("final_sql"),
        "source_status": source.get("status"),
        "cutoff_turn": cutoff, "cutoff_reason": cut_reason,
        "max_prefix_turns": max_prefix_turns,
        "gold_check": gold_check, "gold_hint_sha256": hashlib.sha256(task.reference_sql.encode()).hexdigest(),
        "teacher_model": client.model, "teacher_used_gold_hint": True,
        "status": status, "error": error, "correct": status == "repaired",
        "final_sql": final_sql, "verification": verification,
        "tool_sequence": [turn["tool"] for turn in turns],
        "turns": turns, "rejected_responses": rejected_responses,
        "usage": usage,
        "student_messages": student_messages,
        "trainable_turn_numbers": list(range(cutoff, len(turns) + 1)) if status == "repaired" else [],
        "sft_eligible": status == "repaired",
    }


def validate_complete_source(source_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("prompt_version") != SOURCE_PROMPT_VERSION
            or manifest.get("selected_task_count") != 7000
            or manifest.get("samples_per_task") != 1):
        raise ValueError("Source must be the complete 7000-task tagged collection")
    files = sorted((source_dir / "trajectories").glob("*.json"))
    if len(files) != 7000:
        raise ValueError(f"Source collection is incomplete: {len(files)}/7000 trajectories")
    return manifest, files


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair completed tagged Spider trajectories with a teacher-only gold hint")
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/sql_planner/spider_train_7000_reasoning_tool_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sql_planner/spider_train_7000_reasoning_tool_gold_repair_v2"))
    parser.add_argument("--spider-root", type=Path, default=Path("../datasets/spider/spider_data"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env_sql_planner_local_multi_table_v11.yaml"))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of incorrect records after source completion")
    parser.add_argument("--max-prefix-turns", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0 or args.max_prefix_turns < 0 or args.max_tokens < 1:
        parser.error("Invalid numeric argument")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY in the environment")
    source_dir = args.source_dir.expanduser().resolve()
    source_manifest, files = validate_complete_source(source_dir)
    spider_root = args.spider_root.expanduser().resolve()
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=spider_root)
    tasks = {task.task_id: task for task in read_spider_train_tasks(spider_root, limit=0, seed=42)}
    if len(tasks) != 7000:
        parser.error("Expected 7000 official Spider training tasks")
    if source_manifest.get("source_sha256") != hashlib.sha256((spider_root / "train_spider.json").read_bytes()).hexdigest():
        parser.error("Spider source does not match the collection manifest")
    if source_manifest.get("env_config_sha256") != hashlib.sha256(args.env_config.read_bytes()).hexdigest():
        parser.error("Environment config does not match the collection manifest")
    jobs = []
    seen_task_ids: set[str] = set()
    for path in files:
        source = json.loads(path.read_text(encoding="utf-8"))
        task_id = source.get("task_id")
        if task_id not in tasks or task_id in seen_task_ids:
            parser.error(f"Unknown or duplicate source task: {path}")
        seen_task_ids.add(task_id)
        if source.get("status") == "submitted_sql" and not source.get("correct"):
            task = tasks[task_id]
            jobs.append((source, task, path))
    if args.limit:
        jobs = jobs[:args.limit]
    ensure_run_manifest(args.output_dir, {
        "schema_version": 1, "repair_version": REPAIR_VERSION,
        "source_dir": str(source_dir), "source_manifest_sha256": hashlib.sha256((source_dir / "run_manifest.json").read_bytes()).hexdigest(),
        "source_count": len(files), "selected_incorrect_count": len(jobs),
        "spider_source_sha256": source_manifest["source_sha256"],
        "env_config_sha256": source_manifest["env_config_sha256"],
        "teacher_model": args.model, "max_prefix_turns": args.max_prefix_turns,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
    })
    client = DeepSeekClient(api_key, model=args.model)
    pending = [
        (source, task, _record_path(args.output_dir, task, int(source.get("sample", 0))))
        for source, task, _ in jobs
        if not _record_path(args.output_dir, task, int(source.get("sample", 0))).exists()
    ]

    def run(job: tuple[dict[str, Any], TaskRecord, Path]) -> dict[str, Any]:
        source, task, path = job
        result = repair_trajectory(
            source, task, config, client, temperature=args.temperature,
            max_tokens=args.max_tokens, max_prefix_turns=args.max_prefix_turns,
        )
        _atomic_json(path, result)
        return result

    counts = {"selected": len(jobs), "skipped_existing": len(jobs) - len(pending), "created": 0, "repaired": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run, pending):
            counts["created"] += 1
            counts["repaired"] += int(result["sft_eligible"])
            if counts["created"] % 25 == 0:
                print(json.dumps(counts, sort_keys=True), file=sys.stderr, flush=True)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
