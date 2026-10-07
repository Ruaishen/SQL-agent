"""Build a train-only SFT source from originals and audited full regenerations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from experience_memory.evolve import read_json, write_json
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.env import SQLAgentEnv
from sql_agent.protocol import parse_tagged_response
from sql_agent.truncation import canonical_json

EXTERNAL_REFERENCE = re.compile(
    r"\bgold(?:en)?[ _-]+(?:sql|query|answer|result)\b|"
    r"\b(?:provided|teacher|training|privileged)\s+hint\b|"
    r"\binternal(?:ly)?\s+(?:calibrat\w*|supplied)\b|"
    r"\bteacher(?:'s)?\s+(?:answer|query|sql)\b|"
    r"\bcontinuation\s+instruction\b|"
    r"内部校准|参考答案|教师答案|提供的查询|指定的最终\s*SQL",
    re.IGNORECASE,
)


def validate_record(record: dict, task, config: EnvConfig, *, regenerated: bool) -> dict:
    if task.split != "train" or record.get("correct") is not True:
        raise ValueError("Expected an execution-correct training example")
    if record.get("question") != task.question or record.get("db_id") != task.db_id:
        raise ValueError("Source does not match authoritative task")
    if regenerated and (
        record.get("generation_mode") != "full_regeneration_from_question"
        or record.get("leak_audit") != "accept"
        or record.get("sft_eligible") is not True
    ):
        raise ValueError("Regeneration requires a full restart and accepted leakage audit")
    messages = record.get("student_messages", record.get("messages"))
    turns = record.get("turns", [])
    if not messages or len(messages) != 2 * len(turns) + 1:
        raise ValueError("Incomplete student message/turn sequence")
    if messages[0].get("role") != "system" or messages[1] != {
        "role": "user",
        "content": task.question,
    }:
        raise ValueError("Student prefix must contain only base prompt and original question")
    if any(set(message) != {"role", "content"} for message in messages):
        raise ValueError("Student messages contain private or unexpected fields")
    if any("<teacher_only_sql>" in message["content"] for message in messages):
        raise ValueError("Teacher supplement leaked into student messages")
    env = SQLAgentEnv(config, reserve_final_submission=True, system_prompt=messages[0]["content"])
    try:
        env.reset(task)
        if not env.verify(task.reference_sql).correct:
            raise ValueError("Gold execution preflight failed")
        for index, turn in enumerate(turns):
            assistant = messages[2 + 2 * index]
            if assistant.get("role") != "assistant" or assistant["content"] != turn["response"]:
                raise ValueError("Assistant messages differ from recorded turns")
            reasoning, action = parse_tagged_response(assistant["content"])
            if EXTERNAL_REFERENCE.search(reasoning):
                raise ValueError("Public reasoning references an external answer source")
            if action != {"tool": turn["tool"], "arguments": turn["arguments"]}:
                raise ValueError("Recorded action differs from assistant response")
            if index == 0 and action["tool"] != "list_tables":
                raise ValueError("Full trajectories must start with list_tables")
            if index == len(turns) - 1:
                if action["tool"] != "submit_sql" or len(turns) < 2:
                    raise ValueError("Missing final student submission")
                sql = action["arguments"]["sql"]
                previous = turns[index - 1]
                if regenerated and (
                    previous["tool"] != "execute_sql"
                    or previous["arguments"].get("sql") != sql
                    or previous["observation"].get("status") != "success"
                ):
                    raise ValueError(
                        "Submission must immediately follow successful identical execution"
                    )
                if not any(
                    prior["tool"] == "execute_sql"
                    and prior["arguments"].get("sql") == sql
                    and prior["observation"].get("status") == "success"
                    for prior in turns[:index]
                ):
                    raise ValueError("Final SQL lacks a successful matching execution")
                if sql != record.get("final_sql"):
                    raise ValueError("Final SQL field differs from submitted SQL")
                if regenerated and sql != task.reference_sql:
                    raise ValueError("Regeneration must preserve the exact Gold SQL string")
                observation, _ = env.submit_sql(action["arguments"])
                if not observation.get("verification", {}).get("correct"):
                    raise ValueError("Final SQL failed live execution verification")
            else:
                if action["tool"] == "submit_sql":
                    raise ValueError("Premature submission in saved trajectory")
                observation, done = env.step(action)
                if done:
                    raise ValueError("Environment terminated before final submission")
                if canonical_json(observation) != canonical_json(turn["observation"]):
                    raise ValueError("Stored observation differs from live tool result")
                if messages[3 + 2 * index] != {
                    "role": "user",
                    "content": f"<observation>{canonical_json(observation)}</observation>",
                }:
                    raise ValueError("Student observation message differs from actual tool result")
    finally:
        env.close()
    return {
        "task_id": task.task_id,
        "db_id": task.db_id,
        "question": task.question,
        "difficulty": task.difficulty,
        "split": "train",
        "correct": True,
        "merged_origin": "gold_regeneration" if regenerated else "original",
        "messages": messages,
        "turns": turns,
        "final_sql": sql,
        "trainable_turn_numbers": list(range(1, len(turns) + 1)),
    }


def curate(
    *, original: Path, regenerated: Path | None, tasks, config: EnvConfig, output: Path
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new curated output directory")
    registry = {task.task_id: task for task in tasks}
    if len(registry) != len(tasks) or any(task.split != "train" for task in tasks):
        raise ValueError("Authoritative tasks must be unique and train-only")
    retained = set()
    decisions = []
    counts = {"original": 0, "gold_regeneration": 0, "rejected": 0, "skipped": 0}
    for root, is_regenerated in [(original, False), (regenerated, True)]:
        if root is None:
            continue
        paths = sorted((root / "trajectories").glob("*.json"))
        if not paths:
            raise ValueError(f"No trajectories found: {root}")
        for path in paths:
            record = read_json(path)
            task_id = record.get("task_id")
            if task_id not in registry or task_id in retained or record.get("correct") is not True:
                counts["skipped"] += 1
                continue
            try:
                normalized = validate_record(
                    record, registry[task_id], config, regenerated=is_regenerated
                )
                normalized["source_record_path"] = str(path.resolve())
                normalized["source_record_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                key = hashlib.sha256(task_id.encode()).hexdigest()
                write_json(output / "trajectories" / f"{key}.json", normalized)
                retained.add(task_id)
                counts[normalized["merged_origin"]] += 1
            except (ValueError, KeyError, TypeError) as exc:
                counts["rejected"] += 1
                decisions.append({"task_id": task_id, "source": str(path), "reason": str(exc)})
    write_json(output / "audit.json", {"counts": counts, "rejections": decisions})
    if not retained:
        raise ValueError("No eligible SFT trajectories")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path)
    parser.add_argument("--tasks", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sft/curated"))
    args = parser.parse_args()
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=args.spider_root.resolve())
    print(
        json.dumps(
            curate(
                original=args.original,
                regenerated=args.regenerated,
                tasks=load_tasks(args.tasks),
                config=config,
                output=args.output,
            )
        )
    )


if __name__ == "__main__":
    main()
