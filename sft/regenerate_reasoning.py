"""Gold-guided full regeneration with independent public-message leakage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from experience_memory.evolve import ensure_manifest, read_json, write_json
from sft.curate_reasoning import EXTERNAL_REFERENCE
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.deepseek import DeepSeekClient
from sql_agent.env import SQLAgentEnv
from sql_agent.protocol import parse_tagged_response
from sql_agent.truncation import TokenCounter, canonical_json

PROMPTS = Path(__file__).parent / "prompts"


def trajectory_profile(messages: list[dict], counter: TokenCounter) -> dict:
    public = messages[2:]
    tool_calls = 0
    style = []
    for message in public:
        if message.get("role") != "assistant":
            continue
        try:
            reasoning, _ = parse_tagged_response(message["content"])
            tool_calls += 1
            if len(style) < 3 and not EXTERNAL_REFERENCE.search(reasoning):
                style.append(reasoning[:240])
        except ValueError:
            pass
    return {
        "total_tokens": sum(counter.count_text(m["content"]) for m in public if m["role"] == "assistant"),
        "tool_calls": tool_calls,
        "reasoning_style_examples": style,
    }


def trajectory_comparison(original: dict, generated: dict) -> dict:
    return {"original": original, "generated": generated,
            "differences": {key: generated[key] - original[key] for key in ("total_tokens", "tool_calls")},
            "ratios": {key: generated[key] / original[key] if original[key] else None for key in ("total_tokens", "tool_calls")}}


class RegenerationError(ValueError):
    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


def regenerate(task, source: dict, config: EnvConfig, client, *, max_tokens: int = 8192,
               thinking: bool = True, temperature: float = 0.3) -> dict:
    if task.split != "train" or source.get("correct") is not False:
        raise ValueError("Regeneration requires a failed training example")
    if source.get("question") != task.question or source.get("db_id") != task.db_id:
        raise ValueError("Failed source does not match task")
    base = source["messages"][0]
    if base.get("role") != "system":
        raise ValueError("Missing original SFT system prompt")
    supplement = (PROMPTS / "gold_repair_supplement_zh.md").read_text(encoding="utf-8")
    audit_prompt = (PROMPTS / "external_answer_leak_audit_zh.md").read_text(encoding="utf-8")
    counter = TokenCounter(config.tokenizer_path)
    original_profile = trajectory_profile(source["messages"], counter)
    student = [dict(base), {"role": "user", "content": task.question}]
    teacher = [
        dict(base),
        {"role": "user", "content": supplement.replace("{original_profile}", json.dumps(original_profile, ensure_ascii=False)).replace("{gold_sql}", task.reference_sql)},
        dict(student[1]),
    ]
    turns, calls = [], []
    env = SQLAgentEnv(config, reserve_final_submission=True, system_prompt=base["content"])
    tested = False
    try:
        env.reset(task)
        if not env.verify(task.reference_sql).correct:
            raise ValueError("Gold SQL failed preflight")
        for index in range(config.max_turns + 1):
            if thinking:
                completion = client.complete_thinking_text(teacher, max_tokens=max_tokens, stop_at_tool=True)
            else:
                completion = client.complete_text(teacher, max_tokens=max_tokens, temperature=temperature)
            calls.append(
                {
                    "message": completion.message,
                    "usage": completion.usage,
                    "finish_reason": completion.finish_reason,
                }
            )
            if completion.finish_reason != "stop" or (thinking and not completion.message.get("reasoning_content")):
                raise ValueError("Teacher must return complete high-effort text" if thinking
                                 else "Teacher must return complete non-thinking text")
            content = completion.message.get("content", "")
            reasoning, action = parse_tagged_response(content)
            if index == 0 and action["tool"] != "list_tables":
                raise ValueError("Full regeneration must start with list_tables")
            if tested and action["tool"] != "submit_sql":
                raise ValueError("Successful exact Gold execution must be followed by submission")
            if action["tool"] == "submit_sql":
                if not tested or action["arguments"]["sql"] != task.reference_sql:
                    raise ValueError("Submit the exact previously tested Gold SQL")
                observation, _ = env.submit_sql(action["arguments"])
                if not observation.get("verification", {}).get("correct"):
                    raise ValueError("Submission failed verification")
            else:
                if env.turn >= config.max_turns:
                    raise ValueError("Exploration budget exhausted")
                if env.turn == config.max_turns - 1 and (
                    action["tool"] != "execute_sql"
                    or action["arguments"].get("sql") != task.reference_sql
                ):
                    raise ValueError("Reserve final exploration for exact Gold execution")
                observation, done = env.step(action)
                if done:
                    raise ValueError("Environment terminated before submission")
                tested = (
                    action["tool"] == "execute_sql"
                    and action["arguments"]["sql"] == task.reference_sql
                    and observation.get("status") == "success"
                )
            student.append({"role": "assistant", "content": content})
            # Teacher receives its private reasoning again; student never does.
            teacher_message = {"role": "assistant", "content": content}
            if thinking:
                teacher_message["reasoning_content"] = completion.message["reasoning_content"]
            teacher.append(teacher_message)
            turns.append(
                {
                    "turn": len(turns) + 1,
                    "response": content,
                    "reasoning": reasoning,
                    "tool": action["tool"],
                    "arguments": action["arguments"],
                    "observation": observation,
                }
            )
            if action["tool"] == "submit_sql":
                break
            message = {
                "role": "user",
                "content": f"<observation>{canonical_json(observation)}</observation>",
            }
            student.append(dict(message))
            teacher.append(dict(message))
        else:
            raise ValueError("Missing final submission")
    except ValueError as exc:
        raise RegenerationError(str(exc), {
            "sft_eligible": False, "student_messages": student, "turns": turns,
            "private_teacher_messages": teacher, "private_api_calls": calls,
        }) from exc
    finally:
        env.close()
    verdict = "reject"
    comparison = trajectory_comparison(original_profile, trajectory_profile(student, counter))
    if not any(EXTERNAL_REFERENCE.search(turn["reasoning"]) for turn in turns):
        audit = client.complete_thinking_text(
            [
                {"role": "system", "content": audit_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": task.question,
                            "messages": student,
                            "trajectory_comparison": comparison,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=max_tokens,
        )
        calls.append(
            {"message": audit.message, "usage": audit.usage, "finish_reason": audit.finish_reason}
        )
        if audit.finish_reason != "stop" or audit.message.get("content") not in {
            "accept",
            "reject",
        }:
            raise RegenerationError("Audit must return exactly accept or reject", {
                "sft_eligible": False, "student_messages": student, "turns": turns,
                "private_teacher_messages": teacher, "private_api_calls": calls,
            })
        verdict = audit.message["content"]
    return {
        "task_id": task.task_id,
        "db_id": task.db_id,
        "question": task.question,
        "difficulty": task.difficulty,
        "split": task.split,
        "correct": True,
        "sft_eligible": verdict == "accept",
        "leak_audit": verdict,
        "trajectory_comparison": comparison,
        "generation_mode": "full_regeneration_from_question",
        "generation_thinking": thinking,
        "messages": student,
        "student_messages": student,
        "turns": turns,
        "final_sql": task.reference_sql,
        "trainable_turn_numbers": list(range(1, len(turns) + 1)),
        "private_teacher_messages": teacher,
        "private_api_calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sft/gold_regenerated"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.attempts, args.max_tokens) < 1 or args.limit < 0:
        parser.error("Invalid limits")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        parser.error("Set DEEPSEEK_API_KEY in the environment")
    tasks = load_tasks(args.tasks)
    if not tasks or any(task.split != "train" for task in tasks):
        parser.error("Use train-only tasks")
    registry = {task.task_id: task for task in tasks}
    sources = sorted((args.source / "trajectories").glob("*.json"))
    selected = [(path, read_json(path)) for path in sources]
    selected = [
        (path, record)
        for path, record in selected
        if record.get("correct") is False and record.get("task_id") in registry
    ]
    if args.limit:
        selected = selected[: args.limit]
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=args.spider_root.resolve())
    ensure_manifest(
        args.output_dir / "manifest.json",
        {
            "model": args.model,
            "thinking": "enabled",
            "reasoning_effort": "high",
            "source": str(args.source.resolve()),
            "source_hashes": [
                hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in selected
            ],
            "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
            "env_sha256": hashlib.sha256(args.env_config.read_bytes()).hexdigest(),
            "spider_root": str(config.spider_root),
            "attempts": args.attempts,
            "max_tokens": args.max_tokens,
            "prompts": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(PROMPTS.glob("*.md"))
            },
        },
    )
    client = DeepSeekClient(key, model=args.model)
    for path, source in selected:
        task = registry[source["task_id"]]
        identifier = hashlib.sha256(task.task_id.encode()).hexdigest()
        accepted_path = args.output_dir / "trajectories" / f"{identifier}.json"
        if accepted_path.exists():
            continue
        for attempt in range(1, args.attempts + 1):
            attempt_path = args.output_dir / "attempts" / identifier / f"{attempt}.json"
            if attempt_path.exists():
                record = read_json(attempt_path)
            else:
                try:
                    record = regenerate(task, source, config, client, max_tokens=args.max_tokens)
                except (ValueError, OSError) as exc:
                    record = {**getattr(exc, "record", {}),
                              "sft_eligible": False, "error": str(exc)}
                write_json(attempt_path, record)
            if record.get("sft_eligible") is True:
                public = {
                    name: value for name, value in record.items() if not name.startswith("private_")
                }
                public["source_record_path"] = str(path.resolve())
                write_json(accepted_path, public)
                break


if __name__ == "__main__":
    main()
