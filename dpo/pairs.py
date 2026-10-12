"""Pair original no-Gold failures with SFT's verified, audited regeneration."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from dataclasses import replace
from pathlib import Path


def public_messages(record: dict) -> list[dict]:
    messages = record.get("student_messages", record.get("messages"))
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Missing complete public conversation")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("Expected system and original question")
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"} or not isinstance(message["content"], str):
            raise ValueError("Private fields or invalid message content")
        if "<teacher_only_sql>" in message["content"]:
            raise ValueError("Teacher-only supplement in public messages")
        if index >= 2 and message["role"] != ("assistant" if index % 2 == 0 else "user"):
            raise ValueError("Invalid assistant/observation ordering")
    if messages[-1]["role"] != "assistant":
        raise ValueError("Conversation must end in an assistant response")
    return messages


def make_pair(loser: dict, winner: dict) -> dict:
    """Execution replay is performed by build; here validate pairing boundaries."""
    if loser.get("correct") is not False or winner.get("correct") is not True:
        raise ValueError("Expected incorrect y_l and correct y_w")
    for key in ("task_id", "db_id", "question"):
        if loser.get(key) is None or loser[key] != winner.get(key):
            raise ValueError(f"Pair does not match on {key}")
    if loser.get("split") != "train" or winner.get("split") != "train":
        raise ValueError("Use train-only pairs")
    if loser.get("teacher_used_gold_hint") or loser.get("generation_mode") or loser.get("merged_origin") in {"gold_repair", "gold_regeneration"}:
        raise ValueError("y_l must be an original no-Gold trajectory")
    if winner.get("generation_mode") != "full_regeneration_from_question" or winner.get("leak_audit") != "accept" or winner.get("sft_eligible") is not True:
        raise ValueError("y_w must pass SFT full regeneration and leakage audit")
    rejected, chosen = public_messages(loser), public_messages(winner)
    if rejected[:2] != chosen[:2] or rejected[1]["content"] != loser["question"]:
        raise ValueError("System and question must be identical")
    if rejected == chosen:
        raise ValueError("Identical branches")
    return {"schema_version": 1, "task_id": loser["task_id"], "db_id": loser["db_id"],
            "question": loser["question"], "split": "train", "prompt_messages": chosen[:2],
            "chosen_messages": chosen, "rejected_messages": rejected,
            "chosen_final_sql": winner.get("final_sql"), "rejected_final_sql": loser.get("final_sql", loser.get("submitted_sql")),
            "leak_audit": "accept", "chosen_correct": True, "rejected_correct": False,
            "pairing_mode": "full_original_vs_gold_regeneration"}


def build(original: Path, regenerated: Path, tasks, config, output: Path, *, workers: int = 1,
          strict_submission: bool = True) -> dict:
    from sft.curate_reasoning import EXTERNAL_REFERENCE, validate_record
    from sql_agent.env import SQLAgentEnv

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if workers < 1:
        raise ValueError("Validation workers must be positive")
    registry = {task.task_id: task for task in tasks}
    if len(registry) != len(tasks) or not registry or any(t.split != "train" for t in tasks):
        raise ValueError("Authoritative tasks must be unique and train-only")
    def index(root):
        paths = sorted((root / "trajectories").glob("*.json"))
        if not paths:
            raise ValueError(f"No source trajectories: {root}")
        records = {}
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            task_id = record["task_id"]
            if task_id in records:
                raise ValueError(f"Duplicate task: {task_id}")
            records[task_id] = (path, record)
        return records
    originals, winners = index(original), index(regenerated)
    pairs, skipped = [], []
    def validate_one(item):
        task_id, (winner_path, winner) = item
        if task_id not in registry or task_id not in originals:
            return None, {"task_id": task_id, "reason": "No matching train task/original"}
        loser_path, loser = originals[task_id]
        try:
            pair = make_pair(loser, winner)
            # Reuse SFT's real tool replay, Gold comparison and final verification.
            validate_record(winner, registry[task_id], config, regenerated=True,
                            strict_submission=strict_submission)
            for message in pair["rejected_messages"][2::2]:
                reasoning = message["content"].split("</reasoning>", 1)[0]
                if EXTERNAL_REFERENCE.search(reasoning):
                    raise ValueError("Original public reasoning references an external answer")
            final_sql = pair["rejected_final_sql"]
            if isinstance(final_sql, str) and final_sql:
                env = SQLAgentEnv(config)
                try:
                    env.reset(registry[task_id])
                    if env.verify(final_sql).correct:
                        raise ValueError("Original SQL is correct; cannot label it y_l")
                finally:
                    env.close()
            elif loser.get("status") != "invalid_format":
                raise ValueError("Original lacks final SQL and is not an invalid-format failure")
            pair["source_files"] = {
                role: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for role, path in (("y_l", loser_path), ("y_w", winner_path))}
            return pair, None
        except (ValueError, KeyError, TypeError) as exc:
            return None, {"task_id": task_id, "reason": str(exc)}
    # Each replay uses its own read-only SQLite environment; retain stable task order.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for pair, failure in pool.map(validate_one, winners.items()):
            if pair is not None:
                pairs.append(pair)
            else:
                skipped.append(failure)
    if not pairs:
        raise ValueError(f"No eligible preference pairs: {skipped}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    report = {"pairs": len(pairs), "skipped": skipped, "source_kind": "SFT verified/audited full regeneration",
              "submission_validation": "strict_tested_gold" if strict_submission else "live_execution_result_equivalence"}
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    from sql_agent.config import EnvConfig
    from sql_agent.data import load_tasks
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dpo/pairs.jsonl"))
    args = parser.parse_args()
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=args.spider_root.resolve())
    print(json.dumps(build(args.original, args.regenerated, load_tasks(args.tasks), config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
