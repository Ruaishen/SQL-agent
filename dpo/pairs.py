"""Account for every wrong trajectory and build gold-guided DPO pairs.

Selection is offline. ``generate`` needs DEEPSEEK_API_KEY and leaves source files
untouched. A pair is emitted only after replay and final execution verification.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

PROMPT_VERSION = "reasoning_tool_observation_v4"


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fork_info(record: dict, judge_sql: Callable[[str], bool | None]) -> dict:
    """Choose the last verified wrong success, then explicit fallback locations."""
    turns = record.get("turns") or []
    executed = [(number, turn) for number, turn in enumerate(turns, 1)
                if turn.get("tool") == "execute_sql"]
    wrong_success = []
    for number, turn in executed:
        sql = turn.get("arguments", {}).get("sql")
        if (turn.get("observation", {}).get("status") == "success"
                and isinstance(sql, str) and judge_sql(sql) is False):
            wrong_success.append((number, sql))
    if wrong_success:
        number, sql = wrong_success[-1]
        return {"fork_turn": number + 1, "fork_reason": "after_wrong_sql_success",
                "fork_sql": sql}
    failures = [(number, turn) for number, turn in executed
                if turn.get("observation", {}).get("status") != "success"]
    if failures:
        number, turn = failures[-1]
        return {"fork_turn": number + 1, "fork_reason": "after_sql_error",
                "fork_sql": turn.get("arguments", {}).get("sql")}
    if executed:
        number, turn = executed[-1]
        return {"fork_turn": number + 1, "fork_reason": "after_other_sql_success",
                "fork_sql": turn.get("arguments", {}).get("sql")}
    if record.get("status") == "submitted_sql":
        return {"fork_turn": len(turns), "fork_reason": "before_untested_submission",
                "fork_sql": None}
    return {"fork_turn": len(turns) + 1, "fork_reason": "before_invalid_response",
            "fork_sql": None}


def select(source_dir: Path, *, spider_root: Path | None = None,
           env_config: Path | None = None,
           judge_factory: Callable[[dict], tuple[bool, Callable[[str], bool | None]]] | None = None,
           workers: int = 8) -> dict:
    paths = sorted((source_dir / "trajectories").glob("*.json"))
    if not paths:
        raise ValueError("No source trajectories")
    if judge_factory is None:
        if spider_root is None or env_config is None:
            raise ValueError("Spider root and environment config are required")
        from sql_agent.config import EnvConfig
        from sql_agent.verifier import ExecutionVerifier
        from sql_planner.collect import read_spider_train_tasks

        root = spider_root.resolve()
        config = replace(EnvConfig.from_yaml(env_config), spider_root=root)
        tasks = {task.task_id: task for task in read_spider_train_tasks(root, limit=0, seed=42)}

        def judge_factory(record: dict):
            task = tasks[record["task_id"]]
            verifier = ExecutionVerifier(task.resolve_db_path(root), task.reference_sql, config)
            source_check = record.get("verification") or {}
            if source_check.get("error") == "wrong_result":
                # The collection verifier already executed the gold SQL in full.
                gold_valid = True
            elif str(source_check.get("error", "")).startswith("verifier_reference_"):
                gold_valid = False
            else:
                gold = verifier.verify(task.reference_sql)
                gold_valid = bool(gold.correct and gold.agent_sql_executable)
            cache = {}

            def judge(sql: str) -> bool | None:
                if sql not in cache:
                    if sql == record.get("final_sql") and source_check.get("error") == "wrong_result":
                        cache[sql] = False
                    else:
                        verdict = verifier.verify(sql)
                        cache[sql] = (verdict.correct if verdict.agent_sql_executable
                                      and not (verdict.error or "").startswith("verifier_reference_")
                                      else None)
                return cache[sql]

            return gold_valid, judge

    def inspect(path: Path) -> dict | None:
        raw = path.read_bytes()
        record = json.loads(raw)
        if record.get("correct") is True:
            return None
        if (record.get("split") != "train" or record.get("prompt_version") != PROMPT_VERSION
                or record.get("status") not in {"submitted_sql", "invalid_format"}):
            raise ValueError(f"Unsupported wrong source record: {path}")
        gold_valid, judge = judge_factory(record)
        entry = {"file": path.name, "task_id": record["task_id"],
                 "source_status": record["status"], "source_sha256": hashlib.sha256(raw).hexdigest(),
                 "eligible": gold_valid}
        if gold_valid:
            entry.update(fork_info(record, judge))
            index = 2 * entry["fork_turn"]
            messages = record.get("messages") or []
            if (len(messages) <= index or messages[index].get("role") != "assistant"
                    or messages[:2] != [
                        {"role": "system", "content": messages[0].get("content")},
                        {"role": "user", "content": record.get("question")},
                    ]):
                raise ValueError(f"Invalid source message alignment: {path}")
        else:
            entry["block_reason"] = "gold_not_executable"
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        entries = [entry for entry in pool.map(inspect, paths) if entry is not None]
    ids = [entry["task_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs in wrong trajectories")
    reasons = Counter(entry.get("fork_reason", entry.get("block_reason")) for entry in entries)
    return {"schema_version": 2, "source_dir": str(source_dir.resolve()),
            "source_files": len(paths), "candidate_count": len(entries),
            "eligible_count": sum(entry["eligible"] for entry in entries),
            "fork_reasons": dict(sorted(reasons.items())), "entries": entries}


def make_pair(source: dict, repaired: dict, fork: int, reason: str) -> dict:
    if repaired.get("status") not in {"repaired", "repaired_no_test_budget", "repaired_direct_submit"} or repaired.get("cutoff_turn") != fork:
        raise ValueError("Gold-guided suffix did not pass replay and verification")
    rejected = source["messages"]
    chosen = repaired["student_messages"]
    prefix_end = 2 * fork  # system + question + (fork-1) assistant/observation pairs
    if chosen[:prefix_end] != rejected[:prefix_end]:
        raise ValueError("Chosen and rejected branches have different prefixes")
    if chosen[prefix_end]["role"] != "assistant" or rejected[prefix_end]["role"] != "assistant":
        raise ValueError("Fork is not at an assistant response")
    if chosen[prefix_end:] == rejected[prefix_end:]:
        raise ValueError("Preference branches are identical")
    return {"schema_version": 1, "task_id": source["task_id"], "split": "train",
            "db_id": source["db_id"], "fork_turn": fork, "fork_reason": reason,
            "source_sample": source.get("sample", 0),
            "source_status": source["status"], "source_final_sql": source.get("final_sql"),
            "chosen_final_sql": repaired["final_sql"],
            "chosen_verification": repaired["verification"],
            "rejected_verification": source.get("verification"),
            "chosen_tested_before_submit": repaired["status"] == "repaired",
            "prefix_observation_drift": repaired.get("prefix_observation_drift", []),
            "sanitized_reasoning_turns": repaired.get("sanitized_reasoning_turns", []),
            "teacher_used_gold_hint": True,
            "gold_hint_sha256": repaired["gold_hint_sha256"],
            "teacher_model": repaired["teacher_model"],
            "prompt_messages": chosen[:prefix_end],
            "chosen_messages": chosen,
            "rejected_messages": rejected,
            "chosen_turns": repaired["turns"][fork - 1:],
            "rejected_turns": source["turns"][fork - 1:],
            "loss_from_turn": fork}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "generate"))
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/sql_planner/reasoning"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sql_planner/reasoning_dpo_all_errors_v1"))
    parser.add_argument("--spider-root", type=Path, default=Path("../datasets/spider/spider_data"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env_sql_planner_local_multi_table_v11.yaml"))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 0 or args.max_tokens < 1:
        parser.error("Invalid limit or token count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection.json"
    if args.mode == "generate" and selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("source_dir") != str(args.source_dir.resolve()):
            raise ValueError("Selection belongs to another source directory")
    else:
        selection = select(args.source_dir, spider_root=args.spider_root,
                           env_config=args.env_config, workers=args.workers)
        if (selection_path.exists()
                and json.loads(selection_path.read_text(encoding="utf-8")) != selection):
            raise ValueError("Existing selection differs; use a fresh output directory")
        _atomic_json(selection_path, selection)
    if args.mode == "select":
        print(json.dumps({"source_files": selection["source_files"],
                          "candidate_count": selection["candidate_count"],
                          "eligible_count": selection["eligible_count"],
                          "fork_reasons": selection["fork_reasons"],
                          "selection": str(selection_path.resolve())}))
        return
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY in the environment")
    from sql_agent.config import EnvConfig
    from sql_planner.collect import read_spider_train_tasks
    from sql_planner.collect_tagged import build_prompt
    from sql_planner.deepseek import DeepSeekClient
    from sql_planner.repair_tagged import repair_trajectory
    spider_root = args.spider_root.resolve()
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=spider_root)
    tasks = {task.task_id: task for task in read_spider_train_tasks(spider_root, limit=0, seed=42)}
    client = DeepSeekClient(api_key, model=args.model, base_url=args.base_url)
    generation_identity = {
        "schema_version": 1,
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "spider_source_sha256": hashlib.sha256(
            (spider_root / "train_spider.json").read_bytes()).hexdigest(),
        "env_config_sha256": hashlib.sha256(args.env_config.read_bytes()).hexdigest(),
        "model": args.model, "base_url": args.base_url,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
    }
    identity_path = args.output_dir / "generation_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != generation_identity:
            raise ValueError("Existing generation belongs to different settings")
    else:
        _atomic_json(identity_path, generation_identity)
    chosen_dir = args.output_dir / "pairs"
    attempts_dir = args.output_dir / "attempts"
    chosen_dir.mkdir(exist_ok=True)
    attempts_dir.mkdir(exist_ok=True)
    entries = [entry for entry in selection["entries"] if entry["eligible"]]
    entries = entries[:args.limit] if args.limit else entries
    def generate_one(entry: dict[str, Any]) -> str:
        name = entry["file"]
        pair_path = chosen_dir / name
        attempt_path = attempts_dir / name
        if pair_path.exists():
            return "skipped"
        source_path = args.source_dir / "trajectories" / name
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != entry["source_sha256"]:
            raise ValueError(f"Source changed: {source_path}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source["messages"][0]["content"] != build_prompt(config.max_turns):
            raise ValueError(f"System prompt mismatch: {source_path}")
        repaired = repair_trajectory(source, tasks[entry["task_id"]], config, client,
                                     cutoff_turn=entry["fork_turn"],
                                     temperature=args.temperature, max_tokens=args.max_tokens,
                                     response_retries=4)
        _atomic_json(attempt_path, repaired)
        if repaired["status"] not in {"repaired", "repaired_no_test_budget", "repaired_direct_submit"}:
            return "failed"
        _atomic_json(pair_path, make_pair(source, repaired, entry["fork_turn"],
                                          entry["fork_reason"]))
        return "created"

    created = skipped = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_one, entry) for entry in entries]
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            if outcome == "created":
                created += 1
            elif outcome == "skipped":
                skipped += 1
            else:
                failed += 1
            if (created + failed) % 25 == 0 and outcome != "skipped":
                print(f"processed={created + failed} pairs={created} failed={failed}", flush=True)
    expected = {entry["file"] for entry in selection["entries"] if entry["eligible"]}
    saved = {path.name for path in chosen_dir.glob("*.json")}
    if saved - expected:
        raise ValueError("Unexpected pair files in output directory")
    complete = saved == expected
    tested = sum(
        bool(json.loads(path.read_text(encoding="utf-8"))["chosen_tested_before_submit"])
        for path in chosen_dir.glob("*.json")
    )
    report = {"status": "completed" if complete else "partial",
              "source_wrong_trajectories": selection["candidate_count"],
              "eligible_pairs": len(expected),
              "blocked_gold": selection["candidate_count"] - len(expected),
              "saved_pairs": len(saved), "missing_pairs": len(expected - saved),
              "chosen_tested_before_submit": tested,
              "chosen_direct_submit": len(saved) - tested,
              "created_this_run": created, "skipped_this_run": skipped,
              "failed_this_run": failed, "fork_reasons": selection["fork_reasons"]}
    _atomic_json(args.output_dir / "generation_report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
