from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

from sql_agent.config import EnvConfig
from sql_agent.verifier import ExecutionVerifier

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SPIDER = ROOT.parent / "datasets/spider/spider_data"
ARTIFACTS = ROOT / "artifacts/sql_planner"
RUNS = {
    "paper_aligned": "sql_trail_paper_aligned_qwen25_coder_3b_base_external_dev_eval_v2",
    "direct": "qwen25_coder_3b_base_direct_full_schema_external_dev_eval",
}


def load_result_eq():
    source = OUTPUT / "official_exec_eval.py"
    names = {
        "permute_tuple",
        "unorder_row",
        "quick_rej",
        "multiset_eq",
        "get_constraint_permutation",
        "result_eq",
    }
    parsed = ast.parse(source.read_text())
    selected = ast.Module(
        body=[
            node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    namespace = {
        "Tuple": tuple,
        "Any": Any,
        "List": list,
        "Set": set,
        "defaultdict": defaultdict,
        "product": product,
        "random": random,
    }
    exec(compile(selected, str(source), "exec"), namespace)
    return namespace["result_eq"]


def main():
    parser = argparse.ArgumentParser(description="Rescore saved SQL without model inference")
    parser.add_argument("--multi-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    runs = dict(RUNS)
    output = arguments.output_dir or OUTPUT
    if arguments.multi_run is not None:
        if arguments.output_dir is None:
            parser.error("--multi-run requires a separate --output-dir")
        runs["paper_aligned"] = str(arguments.multi_run.resolve())
    output.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    result_eq = load_result_eq()
    task_path = ROOT / "data/external_dev.jsonl"
    tasks = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    original = json.loads((SPIDER / "dev.json").read_text())
    dataset_equal = len(tasks) == len(original) and all(
        (task["db_id"], task["question"], task["reference_sql"])
        == (item["db_id"], item["question"], item["query"])
        for task, item in zip(tasks, original, strict=True)
    )
    config = EnvConfig(spider_root=SPIDER)
    summary = {
        "dataset_exact_match_local_spider_dev": dataset_equal,
        "dataset_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "official_result_eq_sha256": hashlib.sha256(
            (OUTPUT / "official_exec_eval.py").read_bytes()
        ).hexdigest(),
        "scoring_scope": (
            "Original databases only; official result_eq comparator, not full test-suite "
            "evaluation. No value substitution or DISTINCT removal."
        ),
        "runs": {},
        "run_directories": runs,
    }
    details = []
    for name, folder in runs.items():
        counts = Counter()
        errors = Counter()
        finishes = Counter()
        per_task = {}
        for task in tasks:
            record = json.loads(
                (ARTIFACTS / folder / "trajectories" / f"{task['task_id']}.json").read_text()
            )
            verifier = ExecutionVerifier(SPIDER / task["db_path"], task["reference_sql"], config)
            expected = verifier._execute_full(task["reference_sql"])
            cache = {}

            def score(sql, verifier=verifier, expected=expected, cache=cache, task=task):
                if not sql:
                    return {"strict": False, "result_eq": False, "error": "missing_sql"}
                if sql not in cache:
                    verified = verifier.verify(sql)
                    equal = False
                    if verified.agent_sql_executable:
                        predicted = verifier._execute_full(sql)
                        equal = result_eq(
                            list(expected.rows),
                            list(predicted.rows),
                            "order by" in task["reference_sql"].lower(),
                        )
                    cache[sql] = {
                        "strict": verified.correct,
                        "result_eq": equal,
                        "error": verified.error,
                    }
                return cache[sql]

            final_sql = record.get("final_sql", record.get("sql"))
            final = score(final_sql)
            turns = record.get("turns", [])
            sql_calls = [
                turn.get("sql") or turn.get("arguments", {}).get("sql")
                for turn in turns
                if turn.get("sql") or turn.get("tool") == "execute_sql"
            ]
            intermediate = [score(sql) for sql in sql_calls]
            first = intermediate[0] if intermediate else final
            last = intermediate[-1] if intermediate else final
            any_correct = any(item["strict"] for item in intermediate)
            counts["total"] += 1
            counts["stored_correct"] += bool(record["correct"])
            counts["strict_final_correct"] += final["strict"]
            counts["official_result_eq_final_correct"] += final["result_eq"]
            counts["score_disagreement_stored"] += final["strict"] != bool(record["correct"])
            counts["first_sql_correct"] += first["strict"]
            counts["last_exploration_sql_correct"] += last["strict"]
            counts["any_intermediate_or_final_correct_oracle"] += any_correct or final["strict"]
            counts["intermediate_correct_final_wrong"] += any_correct and not final["strict"]
            counts["first_correct_final_wrong"] += first["strict"] and not final["strict"]
            counts["first_wrong_final_correct"] += not first["strict"] and final["strict"]
            counts["missing_final_sql"] += not bool(final_sql)
            counts["missing_final_with_correct_intermediate"] += not final_sql and any_correct
            counts["ten_or_more_sql_calls"] += len(sql_calls) >= 10
            counts["ten_identical_sql_calls"] += len(sql_calls) >= 10 and len(set(sql_calls)) == 1
            counts["turns"] += len(turns)
            counts["trajectories_with_length_finish"] += any(
                turn.get("finish_reason") == "length" for turn in turns
            )
            counts["trajectories_with_synthetic_observation"] += any(
                "<observation>" in turn.get("response", "") for turn in turns
            )
            counts["scorer_gained"] += final["result_eq"] and not final["strict"]
            counts["scorer_lost"] += final["strict"] and not final["result_eq"]
            errors[final["error"]] += 1
            finishes.update(turn.get("finish_reason") for turn in turns)
            detail = {
                "run": name,
                "task_id": task["task_id"],
                "db_id": task["db_id"],
                "question": task["question"],
                "reference_sql": task["reference_sql"],
                "final_sql": final_sql,
                "final": final,
                "sql_calls": len(sql_calls),
                "first": first,
                "any_intermediate_correct": any_correct,
                "correct_intermediate_sql": next(
                    (sql for sql in sql_calls if score(sql)["strict"]), None
                ),
            }
            details.append(detail)
            per_task[task["task_id"]] = final["strict"]
        summary["runs"][name] = {
            "counts": dict(counts),
            "errors": dict(errors),
            "finish_reasons": dict(finishes),
            "per_task": per_task,
        }
        print(name, dict(counts), flush=True)
    paper = summary["runs"]["paper_aligned"].pop("per_task")
    direct = summary["runs"]["direct"].pop("per_task")
    summary["paired"] = dict(
        Counter(f"direct_{direct[task_id]}_multi_{paper[task_id]}" for task_id in paper)
    )
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "audit_details.jsonl").write_text(
        "".join(json.dumps(detail) + "\n" for detail in details)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
