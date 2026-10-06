"""Paired teacher comparison on a deterministic 100-task DPO sample.

Construct validated DPO pairs using the production exact-gold retry policy.
Chosen reasoning is preserved without hint sanitization for this audit.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, median

from sql_agent.config import EnvConfig
from sql_planner.collect import _atomic_json, read_spider_train_tasks
from sql_planner.deepseek import DeepSeekClient
from sql_planner.repair_tagged import GOLD_MENTION, repair_trajectory
from dpo.pairs import make_pair

VARIANTS = {
    "flash_nonthinking": ("deepseek-flash", False),
    "flash_high": ("deepseek-flash", True),
    "pro_high": ("deepseek-v4-pro", True),
}
TAG = re.compile(r"<reasoning>(.*?)(?:</reasoning>|$)", re.S)


class AuditClient(DeepSeekClient):
    def __init__(self, key: str, model: str, thinking: bool):
        super().__init__(key, model=model, timeout_seconds=240, max_retries=2)
        self.thinking = thinking
        self.calls: list[dict] = []

    def _request(self, payload: dict):
        payload = {**payload, "thinking": {"type": "enabled" if self.thinking else "disabled"}}
        if self.thinking:
            payload["reasoning_effort"] = "high"
            payload.pop("temperature", None)
        request_messages = json.loads(json.dumps(payload["messages"]))
        started = time.monotonic()
        result = super()._request(payload)
        self.calls.append({
            "request_id": result.request_id, "api_model": result.model,
            "finish_reason": result.finish_reason, "message": result.message,
            "usage": result.usage, "elapsed_seconds": time.monotonic() - started,
            "request_messages": request_messages,
        })
        return result


def sample_entries(selection: dict, n: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for entry in selection["entries"]:
        if entry["eligible"]:
            groups[entry["fork_reason"]].append(entry)
    total = sum(map(len, groups.values()))
    if not 1 <= n <= total:
        raise ValueError("Invalid sample size")
    quotas = {key: n * len(group) // total for key, group in groups.items()}
    ranked = sorted(groups, key=lambda key: (-(n * len(groups[key]) % total), key))
    for key in ranked[:n - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(seed)
    sample = []
    for key in sorted(groups):
        sample.extend(rng.sample(sorted(groups[key], key=lambda x: x["file"]), quotas[key]))
    rng.shuffle(sample)
    return sample


def metrics(record: dict) -> dict:
    calls = record["api_calls"]
    usage = Counter()
    visible = []
    private_leaks = 0
    for call in calls:
        u = call["usage"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens",
                    "prompt_cache_miss_tokens"):
            usage[key] += int(u.get(key, 0))
        usage["reasoning_tokens"] += int((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
        text = call["message"].get("content") or ""
        matches = TAG.findall(text)
        visible.append(any(GOLD_MENTION.search(reasoning) for reasoning in matches))
        private_leaks += bool(GOLD_MENTION.search(call["message"].get("reasoning_content") or ""))
    repaired = record["repair"]
    pair = record.get("pair")
    chosen_leaks = [] if pair is None else [
        bool(GOLD_MENTION.search(turn["reasoning"])) for turn in pair["chosen_turns"]
    ]
    return {
        "constructed_pair": pair is not None,
        "correct": bool((repaired.get("verification") or {}).get("correct")),
        "exact_gold_submission": record["exact_gold_submission"],
        "visible_hint_leak": any(chosen_leaks),
        "chosen_leaking_turns": sum(chosen_leaks),
        "raw_call_hint_leak": any(visible),
        "first_response_hint_leak": bool(visible and visible[0]),
        "visible_leaking_calls": sum(visible), "private_thinking_leaking_calls": private_leaks,
        "tested_before_submit": repaired["status"] == "repaired",
        "status": repaired["status"], "api_calls": len(calls), "usage": dict(usage),
        "api_seconds": sum(call["elapsed_seconds"] for call in calls),
        "elapsed_seconds": record["elapsed_seconds"],
        "api_models": sorted({call["api_model"] for call in calls if call["api_model"]}),
    }


def summarize(output: Path, sample: list[dict], variant_names=None) -> dict:
    variants = {}
    records = {}
    for name in (variant_names or VARIANTS):
        rows = []
        for entry in sample:
            path = output / name / entry["file"]
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))["metrics"]
                rows.append(row)
                records[name, entry["task_id"]] = row
        if not rows:
            variants[name] = {"completed": 0}
            continue
        totals = Counter()
        for row in rows:
            totals.update(row["usage"])
        variants[name] = {
            "completed": len(rows),
            **{key: sum(row[key] for row in rows) for key in (
                "correct", "constructed_pair", "chosen_leaking_turns", "exact_gold_submission", "visible_hint_leak", "first_response_hint_leak",
                "tested_before_submit", "api_calls", "visible_leaking_calls", "private_thinking_leaking_calls")},
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "usage": dict(totals),
            "mean_case_seconds": mean(row["elapsed_seconds"] for row in rows),
            "median_case_seconds": median(row["elapsed_seconds"] for row in rows),
            "api_models": sorted({model for row in rows for model in row["api_models"]}),
        }
        hit = totals["prompt_cache_hit_tokens"]
        miss = totals["prompt_tokens"] - hit
        out = totals["completion_tokens"]
        rates = (0.15, 4.5, 13.5) if name == "pro_high" else (0.02, 1, 4)
        cost = (hit * rates[0] + miss * rates[1] + out * rates[2]) / 1e6
        variants[name]["estimated_cny_off_peak"] = cost
        variants[name]["estimated_cny_peak"] = 2 * cost
        variants[name]["chosen_hint_leak_rate"] = variants[name]["visible_hint_leak"] / variants[name]["constructed_pair"] if variants[name]["constructed_pair"] else None
    paired = {}
    for a, b in (("flash_nonthinking", "flash_high"), ("flash_nonthinking", "pro_high"), ("flash_high", "pro_high")):
        comparisons = {}
        for metric in ("correct", "visible_hint_leak"):
            counts = Counter()
            for entry in sample:
                ra, rb = records.get((a, entry["task_id"])), records.get((b, entry["task_id"]))
                if ra is not None and rb is not None:
                    counts[f"{int(ra[metric])}{int(rb[metric])}"] += 1
            discordant = counts["01"] + counts["10"]
            p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(counts["01"], counts["10"]) + 1)) / 2**discordant) if discordant else 1.0
            comparisons[metric] = {"paired_counts_a_b": dict(counts), "mcnemar_exact_p": p}
        paired[f"{a}_vs_{b}"] = comparisons
    report = {"status": "completed" if all(v["completed"] == len(sample) for v in variants.values()) else "partial",
              "sample_size": len(sample), "variants": variants, "paired_comparisons": paired,
              "protocol": "Same gold hint and fork; production exact-gold requirement and 4 response retries; up to 3 trajectory attempts. Validated DPO chosen_messages are audited without reasoning edits. Private thinking is excluded from the chosen hint metric.",
              "cost_note": "Estimates from published cache/input/output rates, not the account bill."}
    _atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sql_planner/dpo_chosen_hint_compare_100_v2"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--sample-manifest", type=Path)
    args = parser.parse_args()
    root = Path("artifacts/sql_planner/reasoning")
    selection_path = Path("artifacts/sql_planner/reasoning_dpo_all_errors_v1/selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sample = sample_entries(selection, 100, 42)
    if args.sample_manifest:
        previous_sample = json.loads(args.sample_manifest.read_text(encoding="utf-8"))["sample"]
        if sample != previous_sample:
            raise ValueError("Previous experiment sample differs from the current sample")
        sample = previous_sample
    config_path = Path("configs/env_sql_planner_local_multi_table_v11.yaml")
    spider_root = Path("../datasets/spider/spider_data").resolve()
    config = replace(EnvConfig.from_yaml(config_path), spider_root=spider_root)
    tasks = {task.task_id: task for task in read_spider_train_tasks(spider_root, limit=0, seed=42)}
    identity = {
        "protocol_version": 3, "seed": 42, "sample_size": 100, "max_tokens": 16384,
        "prompt_protocol": "ordinary system; gold and grounding constraints only in post-fork user hint; retries do not repeat gold",
        "variants": {name: VARIANTS[name] for name in args.variants}, "temperature_nonthinking": 0.3, "response_retries": 4,
        "trajectory_attempts": 3, "require_exact_gold_submission": True,
        "sanitize_hint_reasoning": False,
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repair_code_sha256": hashlib.sha256(Path("sql_planner/repair_tagged.py").read_bytes()).hexdigest(),
        "sample": sample, "fork_counts": dict(Counter(e["fork_reason"] for e in sample)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(identity)):
        raise ValueError("Experiment identity changed; use a fresh output directory")
    _atomic_json(manifest, identity)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("Missing DEEPSEEK_API_KEY")

    def run(entry, name):
        directory = args.output_dir / name
        directory.mkdir(exist_ok=True)
        dest = directory / entry["file"]
        if dest.exists():
            return "skipped"
        source_bytes = (root / "trajectories" / entry["file"]).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != entry["source_sha256"]:
            raise ValueError("Source changed")
        source = json.loads(source_bytes)
        model, thinking = VARIANTS[name]
        client = AuditClient(key, model, thinking)
        started = time.monotonic()
        attempts = []
        pair = None
        for attempt_index in range(3):
            repaired = repair_trajectory(source, tasks[entry["task_id"]], config, client,
                cutoff_turn=entry["fork_turn"], temperature=0.3, max_tokens=16384,
                response_retries=4, sanitize_hint_reasoning=False, require_exact_gold_submission=True)
            attempts.append(repaired)
            if repaired["status"] in {"repaired", "repaired_direct_submit", "repaired_no_test_budget"}:
                pair = make_pair(source, repaired, entry["fork_turn"], entry["fork_reason"])
                pair_directory = directory / "pairs"
                pair_directory.mkdir(exist_ok=True)
                _atomic_json(pair_directory / entry["file"], pair)
                break
        record = {"task_id": entry["task_id"], "variant": name, "fork_reason": entry["fork_reason"],
                  "repair": repaired, "repair_attempts": attempts, "pair": pair, "api_calls": client.calls,
                  "exact_gold_submission": repaired.get("final_sql") == tasks[entry["task_id"]].reference_sql,
                  "elapsed_seconds": time.monotonic() - started}
        record["metrics"] = metrics(record)
        _atomic_json(dest, record)
        return f"{name} correct={int(record['metrics']['correct'])} leak={int(record['metrics']['visible_hint_leak'])} status={repaired['status']}"

    jobs = [(entry, name) for entry in (sample[:1] if args.preflight else sample) for name in args.variants]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            if args.preflight or index % 10 == 0:
                print(f"completed={index}/{len(jobs)} {result}", flush=True)
    report = summarize(args.output_dir, sample, args.variants)
    print(json.dumps({"status": report["status"], "completed": {k: v["completed"] for k, v in report["variants"].items()}}), flush=True)


if __name__ == "__main__":
    main()
