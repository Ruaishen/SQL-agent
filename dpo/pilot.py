"""Fixed random sample of original failures; one SFT regeneration attempt each."""
from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from dpo.pairs import build
from experience_memory.evolve import read_json, write_json
from sft.regenerate_reasoning import PROMPTS, regenerate
from sql_agent.config import EnvConfig
from sql_agent.deepseek import DeepSeekClient
from sql_agent.models import TaskRecord


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh pilot output directory")
    rows = []
    for name in ("train_spider.json", "train_others.json"):
        rows.extend(read_json(args.spider_root / name))
    candidates = []
    total = 0
    for path in sorted((args.source / "trajectories").glob("*.json")):
        record = read_json(path)
        total += 1
        if record.get("correct") is False and record.get("split") == "train":
            candidates.append((path, record))
    if len(candidates) < args.count:
        raise ValueError("Too few failed training trajectories")
    selected = random.Random(args.seed).sample(candidates, args.count)
    tasks = []
    entries = []
    for path, record in selected:
        task_id = record["task_id"]
        number = int(task_id.removeprefix("spider_train_"))
        row = rows[number]
        if row["question"] != record["question"] or row["db_id"] != record["db_id"]:
            raise ValueError(f"Spider mapping mismatch: {task_id}")
        task = TaskRecord(task_id=task_id, source="spider", db_id=row["db_id"],
                          db_path=f"database/{row['db_id']}/{row['db_id']}.sqlite",
                          question=row["question"], reference_sql=row["query"],
                          difficulty=record["difficulty"], split="train")
        if not task.resolve_db_path(args.spider_root).is_file():
            raise FileNotFoundError(task.resolve_db_path(args.spider_root))
        tasks.append(task.to_dict())
        entries.append({"file": path.name, "task_id": task_id, "source_sha256": sha256(path),
                        "difficulty": task.difficulty, "status": record["status"],
                        "max_turns": record["max_turns"]})
    if len({entry["task_id"] for entry in entries}) != args.count:
        raise ValueError("Duplicate sampled tasks")
    args.output.mkdir(parents=True)
    original = args.output / "original" / "trajectories"
    original.mkdir(parents=True)
    for (path, _), entry in zip(selected, entries):
        shutil.copyfile(path, original / entry["file"])
    task_path = args.output / "tasks.jsonl"
    task_path.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=args.spider_root.resolve(),
                     tokenizer_path=args.tokenizer.resolve(), processed_data_root=args.output.resolve())
    config_values = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    manifest = {"seed": args.seed, "count": args.count, "attempts_per_task": 1,
                "source": str(args.source.resolve()), "source_total": total,
                "wrong_train_candidates": len(candidates), "sample": entries,
                "config": config_values, "tasks_sha256": sha256(task_path),
                "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
                "prompt_sha256": {p.name: sha256(p) for p in sorted(PROMPTS.glob("*.md"))},
                "implementation_sha256": sha256(Path(__file__).parents[1] / "sft" / "regenerate_reasoning.py")}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"sampled": args.count, "wrong_candidates": len(candidates), "seed": args.seed,
                      "difficulty": dict(Counter(e["difficulty"] for e in entries))}), flush=True)


class RecordingClient(DeepSeekClient):
    def __init__(self, key, model):
        super().__init__(key, model=model, timeout_seconds=240, max_retries=2)
        self.calls = []

    def _request(self, payload):
        start = time.monotonic()
        result = super()._request(payload)
        self.calls.append({"message": result.message, "usage": result.usage,
                           "finish_reason": result.finish_reason, "model": result.model,
                           "elapsed_seconds": time.monotonic() - start})
        return result


def run(args):
    manifest = read_json(args.output / "manifest.json")
    config_data = manifest["config"]
    for field in ("spider_root", "tokenizer_path", "processed_data_root"):
        config_data[field] = Path(config_data[field])
    config = EnvConfig(**config_data)
    if manifest["tasks_sha256"] != sha256(args.output / "tasks.jsonl") or manifest["tokenizer_sha256"] != sha256(config.tokenizer_path / "tokenizer.json"):
        raise ValueError("Pilot tasks/tokenizer changed")
    if manifest["prompt_sha256"] != {p.name: sha256(p) for p in sorted(PROMPTS.glob("*.md"))} or manifest["implementation_sha256"] != sha256(Path(__file__).parents[1] / "sft" / "regenerate_reasoning.py"):
        raise ValueError("Pilot prompts/implementation changed")
    tasks = [TaskRecord.from_dict(json.loads(line)) for line in (args.output / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    registry = {task.task_id: task for task in tasks}
    key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("API key (hidden): ")
    # Read-only API check before scheduling paid generation requests.
    request = urllib.request.Request("https://api.deepseek.com/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        models = {row["id"] for row in json.load(response)["data"]}
    model = args.model or next((m for m in ("deepseek-flash", "deepseek-v4-flash", "deepseek-reasoner") if m in models), None)
    if model not in models:
        raise ValueError("Requested generation model unavailable")
    write_json(args.output / "run_identity.json", {"model": model, "workers": args.workers,
                "thinking": args.generation_mode, "audit_thinking": "enabled",
                "reasoning_effort": "high" if args.generation_mode == "enabled" else None,
                "temperature": args.temperature if args.generation_mode == "disabled" else None,
                "max_tokens": args.max_tokens})
    print(json.dumps({"model": model, "scheduled": len(tasks), "workers": args.workers}), flush=True)
    outcomes = []

    def one(entry):
        path = args.output / "original" / "trajectories" / entry["file"]
        if sha256(path) != entry["source_sha256"]:
            raise ValueError("Sample source changed")
        result_path = args.output / "results" / f"{entry['task_id']}.json"
        if result_path.exists():
            return read_json(result_path)
        # Persist before API use: interruptions never silently grant extra attempts.
        attempt_path = args.output / "attempts" / f"{entry['task_id']}.json"
        if attempt_path.exists():
            return {"task_id": entry["task_id"], "outcome": "interrupted_attempt"}
        write_json(attempt_path, {"status": "started", "task_id": entry["task_id"]})
        client = RecordingClient(key, model)
        start = time.monotonic()
        try:
            record = regenerate(registry[entry["task_id"]], read_json(path), replace(config, max_turns=entry["max_turns"]), client, max_tokens=args.max_tokens,
                                thinking=args.generation_mode == "enabled", temperature=args.temperature)
            outcome = "accepted" if record["sft_eligible"] else "audit_rejected"
            if record["sft_eligible"]:
                public = {k: v for k, v in record.items() if not k.startswith("private_")}
                write_json(args.output / "regenerated" / "trajectories" / entry["file"], public)
            error = None
        except Exception as exc:
            record = getattr(exc, "record", {})
            error = str(exc).replace(key, "[redacted]")
            outcome = "generation_or_verification_failed"
        usage = Counter()
        for call in client.calls:
            for name in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens"):
                usage[name] += int(call["usage"].get(name, 0))
        write_json(attempt_path, {"status": "completed", "task_id": entry["task_id"],
                   "outcome": outcome, "error": error, "record": record, "private_api_calls": client.calls})
        result = {"task_id": entry["task_id"], "outcome": outcome, "error": error,
                  "api_calls": len(client.calls), "usage": dict(usage),
                  "elapsed_seconds": time.monotonic() - start,
                  "trajectory_comparison": record.get("trajectory_comparison")}
        write_json(result_path, result)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, entry) for entry in manifest["sample"]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            outcomes.append(result)
            counts = dict(Counter(r["outcome"] for r in outcomes))
            write_json(args.output / "progress.json", {"completed": len(outcomes), "total": len(tasks), "counts": counts})
            print(json.dumps({"completed": len(outcomes), "total": len(tasks), "counts": counts,
                              "last_task": result["task_id"], "last_outcome": result["outcome"]}), flush=True)
    pairs_path = args.output / "pairs.jsonl"
    accepted = sum(r["outcome"] == "accepted" for r in outcomes)
    paired = {"pairs": 0, "skipped": []}
    if accepted:
        if pairs_path.exists():
            paired = read_json(pairs_path.with_suffix(".report.json"))
        else:
            paired = build(args.output / "original", args.output / "regenerated", tasks, config, pairs_path, workers=args.workers)
    usage = Counter()
    for result in outcomes:
        usage.update(result.get("usage", {}))
    report = {"status": "completed", "sample_size": len(tasks), "attempts_per_task": 1,
              "model": model, "seed": manifest["seed"], "counts": dict(Counter(r["outcome"] for r in outcomes)),
              "accepted_audit": accepted, "retained_dpo_pairs": paired["pairs"],
              "retention_rate": paired["pairs"] / len(tasks), "pairing_skipped": paired["skipped"],
              "usage": dict(usage)}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--source", type=Path, default=Path("data/reasoning"))
    parser.add_argument("--spider-root", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--model")
    parser.add_argument("--generation-mode", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()
    if min(args.count, args.workers, args.max_tokens) < 1:
        parser.error("Parameters must be positive")
    if args.mode == "prepare" and (args.spider_root is None or args.tokenizer is None):
        parser.error("Preparation requires Spider root and tokenizer")
    (prepare if args.mode == "prepare" else run)(args)


if __name__ == "__main__":
    main()
