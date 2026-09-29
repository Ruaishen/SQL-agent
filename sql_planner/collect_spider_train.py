from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from sql_agent.config import EnvConfig
from sql_agent.truncation import canonical_json
from sql_planner.collect import (
    PROMPT_VERSION,
    api_tools,
    build_prompt,
    collect,
    ensure_run_manifest,
    read_spider_train_tasks,
)
from sql_planner.deepseek import DeepSeekClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect free-order DeepSeek trajectories for all 7,000 "
            "train_spider.json questions"
        )
    )
    parser.add_argument(
        "--spider-root", type=Path, default=Path("../datasets/spider/spider_data")
    )
    parser.add_argument(
        "--env-config", type=Path, default=Path("configs/env_sql_planner_qwen25_coder_3b.yaml")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sql_planner/spider_train_7000_multi_table_v11"),
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--limit", type=int, default=0, help="0 collects all 7,000 tasks")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.limit < 0 or args.samples_per_task < 1 or not 0 <= args.temperature <= 2:
        parser.error("Invalid limit, sample count, or temperature")
    if args.max_tokens < 1 or args.workers < 1 or args.progress_every < 0:
        parser.error("max-tokens/workers must be positive; progress-every must be non-negative")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY in the environment")

    spider_root = args.spider_root.expanduser().resolve()
    source_path = spider_root / "train_spider.json"
    if not source_path.is_file():
        parser.error(f"Spider training file does not exist: {source_path}")
    env_config = replace(EnvConfig.from_yaml(args.env_config), spider_root=spider_root)
    tasks = read_spider_train_tasks(spider_root, limit=args.limit, seed=args.seed)
    if args.limit == 0 and len(tasks) != 7_000:
        parser.error(f"Expected 7,000 train_spider.json tasks, found {len(tasks)}")

    ensure_run_manifest(
        args.output_dir,
        {
            "schema_version": 2,
            "prompt_version": PROMPT_VERSION,
            "exploration_budget": env_config.max_turns,
            "submission_budget": 1,
            "tool_calls_per_response": 1,
            "schema_tool": "inspect_tables",
            "prompt_sha256": hashlib.sha256(
                build_prompt(env_config.max_turns).encode("utf-8")
            ).hexdigest(),
            "source": "Spider train_spider.json only",
            "source_path": str(source_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "selected_task_count": len(tasks),
            "env_config_sha256": hashlib.sha256(args.env_config.read_bytes()).hexdigest(),
            "tools_sha256": hashlib.sha256(canonical_json(api_tools()).encode()).hexdigest(),
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "samples_per_task": args.samples_per_task,
            "seed": args.seed,
        },
    )
    client = DeepSeekClient(api_key, model=args.model, base_url=args.base_url)
    counts = collect(
        tasks,
        env_config,
        client,
        output_dir=args.output_dir,
        samples_per_task=args.samples_per_task,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
