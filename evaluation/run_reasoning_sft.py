from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from sql_agent.action_parser import ActionParseError, parse_action
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.env import SQLAgentEnv
from sql_agent.truncation import canonical_json

RESPONSE = re.compile(r"\s*<reasoning>(.*?)</reasoning>\s*<tool>(.*?)</tool>\s*", re.DOTALL)
TOOLS = {"list_tables", "inspect_tables", "inspect_values", "execute_sql", "submit_sql"}


def parse_response(text: str) -> tuple[str, dict]:
    match = RESPONSE.fullmatch(text)
    if match is None or not match.group(1).strip():
        raise ValueError("Invalid reasoning/tool blocks")
    try:
        value = json.loads(match.group(2))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid tool JSON") from exc
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        raise ValueError("Tool JSON must contain exactly name and arguments")
    name, arguments = value["name"], value["arguments"]
    if name not in TOOLS:
        raise ValueError(f"Unavailable tool: {name}")
    if name == "submit_sql":
        if not isinstance(arguments, dict) or set(arguments) != {"sql"} or not isinstance(arguments["sql"], str) or not arguments["sql"].strip():
            raise ValueError("Invalid submit_sql arguments")
    else:
        try:
            parse_action({"tool": name, "arguments": arguments})
        except ActionParseError as exc:
            raise ValueError(str(exc)) from exc
    return name, arguments


def summary(records: list[dict], total: int) -> dict:
    by_difficulty = {}
    for difficulty in ("easy", "medium", "hard", "extra"):
        group = [r for r in records if r["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "count": len(group),
            "correct": sum(r["correct"] for r in group),
        }
    return {
        "total": total,
        "completed": len(records),
        "correct": sum(r["correct"] for r in records),
        "execution_accuracy": sum(r["correct"] for r in records) / len(records) if records else 0.0,
        "by_difficulty": by_difficulty,
        "status_counts": dict(Counter(r["status"] for r in records)),
    }


def evaluate_batch(tasks, llm, tokenizer, prompt: str, config: EnvConfig, max_tokens: int) -> list[dict]:
    states = []
    for task in tasks:
        env = SQLAgentEnv(config, reserve_final_submission=True, system_prompt=prompt)
        env.reset(task)
        env.history = env.history[:2]  # Collection starts with the question; no initial ready observation.
        states.append({
            "task": task,
            "env": env,
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": task.question}],
            "turns": [],
            "status": "missing_submission",
            "correct": False,
            "final_sql": None,
            "verification": None,
            "done": False,
        })
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=["</tool>"])
    for _ in range(config.max_turns + 1):
        active = [state for state in states if not state["done"]]
        if not active:
            break
        prompts = [
            tokenizer.apply_chat_template(state["messages"], tokenize=False, add_generation_prompt=True)
            for state in active
        ]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        for state, output in zip(active, outputs, strict=True):
            response = output.outputs[0].text
            if "<tool>" in response and "</tool>" not in response and output.outputs[0].finish_reason == "stop":
                response += "</tool>"
            turn = {"response": response}
            state["turns"].append(turn)
            try:
                name, arguments = parse_response(response)
                turn["tool"] = name
                turn["arguments"] = arguments
                if name == "submit_sql":
                    observation, _ = state["env"].submit_sql(arguments)
                    state["final_sql"] = arguments["sql"]
                    state["verification"] = observation.get("verification")
                    state["correct"] = bool((state["verification"] or {}).get("correct"))
                    state["status"] = "submitted_sql"
                    state["done"] = True
                    continue
                if state["env"].turn >= config.max_turns:
                    raise ValueError("Exploration budget exhausted")
                observation, done = state["env"].step({"tool": name, "arguments": arguments})
                turn["observation"] = observation
                state["messages"].extend([
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"<observation>{canonical_json(observation)}</observation>"},
                ])
                if done:
                    state["status"] = observation.get("termination_reason", "environment_done")
                    state["done"] = True
            except (ValueError, RuntimeError) as exc:
                turn["error"] = str(exc)
                state["status"] = "invalid_response"
                state["done"] = True
    results = []
    for state in states:
        state["env"].close()
        task = state["task"]
        results.append({
            "task_id": task.task_id,
            "db_id": task.db_id,
            "difficulty": task.difficulty,
            "status": state["status"],
            "correct": state["correct"],
            "final_sql": state["final_sql"],
            "verification": state["verification"],
            "turns": state["turns"],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/sql_planner/reasoning_gold_merged_sft"))
    parser.add_argument("--tasks", type=Path, default=Path("data/internal_validation.jsonl"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env_sql_planner_qwen25_coder_3b.yaml"))
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = EnvConfig.from_yaml(args.env_config)
    from dataclasses import replace
    config = replace(config, spider_root=args.spider_root.resolve())
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    prompt = manifest["prompt"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "trajectories"
    records_dir.mkdir(exist_ok=True)
    eval_manifest = {
        "model": str(args.model.resolve()),
        "dataset": str(args.tasks.resolve()),
        "dataset_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "spider_root": str(config.spider_root),
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "max_turns": config.max_turns,
        "temperature": 0.0,
        "scorer": "ExecutionVerifier",
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != eval_manifest:
        raise ValueError("Evaluation manifest differs; use another output directory")
    manifest_path.write_text(json.dumps(eval_manifest, indent=2) + "\n")
    pending = [t for t in tasks if not (records_dir / f"{t.task_id}.json").exists()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if pending:
        llm = LLM(model=str(args.model), dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.82, max_num_seqs=16)
        for start in range(0, len(pending), args.batch_size):
            for record in evaluate_batch(pending[start:start + args.batch_size], llm, tokenizer, prompt, config, args.max_tokens):
                (records_dir / f"{record['task_id']}.json").write_text(json.dumps(record, ensure_ascii=False) + "\n")
            records = [json.loads(path.read_text()) for path in records_dir.glob("*.json")]
            result = summary(records, len(tasks))
            (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    else:
        records = [json.loads(path.read_text()) for path in records_dir.glob("*.json")]
        result = summary(records, len(tasks))
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
