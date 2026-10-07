from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.env import SQLAgentEnv
from sql_agent.protocol import parse_response
from sql_agent.truncation import canonical_json


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


def force_final_submission(state: dict) -> None:
    sql = state["env"].last_executed_sql
    if sql is None:
        state["status"] = "missing_submission"
        state["done"] = True
        return
    arguments = {"sql": sql}
    observation, _ = state["env"].submit_sql(arguments)
    response = (
        "<reasoning>The exploration budget is exhausted; "
        "submit the last executed SQL.</reasoning>\n"
        f"<tool>{canonical_json({'name': 'submit_sql', 'arguments': arguments})}</tool>"
    )
    state["turns"].append(
        {
            "response": response,
            "tool": "submit_sql",
            "arguments": arguments,
            "observation": observation,
            "source": "evaluator_fallback",
            "reason": "exploration_budget_exhausted",
        }
    )
    state["final_sql"] = sql
    state["verification"] = observation.get("verification")
    state["correct"] = bool((state["verification"] or {}).get("correct"))
    state["status"] = "forced_submit_sql"
    state["forced_submission"] = True
    state["done"] = True


def with_remaining_rounds(content: str, exploration_remaining: int) -> str:
    reminder = (
        f"当前剩余轮数：{exploration_remaining + 1}（包含本轮和最终提交轮）；"
        f"剩余探索轮数：{exploration_remaining}。"
    )
    if exploration_remaining == 0:
        reminder += "\n这是最后一轮。请在此轮使用submit_sql工具提交SQL。"
    return f"{content}\n\n{reminder}"


def evaluate_batch(
    tasks,
    llm,
    tokenizer,
    prompt: str,
    config: EnvConfig,
    max_tokens: int,
    *,
    memory_contexts: dict[str, str] | None = None,
) -> list[dict]:
    from vllm import SamplingParams

    states = []
    for task in tasks:
        env = SQLAgentEnv(config, reserve_final_submission=True, system_prompt=prompt)
        env.reset(task)
        env.history = env.history[
            :2
        ]  # Collection starts with the question; no initial ready observation.
        question = task.question
        context = (memory_contexts or {}).get(task.task_id, "")
        if context:
            question += "\n\n" + context
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": with_remaining_rounds(question, config.max_turns)},
        ]
        states.append(
            {
                "task": task,
                "env": env,
                "messages": messages,
                "initial_messages": [dict(message) for message in messages],
                "turns": [],
                "status": "missing_submission",
                "correct": False,
                "final_sql": None,
                "verification": None,
                "forced_submission": False,
                "done": False,
            }
        )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=["</tool>"])
    for _ in range(config.max_turns + 1):
        active = [state for state in states if not state["done"]]
        if not active:
            break
        prompts = [
            tokenizer.apply_chat_template(
                state["messages"], tokenize=False, add_generation_prompt=True
            )
            for state in active
        ]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        for state, output in zip(active, outputs, strict=True):
            response = output.outputs[0].text
            if (
                "<tool>" in response
                and "</tool>" not in response
                and output.outputs[0].finish_reason == "stop"
            ):
                response += "</tool>"
            turn = {"response": response}
            state["turns"].append(turn)
            final_round = state["env"].turn >= config.max_turns
            try:
                name, arguments = parse_response(response)
                turn["tool"] = name
                turn["arguments"] = arguments
                if name == "submit_sql":
                    observation, _ = state["env"].submit_sql(arguments)
                    turn["observation"] = observation
                    state["final_sql"] = arguments["sql"]
                    state["verification"] = observation.get("verification")
                    state["correct"] = bool((state["verification"] or {}).get("correct"))
                    state["status"] = "submitted_sql"
                    state["done"] = True
                    continue
                if final_round:
                    raise ValueError("Exploration budget exhausted")
                observation, done = state["env"].step({"tool": name, "arguments": arguments})
                turn["observation"] = observation
                state["messages"].extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": with_remaining_rounds(
                                f"<observation>{canonical_json(observation)}</observation>",
                                config.max_turns - state["env"].turn,
                            ),
                        },
                    ]
                )
                if done:
                    state["status"] = observation.get("termination_reason", "environment_done")
                    state["done"] = True
            except (ValueError, RuntimeError) as exc:
                turn["error"] = str(exc)
                if final_round:
                    force_final_submission(state)
                    continue
                state["status"] = "invalid_response"
                state["done"] = True
    results = []
    for state in states:
        state["env"].close()
        task = state["task"]
        results.append(
            {
                "task_id": task.task_id,
                "db_id": task.db_id,
                "difficulty": task.difficulty,
                "status": state["status"],
                "correct": state["correct"],
                "final_sql": state["final_sql"],
                "verification": state["verification"],
                "forced_submission": state["forced_submission"],
                "turns": state["turns"],
                "initial_messages": state["initial_messages"],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/sft/dataset")
    )
    parser.add_argument("--tasks", type=Path, default=Path("data/internal_validation.jsonl"))
    parser.add_argument(
        "--env-config", type=Path, default=Path("configs/env.yaml")
    )
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--memory-db", type=Path)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--memory-top-k", type=int, default=10)
    parser.add_argument("--memory-max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.memory_top_k < 0 or min(
        args.memory_max_tokens, args.embedding_batch_size, args.embedding_max_length
    ) < 1:
        parser.error("Invalid memory retrieval limits")
    if args.memory_db and not args.memory_db.is_file():
        parser.error("Memory database does not exist")
    from transformers import AutoTokenizer
    from vllm import LLM

    config = EnvConfig.from_yaml(args.env_config)
    from dataclasses import replace

    config = replace(config, spider_root=args.spider_root.resolve())
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    prompt = manifest["prompt"]
    memory_contexts = None
    memory_snapshot = None
    if args.memory_db:
        from experience_memory.retrieve import EmbeddingEncoder, contexts_for_tasks
        from experience_memory.store import MemoryStore
        from sql_agent.truncation import TokenCounter

        memories = MemoryStore(args.memory_db).all()
        memory_snapshot = hashlib.sha256(
            json.dumps(memories, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        memory_contexts = contexts_for_tasks(
            tasks,
            memories,
            top_k=args.memory_top_k,
            max_tokens=args.memory_max_tokens,
            count_tokens=TokenCounter(config.tokenizer_path).count_text,
            encoder=EmbeddingEncoder(
                args.embedding_model, device=args.embedding_device,
                batch_size=args.embedding_batch_size, max_length=args.embedding_max_length,
            ),
        )
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
        "final_submission_policy": "model_submit_then_last_executed_sql_on_budget_exhaustion",
        "round_prompt_version": "remaining_rounds_v1",
    }
    if args.memory_db:
        eval_manifest["memory"] = {
            "snapshot_sha256": memory_snapshot,
            "top_k": args.memory_top_k,
            "max_tokens": args.memory_max_tokens,
            "retriever": "embedding_v1",
            "embedding_model": args.embedding_model,
            "embedding_device": args.embedding_device,
            "embedding_batch_size": args.embedding_batch_size,
            "embedding_max_length": args.embedding_max_length,
        }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != eval_manifest:
        raise ValueError("Evaluation manifest differs; use another output directory")
    manifest_path.write_text(json.dumps(eval_manifest, indent=2) + "\n")
    pending = [t for t in tasks if not (records_dir / f"{t.task_id}.json").exists()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if pending:
        llm = LLM(
            model=str(args.model),
            dtype="bfloat16",
            max_model_len=16384,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
        )
        for start in range(0, len(pending), args.batch_size):
            for record in evaluate_batch(
                pending[start : start + args.batch_size],
                llm,
                tokenizer,
                prompt,
                config,
                args.max_tokens,
                memory_contexts=memory_contexts,
            ):
                (records_dir / f"{record['task_id']}.json").write_text(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
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
