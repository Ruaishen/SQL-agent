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
    execute_sql_memory=None,
    execute_sql_memory_requires_success=True,
    execute_sql_memory_start_index=1,
    execute_sql_memory_repeat=False,
    record_token_ids=False,
) -> list[dict]:
    from vllm import SamplingParams

    if memory_contexts and execute_sql_memory is not None:
        raise ValueError("First-SQL retrieval cannot use initial question memories")
    if execute_sql_memory_start_index < 1:
        raise ValueError("SQL retrieval start index must be positive")
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
                "memory_retrieval": None,
                "memory_retrievals": [],
                "execute_sql_count": 0,
                "memory_message_restore": None,
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
        pending_retrieval = []
        for state, output in zip(active, outputs, strict=True):
            response = output.outputs[0].text
            if (
                "<tool>" in response
                and "</tool>" not in response
                and output.outputs[0].finish_reason == "stop"
            ):
                response += "</tool>"
            turn = {"response": response}
            if record_token_ids:
                turn["prompt_token_ids"] = list(output.prompt_token_ids)
                turn["generated_token_ids"] = list(output.outputs[0].token_ids)
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
                if name == "execute_sql":
                    state["execute_sql_count"] += 1
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
                if (name == "execute_sql" and execute_sql_memory is not None
                        and (not execute_sql_memory_requires_success
                             or observation.get("status") == "success")
                        and state["execute_sql_count"] >= execute_sql_memory_start_index
                        and (execute_sql_memory_repeat or state["memory_retrieval"] is None)
                        and not done):
                    pending_retrieval.append((state, turn, arguments["sql"]))
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
        if pending_retrieval:
            payloads = execute_sql_memory([sql for _, _, sql in pending_retrieval])
            if len(payloads) != len(pending_retrieval):
                raise ValueError("SQL retrieval batch does not match requested queries")
            for (state, turn, sql), payload in zip(pending_retrieval, payloads, strict=True):
                if payload.get("query_sql") != sql:
                    raise ValueError("Retrieved memory belongs to another SQL")
                state["memory_retrieval"] = payload
                turn["memory_retrieval"] = payload
                if execute_sql_memory_repeat:
                    state["memory_retrievals"].append({
                        "execute_index": state["execute_sql_count"],
                        "turn_index": len(state["turns"]) - 1, **payload,
                    })
                    if state["memory_message_restore"] is not None:
                        previous_message, original_content = state["memory_message_restore"]
                        previous_message["content"] = original_content
                    state["memory_message_restore"] = (
                        state["messages"][-1], state["messages"][-1]["content"]
                    )
                content = f"<observation>{canonical_json(turn['observation'])}</observation>"
                if payload["context"]:
                    content += "\n\n" + payload["context"]
                state["messages"][-1]["content"] = with_remaining_rounds(
                    content, config.max_turns - state["env"].turn
                )
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
                **({"memory_retrieval": state["memory_retrieval"]}
                   if execute_sql_memory is not None else {}),
                **({"memory_retrievals": state["memory_retrievals"]}
                   if execute_sql_memory is not None and execute_sql_memory_repeat else {}),
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
    parser.add_argument("--record-token-ids", action="store_true")
    parser.add_argument("--memory-db", type=Path)
    parser.add_argument("--memory-retriever", choices=("embedding", "embedding_experience", "tfidf",
                                                      "embedding_first_successful_sql", "embedding_first_sql",
                                                      "embedding_second_sql_every_turn", "tfidf_first_sql", "bm25_first_sql"),
                        default="embedding")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--memory-sql-field", choices=("sql_before", "sql_first_execute"),
                        default="sql_before")
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--memory-max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.memory_top_k < 0 or min(
        args.memory_max_tokens, args.embedding_batch_size, args.embedding_max_length
    ) < 1:
        parser.error("Invalid memory retrieval limits")
    if args.memory_sql_field != "sql_before" and args.memory_retriever not in (
        "embedding_first_successful_sql", "embedding_first_sql",
        "embedding_second_sql_every_turn", "tfidf_first_sql", "bm25_first_sql",
    ):
        parser.error("--memory-sql-field requires an execute-SQL retrieval mode")
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
    execute_sql_memory = None
    if args.memory_db:
        from experience_memory.retrieve import (
            SQL_QUERY_INSTRUCTION, EmbeddingEncoder, SqlMemoryRetriever, contexts_for_tasks,
        )
        from experience_memory.store import MemoryStore
        from sql_agent.truncation import TokenCounter

        memories = MemoryStore(args.memory_db).all()
        memory_snapshot = hashlib.sha256(
            json.dumps(memories, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        if args.memory_retriever in ("embedding_first_successful_sql", "embedding_first_sql",
                                     "embedding_second_sql_every_turn", "tfidf_first_sql", "bm25_first_sql"):
            execute_sql_memory = SqlMemoryRetriever(
                memories,
                EmbeddingEncoder(args.embedding_model, device=args.embedding_device,
                                 batch_size=args.embedding_batch_size,
                                 max_length=args.embedding_max_length,
                                 query_instruction=SQL_QUERY_INSTRUCTION)
                if args.memory_retriever not in ("tfidf_first_sql", "bm25_first_sql") else None,
                top_k=args.memory_top_k, max_tokens=args.memory_max_tokens,
                text_field=args.memory_sql_field,
                count_tokens=TokenCounter(config.tokenizer_path).count_text,
                retriever=(args.memory_retriever.split("_", 1)[0]
                           if args.memory_retriever in ("tfidf_first_sql", "bm25_first_sql")
                           else "embedding"),
            )
            execute_sql_memory.prepare()
        else:
            memory_contexts = contexts_for_tasks(
                tasks,
                memories,
                top_k=args.memory_top_k,
                max_tokens=args.memory_max_tokens,
                count_tokens=TokenCounter(config.tokenizer_path).count_text,
                retriever=args.memory_retriever,
                encoder=EmbeddingEncoder(
                    args.embedding_model, device=args.embedding_device,
                    batch_size=args.embedding_batch_size, max_length=args.embedding_max_length,
                ) if args.memory_retriever != "tfidf" else None,
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
        "max_context_tokens": config.max_context_tokens,
        "max_model_len": config.max_context_tokens,
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
        }
        if args.memory_retriever not in ("tfidf", "tfidf_first_sql", "bm25_first_sql"):
            eval_manifest["memory"].update({
                "retriever": ("embedding_question_v1" if args.memory_retriever == "embedding"
                              else "embedding_v1"),
                "embedding_model": args.embedding_model,
                "embedding_device": args.embedding_device,
                "embedding_batch_size": args.embedding_batch_size,
                "embedding_max_length": args.embedding_max_length,
            })
            if args.memory_retriever in ("embedding_first_successful_sql", "embedding_first_sql",
                                         "embedding_second_sql_every_turn"):
                eval_manifest["memory"].update({
                    "retriever": args.memory_retriever + "_v1",
                    "text_field": args.memory_sql_field,
                    "trigger": ("after_first_execute_sql" if args.memory_retriever == "embedding_first_sql"
                                else "after_first_execute_sql_with_status_success"),
                    "query_instruction": SQL_QUERY_INSTRUCTION,
                    "retrievals_per_task": 1,
                    "initial_memory_context": False,
                })
                if args.memory_retriever == "embedding_second_sql_every_turn":
                    eval_manifest["memory"].update({
                        "trigger": "after_every_execute_sql_from_second_call_including_errors",
                        "start_execute_index": 2,
                        "retrievals_per_task": "every_execute_sql_from_second",
                        "previous_memory_context": "replace_with_latest_retrieval",
                    })
        else:
            eval_manifest["memory"].update({
                "retriever": "tfidf_v1",
                "tokenizer": "lowercase_ascii_words_cjk_characters",
                "ngram_range": [1, 1],
                "tf": "raw_count",
                "idf": "1 + log((1 + memory_count) / (1 + document_frequency))",
                "normalization": "l2",
                "fit_corpus": (f"frozen_memory_{args.memory_sql_field}_only"
                               if args.memory_retriever in ("tfidf_first_sql", "bm25_first_sql")
                               else "frozen_memory_experience_only"),
                "positive_scores_only": True,
            })
            if args.memory_retriever in ("tfidf_first_sql", "bm25_first_sql"):
                eval_manifest["memory"].update({
                    "retriever": args.memory_retriever + "_v1",
                    "text_field": args.memory_sql_field,
                    "trigger": "after_first_execute_sql",
                    "retrievals_per_task": 1,
                    "initial_memory_context": False,
                })
        if args.memory_retriever == "bm25_first_sql":
            # BM25 scores are unnormalized and use term saturation/length correction.
            for key in ("tf", "idf", "normalization"):
                eval_manifest["memory"].pop(key, None)
            eval_manifest["memory"].update({
                "k1": 1.5,
                "b": 0.75,
                "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
                "tf": "bm25_saturation",
                "query_tf": "unique_terms",
                "normalization": "bm25_document_length",
                "tie_break": "memory_insertion_order",
            })
        if execute_sql_memory is not None and args.memory_sql_field == "sql_first_execute":
            eval_manifest["memory"].update({
                "indexed_memory_count": execute_sql_memory.indexed_memory_count,
                "excluded_missing_sql_count": execute_sql_memory.excluded_missing_sql_count,
                "missing_sql_policy": "exclude_without_final_sql_fallback",
            })
    if args.record_token_ids:
        eval_manifest["record_token_ids"] = True
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != eval_manifest:
        raise ValueError("Evaluation manifest differs; use another output directory")
    manifest_path.write_text(json.dumps(eval_manifest, indent=2) + "\n")
    if args.memory_db:
        snapshot_path = args.output_dir / "memory_snapshot.json"
        if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != memories:
            raise ValueError("Evaluation memory snapshot differs; use another output directory")
        snapshot_path.write_text(json.dumps(memories, ensure_ascii=False, indent=2) + "\n")
    pending = [t for t in tasks if not (records_dir / f"{t.task_id}.json").exists()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if pending:
        llm = LLM(
            model=str(args.model),
            dtype="bfloat16",
            max_model_len=config.max_context_tokens,
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
                execute_sql_memory=execute_sql_memory,
                execute_sql_memory_requires_success=args.memory_retriever == "embedding_first_successful_sql",
                execute_sql_memory_start_index=2 if args.memory_retriever == "embedding_second_sql_every_turn" else 1,
                execute_sql_memory_repeat=args.memory_retriever == "embedding_second_sql_every_turn",
                record_token_ids=args.record_token_ids,
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
