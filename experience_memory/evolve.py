from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from evaluation.schema import render_schema
from experience_memory.reflect import reflect
from experience_memory.retrieve import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingEncoder,
    contexts_for_tasks,
)
from experience_memory.store import MemoryStore
from sql_agent.config import EnvConfig
from sql_agent.data import load_tasks
from sql_agent.deepseek import DeepSeekClient
from sql_agent.truncation import TokenCounter
from sql_agent.verifier import ExecutionVerifier


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_manifest(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Run configuration differs; use another output directory")
    else:
        write_json(path, value)


def run_round(
    *,
    tasks,
    student,
    teacher,
    config: EnvConfig,
    store: MemoryStore,
    output_dir: Path,
    top_k: int = 5,
    memory_max_tokens: int = 8192,
    teacher_max_tokens: int = 1024,
    batch_size: int = 16,
    embedding_encoder=None,
) -> dict:
    """student(tasks, contexts) starts fresh episodes with the fixed checkpoint."""
    if batch_size < 1 or top_k < 0 or memory_max_tokens < 1:
        raise ValueError("Invalid round limits")
    if any(task.split != "train" for task in tasks):
        raise ValueError("Memory construction accepts only train tasks")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Duplicate tasks")
    embedding_encoder = embedding_encoder or EmbeddingEncoder()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(
        output_dir / "round_manifest.json",
        {
            "version": 1,
            "tasks": [task.to_dict() for task in tasks],
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "memory_db": str(store.path.resolve()),
            "top_k": top_k,
            "memory_max_tokens": memory_max_tokens,
            "teacher_max_tokens": teacher_max_tokens,
            "retriever": "embedding_question_v1",
            "embedding_model": embedding_encoder.model_name,
            "embedding_device": embedding_encoder.device,
            "embedding_max_length": embedding_encoder.max_length,
            "embedding_batch_size": embedding_encoder.batch_size,
        },
    )
    snapshot_path = output_dir / "memory_snapshot.json"
    if not snapshot_path.exists():
        write_json(snapshot_path, store.all())
    snapshot = read_json(snapshot_path)
    current_ids = {record["memory_id"]: record for record in store.all()}
    if any(current_ids.get(record["memory_id"]) != record for record in snapshot):
        raise ValueError("Frozen snapshot is not present in the append-only memory store")
    counter = TokenCounter(config.tokenizer_path)
    contexts = contexts_for_tasks(
        tasks,
        snapshot,
        top_k=top_k,
        max_tokens=memory_max_tokens,
        count_tokens=counter.count_text,
        encoder=embedding_encoder,
    )

    def paths(task):
        key = hashlib.sha256(task.task_id.encode()).hexdigest()
        return (
            output_dir / "initial" / f"{key}.json",
            output_dir / "teacher" / f"{key}.json",
            output_dir / "retry" / f"{key}.json",
            output_dir / "errors" / f"{key}.json",
        )

    def save_student_batch(batch, batch_contexts, path_index):
        records = student(batch, batch_contexts)
        expected = {task.task_id: task for task in batch}
        if len(records) != len(batch) or {r.get("task_id") for r in records} != set(expected):
            raise ValueError("Student batch results do not match requested tasks")
        for record in records:
            write_json(paths(expected[record["task_id"]])[path_index], record)

    pending = [task for task in tasks if not paths(task)[0].exists()]
    for start in range(0, len(pending), batch_size):
        save_student_batch(pending[start : start + batch_size], contexts, 0)
    counts = {
        "tasks": len(tasks),
        "initial_correct": 0,
        "eligible_failures": 0,
        "retry_correct": 0,
        "appended": 0,
        "already_present": 0,
        "skipped": 0,
        "errors": 0,
    }
    accepted = []
    for task in tasks:
        initial_path, teacher_path, retry_path, error_path = paths(task)
        initial = read_json(initial_path)
        if initial.get("task_id") != task.task_id:
            raise ValueError("Initial record belongs to another task")
        if initial.get("correct") is True:
            counts["initial_correct"] += 1
            continue
        if initial.get("correct") is not False or not initial.get("final_sql"):
            counts["skipped"] += 1
            continue
        verifier = ExecutionVerifier(
            task.resolve_db_path(config.spider_root), task.reference_sql, config
        )
        try:
            gold = verifier.verify(task.reference_sql)
            failed = verifier.verify(initial["final_sql"])
            if (
                not gold.correct
                or failed.correct
                or failed.error in {"timeout", "result_too_large"}
                or (failed.error or "").startswith("verifier_reference_")
            ):
                counts["skipped"] += 1
                continue
            counts["eligible_failures"] += 1
            if not teacher_path.exists():
                reflection = reflect(
                    teacher,
                    task=task,
                    schema=render_schema(task.resolve_db_path(config.spider_root)),
                    initial=initial,
                    max_tokens=teacher_max_tokens,
                )
                write_json(teacher_path, reflection)
            experience = read_json(teacher_path)["experience"]
            candidate = "教师提供的候选通用经验：\n" + experience
            # Candidate has priority; old memories remain whole or are omitted.
            if counter.count_text(candidate) > memory_max_tokens:
                raise ValueError("Candidate exceeds memory input budget")
            retry_context = contexts[task.task_id]
            if counter.count_text(retry_context + "\n\n" + candidate) > memory_max_tokens:
                retry_context = ""
            retry_context = (retry_context + "\n\n" + candidate).strip()
            if not retry_path.exists():
                save_student_batch([task], {task.task_id: retry_context}, 2)
            retry = read_json(retry_path)
            if retry.get("task_id") != task.task_id:
                raise ValueError("Retry record belongs to another task")
            if retry.get("correct") is True:
                accepted.append((task, experience, initial_path, retry_path, verifier))
            if error_path.exists():
                error_path.unlink()
        except (ValueError, RuntimeError, OSError) as exc:
            counts["errors"] += 1
            write_json(error_path, {"task_id": task.task_id, "error": str(exc)})

    # Publish only after all attempts; the frozen snapshot was used throughout.
    for task, experience, initial_path, retry_path, verifier in accepted:
        try:
            inserted = store.append_verified(
                task=task,
                experience=experience,
                initial_path=initial_path,
                retry_path=retry_path,
                verifier=verifier,
            )
            counts["retry_correct"] += 1
            counts["appended" if inserted else "already_present"] += 1
        except (ValueError, RuntimeError, OSError) as exc:
            counts["errors"] += 1
            write_json(paths(task)[3], {"task_id": task.task_id, "error": str(exc)})
    write_json(output_dir / "summary.json", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-weight, append-only SQL memory evolution")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="SFT dataset directory containing manifest.json with prompt",
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--memory-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-base-url", default="https://api.deepseek.com")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--teacher-max-tokens", type=int, default=1024)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--memory-max-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    args = parser.parse_args()
    if (
        min(
            args.rounds,
            args.batch_size,
            args.max_tokens,
            args.teacher_max_tokens,
            args.memory_max_tokens,
            args.embedding_batch_size,
            args.embedding_max_length,
        )
        < 1
        or min(args.limit, args.memory_top_k) < 0
    ):
        parser.error("Invalid numeric limits")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY in the environment")
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks or any(task.split != "train" for task in tasks):
        parser.error("Use a nonempty train task file")
    config = replace(EnvConfig.from_yaml(args.env_config), spider_root=args.spider_root.resolve())
    prompt = read_json(args.dataset / "manifest.json")["prompt"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Separate SQLite lock releases automatically on process death; no stale lock files.
    with sqlite3.connect(args.output_dir / ".run_lock.sqlite", timeout=0) as lock:
        lock.execute("CREATE TABLE IF NOT EXISTS run_lock (id INTEGER)")
        lock.execute("BEGIN IMMEDIATE")
        metadata = {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "rounds"
        }
        metadata.update(
            {
                "version": 1,
                "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "env_sha256": hashlib.sha256(args.env_config.read_bytes()).hexdigest(),
            }
        )
        ensure_manifest(args.output_dir / "run_manifest.json", metadata)
        from transformers import AutoTokenizer
        from vllm import LLM

        from evaluation.run_reasoning_sft import evaluate_batch

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        llm = LLM(
            model=str(args.model),
            dtype="bfloat16",
            max_model_len=config.max_context_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.batch_size,
        )
        try:
            teacher = DeepSeekClient(api_key, model=args.teacher_model, base_url=args.teacher_base_url)
            store = MemoryStore(args.memory_db)
            embedding_encoder = EmbeddingEncoder(
                args.embedding_model, device=args.embedding_device,
                batch_size=args.embedding_batch_size, max_length=args.embedding_max_length,
            )

            def student(batch, contexts):
                return evaluate_batch(
                    batch, llm, tokenizer, prompt, config, args.max_tokens, memory_contexts=contexts
                )

            for index in range(1, args.rounds + 1):
                result = run_round(
                    tasks=tasks,
                    student=student,
                    teacher=teacher,
                    config=config,
                    store=store,
                    output_dir=args.output_dir / f"round_{index:03d}",
                    top_k=args.memory_top_k,
                    memory_max_tokens=args.memory_max_tokens,
                    teacher_max_tokens=args.teacher_max_tokens,
                    batch_size=args.batch_size,
                    embedding_encoder=embedding_encoder,
                )
                print(json.dumps({"round": index, **result}), flush=True)
                if result["errors"]:
                    raise RuntimeError("Round has errors; resolve them and resume before later rounds")
        finally:
            # Explicitly close model workers before multiprocessing's exit handler,
            # including when a failed teacher request keeps a traceback alive.
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
