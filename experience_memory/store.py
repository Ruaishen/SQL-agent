from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from sql_agent.models import TaskRecord
from sql_agent.verifier import ExecutionVerifier

FIELDS = {
    "memory_id",
    "experience",
    "sql_before",
    "sql_after",
    "source_task_id",
    "source_split",
    "initial_record_path",
    "retry_record_path",
}


class MemoryStore:
    """Only successful new records are inserted; existing records are immutable."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memories "
                "(memory_id TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )

    def all(self) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT record FROM memories ORDER BY rowid").fetchall()
        records = [json.loads(row[0]) for row in rows]
        for record in records:
            if set(record) != FIELDS:
                raise ValueError("Memory contains unexpected fields")
        return records

    def append_verified(
        self,
        *,
        task: TaskRecord,
        experience: str,
        initial_path: Path,
        retry_path: Path,
        verifier: ExecutionVerifier,
    ) -> bool:
        """Re-read evidence and re-execute both SQLs before inserting anything."""
        if task.split != "train":
            raise ValueError("Only training tasks can create memories")
        if not isinstance(experience, str) or not experience.strip():
            raise ValueError("Empty experience")
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        retry = json.loads(retry_path.read_text(encoding="utf-8"))
        for record in (initial, retry):
            if record.get("task_id") != task.task_id:
                raise ValueError("Evidence task does not match")
        if initial.get("correct") is not False or retry.get("correct") is not True:
            raise ValueError("Admission requires initial false and retry true")
        if retry.get("status") != "submitted_sql" or retry.get("forced_submission"):
            raise ValueError("Retry must contain a student submission")
        before, after = initial.get("final_sql"), retry.get("final_sql")
        if not all(isinstance(sql, str) and sql.strip() for sql in (before, after)):
            raise ValueError("Both final SQL examples are required")
        if verifier.reference_sql != task.reference_sql:
            raise ValueError("Verifier gold does not match task")
        if not verifier.verify(task.reference_sql).correct:
            raise ValueError("Gold cannot be verified")
        old_result, new_result = verifier.verify(before), verifier.verify(after)
        if old_result.correct or not new_result.correct:
            raise ValueError("Live SQL verification failed admission")
        if old_result.error and old_result.error.startswith("verifier_reference_"):
            raise ValueError("Initial failure was a reference evaluation error")
        if old_result.error in {"timeout", "result_too_large"}:
            raise ValueError("Initial failure was an execution resource error")
        # The evidence identity provides resume idempotence, not semantic merging.
        identity = json.dumps(
            [task.task_id, str(initial_path.resolve()), str(retry_path.resolve())],
            ensure_ascii=False,
        )
        record = {
            "memory_id": "mem_" + hashlib.sha256(identity.encode()).hexdigest(),
            "experience": experience.strip(),
            "sql_before": before,
            "sql_after": after,
            "source_task_id": task.task_id,
            "source_split": task.split,
            "initial_record_path": str(initial_path.resolve()),
            "retry_record_path": str(retry_path.resolve()),
        }
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT record FROM memories WHERE memory_id = ?", (record["memory_id"],)
            ).fetchone()
            if existing:
                if existing[0] != serialized:
                    raise ValueError("Existing evidence identity has different content")
                return False
            connection.execute(
                "INSERT INTO memories VALUES (?, ?)", (record["memory_id"], serialized)
            )
        return True
