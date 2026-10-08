"""Explicitly enrich the live memory bank from its immutable source trajectories."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from experience_memory.store import MemoryStore, first_executed_sql


def backfill_first_sql(database: Path, jsonl_path: Path | None = None) -> dict:
    if not database.is_file():
        raise FileNotFoundError(database)
    records = MemoryStore(database).all()
    changes = []
    enriched = []
    for record in records:
        initial = json.loads(Path(record["initial_record_path"]).read_text())
        if (initial.get("task_id") != record["source_task_id"]
                or initial.get("final_sql") != record["sql_before"]):
            raise ValueError("Initial trajectory does not match memory evidence")
        first_sql = first_executed_sql(initial)
        if "sql_first_execute" in record and record["sql_first_execute"] != first_sql:
            raise ValueError("Existing first-execution SQL disagrees with evidence")
        updated = {**record, "sql_first_execute": first_sql}
        enriched.append(updated)
        if "sql_first_execute" not in record:
            changes.append((record, updated))
    # Validate all evidence before writing. Check records again under the write lock.
    with sqlite3.connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for original, updated in changes:
            current = connection.execute("SELECT record FROM memories WHERE memory_id = ?",
                                         (original["memory_id"],)).fetchone()
            if current is None or json.loads(current[0]) != original:
                raise ValueError("Memory changed during backfill")
            connection.execute("UPDATE memories SET record = ? WHERE memory_id = ?",
                               (json.dumps(updated, ensure_ascii=False, sort_keys=True),
                                original["memory_id"]))
    if jsonl_path is not None:
        temporary = jsonl_path.with_name(jsonl_path.name + ".tmp")
        temporary.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                                     for record in enriched))
        temporary.replace(jsonl_path)
    return {"total": len(enriched), "updated": len(changes),
            "with_first_execute_sql": sum(r["sql_first_execute"] is not None for r in enriched),
            "without_execute_sql": sum(r["sql_first_execute"] is None for r in enriched)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-db", required=True, type=Path)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(backfill_first_sql(args.memory_db, args.jsonl), ensure_ascii=False))
