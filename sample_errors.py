"""Sample representative incorrect trajectories per category, with gold SQL."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TRAJ_DIR = Path(r"C:\Users\JSWang\Documents\DLnotes\sqlagent\sql-agent-rl\artifacts\sql_planner\spider_train_7000_submit_sql_v9\trajectories")
SPIDER = Path(r"C:\Users\JSWang\Documents\DLnotes\sqlagent\datasets\spider\spider_data\train_spider.json")

rows = json.loads(SPIDER.read_text(encoding="utf-8"))
gold = {f"spider_train_{i:05d}": r["query"] for i, r in enumerate(rows)}


def bucket(r: dict) -> str:
    v = r.get("verification") or {}
    if r.get("status") != "submitted_sql":
        return "no_submit"
    if not v.get("agent_sql_executable", True):
        return "not_executable"
    col, rows, val = v.get("column_match"), v.get("row_count_match"), v.get("value_match")
    if not col and not rows:
        return "col_and_row_mismatch"
    if not col:
        return "column_count_mismatch"
    if not rows:
        return "row_count_mismatch"
    return "value_mismatch"


records = []
for p in sorted(TRAJ_DIR.glob("*.json")):
    with open(p, encoding="utf-8") as f:
        records.append(json.load(f))

incorrect = [r for r in records if not r.get("correct")]
buckets = defaultdict(list)
for r in incorrect:
    buckets[bucket(r)].append(r)

print("bucket sizes:")
for b, lst in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    print(f"  {b:24s} {len(lst)}")

import sys
only = sys.argv[1] if len(sys.argv) > 1 else None
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8

for b, lst in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    if only and b != only:
        continue
    print(f"\n{'='*100}\n## {b}  (n={len(lst)})\n{'='*100}")
    for r in lst[:n]:
        q = r["question"].replace("\n", " ")
        print(f"\n--- [{r['difficulty']}] {r['task_id']}")
        print(f"Q: {q}")
        print(f"AGENT: {r.get('final_sql')}")
        print(f"GOLD : {gold.get(r['task_id'], '?')}")
