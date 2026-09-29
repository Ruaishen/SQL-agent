"""Analyze error causes in collected SQL-Planner trajectories."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

TRAJ_DIR = Path(r"C:\Users\JSWang\Documents\DLnotes\sqlagent\sql-agent-rl\artifacts\sql_planner\spider_train_7000_submit_sql_v9\trajectories")


def load_all() -> list[dict]:
    records = []
    for p in sorted(TRAJ_DIR.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def main() -> None:
    records = load_all()
    n = len(records)
    correct = [r for r in records if r.get("correct")]
    incorrect = [r for r in records if not r.get("correct")]
    print(f"total={n} correct={len(correct)} incorrect={len(incorrect)} "
          f"acc={len(correct)/n:.4f}")

    # status distribution (all)
    print("\n== trajectory status (all) ==")
    for status, cnt in Counter(r["status"] for r in records).most_common():
        print(f"  {status:30s} {cnt}")

    # status distribution (incorrect only)
    print("\n== trajectory status (incorrect only) ==")
    for status, cnt in Counter(r["status"] for r in incorrect).most_common():
        print(f"  {status:30s} {cnt}")

    # verification.error distribution (incorrect only)
    print("\n== verification.error (incorrect only) ==")
    verr = Counter()
    for r in incorrect:
        v = r.get("verification") or {}
        verr[v.get("error", "(no_verification)")] += 1
    for err, cnt in verr.most_common():
        print(f"  {err:35s} {cnt}")

    # For submitted-but-wrong trajectories, deeper breakdown via verification flags
    print("\n== submitted-but-wrong: verification flag combos ==")
    submitted_wrong = [r for r in incorrect if r.get("status") == "submitted_sql"]
    flags = Counter()
    for r in submitted_wrong:
        v = r.get("verification") or {}
        key = (v.get("agent_sql_valid"), v.get("agent_sql_executable"),
               v.get("column_match"), v.get("row_count_match"), v.get("value_match"))
        flags[key] += 1
    for key, cnt in flags.most_common():
        print(f"  valid={key[0]} exec={key[1]} col={key[2]} rows={key[3]} val={key[4]}  -> {cnt}")

    # difficulty x correctness
    print("\n== difficulty x correct ==")
    by_diff = defaultdict(lambda: [0, 0])
    for r in records:
        d = r.get("difficulty", "?")
        by_diff[d][1] += 1
        if r.get("correct"):
            by_diff[d][0] += 1
    for d in ["easy", "medium", "hard", "extra"]:
        if d in by_diff:
            c, t = by_diff[d]
            print(f"  {d:8s} {c}/{t}  acc={c/t:.4f}")

    # error by difficulty (incorrect only)
    print("\n== verification.error x difficulty (incorrect only) ==")
    diff_err = defaultdict(Counter)
    for r in incorrect:
        d = r.get("difficulty", "?")
        v = r.get("verification") or {}
        diff_err[d][v.get("error", "(no_verification)")] += 1
    for d in ["easy", "medium", "hard", "extra"]:
        if d not in diff_err:
            continue
        top = ", ".join(f"{e}={c}" for e, c in diff_err[d].most_common(5))
        print(f"  {d:8s} {top}")

    # tool_sequence of incorrect trajectories (last tool & whether submit happened)
    print("\n== incorrect: last tool in sequence ==")
    last_tool = Counter()
    for r in incorrect:
        seq = r.get("tool_sequence") or []
        last_tool[seq[-1] if seq else "(none)"] += 1
    for tool, cnt in last_tool.most_common():
        print(f"  {tool:25s} {cnt}")

    # turn count distribution for incorrect
    print("\n== incorrect: turn count distribution ==")
    turncnt = Counter(len(r.get("tool_sequence") or []) for r in incorrect)
    for t in sorted(turncnt):
        print(f"  turns={t:3d} {turncnt[t]}")


if __name__ == "__main__":
    main()
