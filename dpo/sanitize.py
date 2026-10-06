"""Remove teacher-hint language from generated DPO reasoning, preserving tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpo.pairs import make_pair
from sql_planner.collect import _atomic_json
from sql_planner.collect_tagged import parse_response
from sql_planner.repair_tagged import GOLD_MENTION, sanitize_reasoning_response


def sanitize_attempt(attempt: dict) -> list[int]:
    """Edit only suffix reasoning; keep SQL, tool action, and observations intact."""
    changed: list[int] = []
    for turn in attempt["turns"]:
        if turn["origin"] != "gold_guided_suffix" or not GOLD_MENTION.search(turn["reasoning"]):
            continue
        old_response = turn["response"]
        _, old_action = parse_response(old_response)
        new_response, replacement, _ = sanitize_reasoning_response(
            old_response, turn["reasoning"], old_action
        )
        message_index = 2 * turn["turn"]
        message = attempt["student_messages"][message_index]
        if message != {"role": "assistant", "content": old_response}:
            raise ValueError("Attempt messages and turns disagree")
        turn["reasoning"] = replacement
        turn["response"] = new_response
        attempt["student_messages"][message_index] = {"role": "assistant", "content": new_response}
        changed.append(turn["turn"])
    attempt["sanitized_reasoning_turns"] = sorted(set(
        attempt.get("sanitized_reasoning_turns", []) + changed
    ))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=Path("artifacts/sql_planner/reasoning"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/sql_planner/reasoning_dpo_all_errors_v1"))
    args = parser.parse_args()
    selection = json.loads((args.output_dir / "selection.json").read_text(encoding="utf-8"))
    entries = {entry["file"]: entry for entry in selection["entries"] if entry["eligible"]}
    pairs_dir = args.output_dir / "pairs"
    files = {path.name: path for path in pairs_dir.glob("*.json")}
    if files.keys() != entries.keys():
        raise ValueError("All eligible pairs must exist before sanitization")
    edited_pairs = edited_turns = 0
    for name, path in sorted(files.items()):
        entry = entries[name]
        attempt_path = args.output_dir / "attempts" / name
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        source = json.loads((args.source_dir / "trajectories" / name).read_text(encoding="utf-8"))
        changed = sanitize_attempt(attempt)
        pair = make_pair(source, attempt, entry["fork_turn"], entry["fork_reason"])
        if any(GOLD_MENTION.search(turn["reasoning"]) for turn in pair["chosen_turns"]):
            raise ValueError(f"Teacher hint remains in {name}")
        old_pair = json.loads(path.read_text(encoding="utf-8"))
        if pair["chosen_verification"] != old_pair["chosen_verification"]:
            raise ValueError(f"Verification changed in {name}")
        if changed:
            _atomic_json(attempt_path, attempt)
            edited_pairs += 1
            edited_turns += len(changed)
        if pair != old_pair:
            _atomic_json(path, pair)
    print(json.dumps({"pairs": len(files), "sanitized_pairs": edited_pairs,
                      "sanitized_turns": edited_turns}))


if __name__ == "__main__":
    main()
