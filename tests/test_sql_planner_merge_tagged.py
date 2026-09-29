from __future__ import annotations

from pathlib import Path

from sql_planner.merge_tagged import _unified_record, select_record


def test_merge_priority_and_gold_student_view():
    original = {"correct": False, "turns": [], "messages": [{"role": "user", "content": "q"}]}
    retry = {"correct": True, "turns": [{"tool": "submit_sql"}], "messages": [{"role": "user", "content": "q"}]}
    gold = {
        "correct": True, "sft_eligible": True,
        "turns": [{"tool": "execute_sql"}, {"tool": "submit_sql"}],
        "student_messages": [{"role": "user", "content": "q"}],
        "trainable_turn_numbers": [2],
    }
    assert select_record({**original, "correct": True}, [retry], gold)[0] == "original"
    assert select_record(original, [retry], gold)[0] == "retry"
    origin, selected = select_record(original, [{**retry, "correct": False}], gold)
    assert origin == "gold_repair" and selected is gold
    unified = _unified_record(gold, task_id="task-1", origin=origin, source_path=Path("gold.json"))
    assert unified["messages"] == gold["student_messages"]
    assert unified["trainable_turn_numbers"] == [2]
    assert select_record(original, [], {**gold, "sft_eligible": False}) == (None, None)
