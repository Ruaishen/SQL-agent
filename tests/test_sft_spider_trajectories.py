from __future__ import annotations

import json

from sft.spider_trajectories import (
    SAMPLE_COUNTS,
    _normalize_messages,
    build_sft_prompt,
    sample_trajectories,
)
from sql_planner.collect import build_prompt


def test_sft_prompt_changes_only_order_and_ending() -> None:
    original = build_prompt(10)
    prompt = build_sft_prompt(10)
    assert "First call list_tables, then call inspect_tables" in prompt
    assert "Every trajectory must end with submit_sql." in prompt
    assert "no required tool order" not in prompt
    assert "You have at most 10 exploratory tool calls" in prompt
    assert "After 10 exploratory calls, your next response must call submit_sql" in prompt
    assert original == build_prompt(10)


def test_normalization_does_not_change_source_record() -> None:
    record = {
        "task_id": "sample",
        "max_turns": 10,
        "messages": [
            {"role": "system", "content": build_prompt(10)},
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "list_tables", "arguments": "{}"}}],
            },
        ],
    }
    before = json.dumps(record)
    messages = _normalize_messages(record)
    assert json.dumps(record) == before
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {}
    assert messages[0]["content"] == build_sft_prompt(10)


def test_sampling_exact_reproducible_and_leaves_source_unchanged(tmp_path) -> None:
    source = tmp_path / "trajectories.jsonl"
    records = [
        {
            "task_id": f"{difficulty}_{index}",
            "difficulty": difficulty,
            "correct": True,
            "split": "train",
        }
        for difficulty, count in SAMPLE_COUNTS.items()
        for index in range(count + 2)
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    original = source.read_bytes()
    selected, available = sample_trajectories(source, 42)
    again, _ = sample_trajectories(source, 42)
    assert len(selected) == 2500
    assert {
        difficulty: sum(record["difficulty"] == difficulty for record in selected)
        for difficulty in SAMPLE_COUNTS
    } == SAMPLE_COUNTS
    assert available == {difficulty: count + 2 for difficulty, count in SAMPLE_COUNTS.items()}
    assert [record["task_id"] for record in selected] == [record["task_id"] for record in again]
    assert source.read_bytes() == original
