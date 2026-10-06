from __future__ import annotations

import json
from collections import deque
from dataclasses import replace

import pytest

from sql_planner.collect_tagged import collect_trajectory
from sql_planner.deepseek import Completion
from sql_planner.repair_tagged import choose_cutoff, repair_trajectory, validate_complete_source


class FakeTextClient:
    model = "fake-deepseek"

    def __init__(self, *responses):
        self.responses = deque(responses)
        self.requests = []

    def complete_text(self, messages, *, temperature, max_tokens):
        self.requests.append([message.copy() for message in messages])
        return Completion(
            {"role": "assistant", "content": self.responses.popleft()},
            "stop", "fake-request", self.model,
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def response(reasoning, name, arguments):
    return (
        f"<reasoning>{reasoning}</reasoning>"
        f"<tool>{json.dumps({'name': name, 'arguments': arguments})}</tool>"
    )


def test_cutoff_rewinds_repeated_wrong_query_and_caps_long_prefix():
    turns = [
        {"tool": "list_tables", "arguments": {}},
        {"tool": "inspect_tables", "arguments": {}},
        {"tool": "execute_sql", "arguments": {"sql": "SELECT 1"}},
        {"tool": "submit_sql", "arguments": {"sql": "SELECT 1"}},
    ]
    assert choose_cutoff(turns) == (3, "repeated_final_wrong_sql")
    turns[2]["arguments"]["sql"] = "SELECT 2"
    assert choose_cutoff(turns) == (4, "replace_submission")
    long_turns = turns[:-1] + [turns[0]] * 5 + [turns[-1]]
    assert choose_cutoff(long_turns) == (7, "prefix_turn_cap")


def test_repair_replays_prefix_and_hides_teacher_hint_from_student(sample_db, config, task):
    del sample_db
    wrong = "SELECT 1"
    gold = "SELECT count(*) FROM employees"
    source_client = FakeTextClient(
        response("Discover tables.", "list_tables", {}),
        response("Inspect employees.", "inspect_tables", {"table_names": ["employees"]}),
        response("Try a count.", "execute_sql", {"sql": wrong}),
        response("Submit the result.", "submit_sql", {"sql": wrong}),
    )
    source = collect_trajectory(task, config, source_client)
    assert source["status"] == "submitted_sql" and not source["correct"]
    teacher = FakeTextClient(
        response("The question asks for the employee count; test a count of employee rows.", "execute_sql", {"sql": gold}),
        response("The execution returned 25, so submit this query.", "submit_sql", {"sql": gold}),
    )
    repaired = repair_trajectory(source, task, config, teacher)
    assert repaired["status"] == "repaired" and repaired["sft_eligible"]
    assert repaired["cutoff_turn"] == 3
    assert repaired["trainable_turn_numbers"] == [3, 4]
    assert repaired["tool_sequence"] == ["list_tables", "inspect_tables", "execute_sql", "submit_sql"]
    assert repaired["turns"][2]["observation"]["rows"] == [[25]]
    assert any("The gold SQL for this continuation is:" in m["content"] for m in teacher.requests[0])
    assert not any("The gold SQL for this continuation is:" in m["content"] for m in repaired["student_messages"])
    assert all("gold" not in m["content"].lower() for m in repaired["student_messages"] if m["role"] == "user")


def test_incomplete_source_is_refused(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"prompt_version": "reasoning_tool_observation_v4", "selected_task_count": 7000, "samples_per_task": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete_source(tmp_path)


def test_invalid_gold_is_skipped_without_teacher_call(sample_db, config, task):
    del sample_db
    source_client = FakeTextClient(
        response("Discover.", "list_tables", {}),
        response("Submit.", "submit_sql", {"sql": "SELECT 1"}),
    )
    source = collect_trajectory(task, config, source_client)
    invalid_gold_task = replace(task, reference_sql="SELECT missing FROM employees")
    teacher = FakeTextClient()
    result = repair_trajectory(source, invalid_gold_task, config, teacher)
    assert result["status"] == "gold_not_executable"
    assert not result["sft_eligible"]
    assert teacher.requests == []


def test_correct_but_untested_suffix_is_not_sft_eligible(sample_db, config, task):
    del sample_db
    source = collect_trajectory(
        task, config,
        FakeTextClient(
            response("Discover.", "list_tables", {}),
            response("Submit.", "submit_sql", {"sql": "SELECT 1"}),
        ),
    )
    gold = "SELECT count(*) FROM employees"
    teacher = FakeTextClient(response("Count employee rows.", "submit_sql", {"sql": gold}))
    result = repair_trajectory(source, task, config, teacher)
    assert result["status"] == "untested_submission"
    assert result["verification"]["correct"]
    assert not result["sft_eligible"]
