from __future__ import annotations

import json
from dataclasses import replace

from dpo.pairs import make_pair
from sql_planner.collect_tagged import build_prompt, collect_trajectory
from sql_planner.deepseek import Completion
from sql_planner.repair_tagged import GOLD_MENTION, repair_trajectory, teacher_prompt


def response(name, sql=None):
    arguments = {} if sql is None else {"sql": sql}
    return ("<reasoning>Use the available database evidence.</reasoning><tool>"
            + json.dumps({"name": name, "arguments": arguments}) + "</tool>")


class FakeTeacher:
    model = "offline-fake"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete_text(self, messages, *, temperature, max_tokens):
        self.requests.append([dict(message) for message in messages])
        return Completion(message={"content": self.responses.pop(0)},
                          finish_reason="stop", request_id="offline",
                          model=self.model, usage={})


def test_teacher_system_matches_ordinary_system_and_gold_only_in_fork_hint(sample_db, config, task):
    del sample_db
    for can_test in (True, False):
        assert teacher_prompt(config.max_turns, can_test=can_test) == build_prompt(config.max_turns)
    wrong = collect_trajectory(task, config, FakeTeacher(
        response("list_tables"), response("submit_sql", "SELECT 1")))
    teacher = FakeTeacher(response("submit_sql", "SELECT 1"),
                          response("submit_sql", task.reference_sql))
    repaired = repair_trajectory(wrong, task, config, teacher, cutoff_turn=2,
                                 sanitize_hint_reasoning=False)
    assert repaired["correct"]
    first, retry = teacher.requests
    assert first[0]["content"] == wrong["messages"][0]["content"]
    assert first[:-1] == wrong["messages"][:4]
    assert task.reference_sql in first[-1]["content"]
    assert "Do not quote or mention this hint in your reasoning." in first[-1]["content"]
    assert "Every claimed observation must come from an actual tool result." in first[-1]["content"]
    assert retry[:-1] == first
    assert task.reference_sql not in retry[-1]["content"]
    assert "gold" not in retry[-1]["content"].lower()
    assert repaired["student_messages"][:4] == first[:-1]


def test_untested_submission_forks_before_submit(sample_db, config, task):
    del sample_db
    wrong = collect_trajectory(task, config, FakeTeacher(
        response("list_tables"), response("submit_sql", "SELECT 1")))
    teacher = FakeTeacher(response("execute_sql", task.reference_sql),
                          response("submit_sql", task.reference_sql))
    repaired = repair_trajectory(wrong, task, config, teacher, cutoff_turn=2)
    assert repaired["status"] == "repaired"
    pair = make_pair(wrong, repaired, 2, "before_untested_submission")
    assert pair["prompt_messages"][-1]["role"] == "user"
    assert pair["rejected_messages"][4]["role"] == "assistant"


def test_invalid_format_forks_before_bad_response(sample_db, config, task):
    del sample_db
    wrong = collect_trajectory(task, config, FakeTeacher(
        response("list_tables"), "<reasoning>Broken format.</reasoning>"),
        response_retries=0)
    assert wrong["status"] == "invalid_format"
    repaired = repair_trajectory(wrong, task, config, FakeTeacher(
        response("execute_sql", task.reference_sql),
        response("submit_sql", task.reference_sql)), cutoff_turn=2)
    assert repaired["status"] == "repaired"
    pair = make_pair(wrong, repaired, 2, "before_invalid_response")
    assert pair["rejected_messages"][4]["content"] == "<reasoning>Broken format.</reasoning>"


def test_exhausted_budget_accepts_verified_direct_submit(sample_db, config, task):
    del sample_db
    tiny_budget = replace(config, max_turns=1)
    wrong = collect_trajectory(task, tiny_budget, FakeTeacher(
        response("list_tables"), response("submit_sql", "SELECT 1")))
    repaired = repair_trajectory(wrong, task, tiny_budget,
                                 FakeTeacher(response("submit_sql", task.reference_sql)),
                                 cutoff_turn=2)
    assert repaired["status"] == "repaired_no_test_budget"
    pair = make_pair(wrong, repaired, 2, "before_untested_submission")
    assert pair["chosen_tested_before_submit"] is False
    assert pair["chosen_verification"]["correct"] is True


def test_wrong_submit_is_retried_and_exact_gold_can_be_submitted(sample_db, config, task):
    del sample_db
    wrong = collect_trajectory(task, config, FakeTeacher(
        response("list_tables"), response("submit_sql", "SELECT 1")))
    repaired = repair_trajectory(wrong, task, config, FakeTeacher(
        response("submit_sql", "SELECT 1"),
        response("submit_sql", task.reference_sql)), cutoff_turn=2)
    assert repaired["status"] == "repaired_direct_submit"
    assert len(repaired["rejected_responses"]) == 1
    pair = make_pair(wrong, repaired, 2, "before_untested_submission")
    assert pair["chosen_tested_before_submit"] is False
    assert pair["chosen_verification"]["correct"] is True


def test_gold_column_name_does_not_count_as_hint_leak():
    assert GOLD_MENTION.search("Return the gold value for the top club") is None
    assert GOLD_MENTION.search("The gold SQL says to join coach") is not None
    assert GOLD_MENTION.search("The hint suggests joining coach") is not None


def test_dpo_suffix_neutralizes_teacher_hint_without_changing_sql(sample_db, config, task):
    del sample_db
    wrong = collect_trajectory(task, config, FakeTeacher(
        response("list_tables"), response("submit_sql", "SELECT 1")))
    teacher_response = (
        "<reasoning>The hint suggests using the target SQL.</reasoning><tool>"
        + json.dumps({"name": "submit_sql", "arguments": {"sql": task.reference_sql}})
        + "</tool>"
    )
    repaired = repair_trajectory(wrong, task, config, FakeTeacher(teacher_response),
                                 cutoff_turn=2)
    assert repaired["status"] == "repaired_direct_submit"
    assert repaired["sanitized_reasoning_turns"] == [2]
    pair = make_pair(wrong, repaired, 2, "before_untested_submission")
    assert pair["chosen_final_sql"] == task.reference_sql
    assert GOLD_MENTION.search(pair["chosen_turns"][0]["reasoning"]) is None


def test_comparison_preserves_hint_and_measures_wrong_submission(sample_db, config, task):
    del sample_db
    wrong = collect_trajectory(task, config, FakeTeacher(
        response("list_tables"), response("submit_sql", "SELECT 1")))
    raw = response("submit_sql", "SELECT 1").replace(
        "Use the available database evidence.", "The hint suggests the target SQL.")
    repaired = repair_trajectory(wrong, task, config, FakeTeacher(raw), cutoff_turn=2,
        response_retries=0, sanitize_hint_reasoning=False, require_exact_gold_submission=False)
    assert repaired["status"] == "wrong_final_sql"
    assert repaired["turns"][-1]["response"] == raw
    assert repaired["sanitized_reasoning_turns"] == []
