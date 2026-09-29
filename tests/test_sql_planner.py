from __future__ import annotations

import json
from collections import deque
from dataclasses import replace
from threading import Lock

import pytest

from sql_planner.collect import (
    api_tools,
    build_prompt,
    collect,
    collect_trajectory,
    ensure_run_manifest,
    read_spider_train_tasks,
)
from sql_planner.deepseek import Completion, DeepSeekClient


class FakeClient:
    model = "fake-deepseek"

    def __init__(self, *completions: Completion):
        self.completions = deque(completions)
        self.requests = []

    def complete(self, messages, tools, *, temperature, max_tokens):
        self.requests.append((messages.copy(), tools, temperature, max_tokens))
        return self.completions.popleft()


def _tool_completion(name: str, arguments: dict, call_id: str) -> Completion:
    return Completion(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        },
        finish_reason="tool_calls",
        request_id=f"request-{call_id}",
        model="fake-deepseek",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def test_free_order_prompt_and_tool_schemas() -> None:
    prompt = build_prompt(10)
    assert "no required tool order" in prompt
    assert "Start with list_tables" not in prompt
    assert "Decide independently" in prompt
    assert "output exactly one native tool call" in prompt
    assert "Wait for that tool's observation" in prompt
    assert "Never return multiple tool calls" in prompt
    assert "valid JSON arguments object" in prompt
    assert "no prose, explanation, reasoning, chain of thought" in prompt
    assert "final argument" not in prompt
    assert "10 exploratory tool calls, excluding submit_sql" in prompt
    assert "After 10 exploratory calls, your next response must call submit_sql" in prompt
    assert "immediately preceding tool call was a successful execute_sql" in prompt
    assert "preferably use the same SQL" in prompt
    assert {tool["function"]["name"] for tool in api_tools()} == {
        "list_tables", "inspect_tables", "inspect_values", "execute_sql", "submit_sql"
    }
    submit_schema = next(
        tool["function"]["parameters"] for tool in api_tools()
        if tool["function"]["name"] == "submit_sql"
    )
    assert submit_schema["required"] == ["sql"]
    assert set(submit_schema["properties"]) == {"sql"}
    assert "final" not in submit_schema["properties"]


def test_read_spider_train_tasks_uses_only_requested_raw_file(tmp_path) -> None:
    sql = {
        "except": None,
        "from": {"conds": [], "table_units": [["table_unit", 0]]},
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [False, [[0, [0, [0, 1, False], None]]]],
        "union": None,
        "where": [],
    }
    rows = [
        {"db_id": "db_a", "question": "q0", "query": "SELECT c FROM t", "sql": sql},
        {"db_id": "db_b", "question": "q1", "query": "SELECT d FROM u", "sql": sql},
    ]
    (tmp_path / "train_spider.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "train_others.json").write_text(
        json.dumps([{**rows[0], "question": "must not load"}]), encoding="utf-8"
    )
    tasks = read_spider_train_tasks(tmp_path, limit=0, seed=42)
    assert {task.question for task in tasks} == {"q0", "q1"}
    assert {task.task_id for task in tasks} == {"spider_train_00000", "spider_train_00001"}
    assert all(task.split == "train" and task.difficulty == "easy" for task in tasks)


def test_collect_trajectory_preserves_chosen_order_and_verifies(sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": "SELECT 1"}, "call-1"),
        _tool_completion("inspect_tables", {"table_names": ["employees"]}, "call-2"),
        _tool_completion("submit_sql", {"sql": "SELECT count(*) FROM employees"}, "call-3"),
    )
    record = collect_trajectory(task, replace(config, max_turns=3), client)
    assert record["tool_sequence"] == ["execute_sql", "inspect_tables", "submit_sql"]
    assert record["correct"] and record["status"] == "submitted_sql"
    assert record["final_sql"] == "SELECT count(*) FROM employees"
    assert record["usage"]["total_tokens"] == 45
    assert not record["has_batched_calls"]
    assert client.requests[1][0][-1]["role"] == "tool"
    assert client.requests[1][0][-1]["tool_call_id"] == "call-1"
    assert {tool["function"]["name"] for tool in client.requests[2][1]} == {
        "list_tables", "inspect_tables", "inspect_values", "execute_sql", "submit_sql"
    }
    assert task.reference_sql not in client.requests[0][0][1]["content"]


def test_execute_without_submit_does_not_create_answer(sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": "SELECT count(*) FROM employees"}, "call-1"),
        _tool_completion("list_tables", {}, "call-2"),
        _tool_completion("list_tables", {}, "call-3"),
        Completion(message={"role": "assistant", "content": "no submission"},
                   finish_reason="stop", request_id="text", model="fake-deepseek", usage={}),
    )
    record = collect_trajectory(task, replace(config, max_turns=3), client)
    assert record["tool_sequence"] == ["execute_sql", "list_tables", "list_tables"]
    assert record["status"] == "max_turns_without_submit"
    assert not record["correct"]
    assert record["final_sql"] is None


def test_early_submit_repeats_previous_successful_sql(sample_db, config, task) -> None:
    del sample_db
    sql = "SELECT count(*) FROM employees"
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": sql}, "call-1"),
        _tool_completion("submit_sql", {"sql": sql}, "call-2"),
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "submitted_sql"
    assert record["correct"]
    assert record["turns"][0]["observation"]["rows"] == [[25]]
    assert record["turns"][1]["arguments"]["sql"] == sql
    assert len(client.requests) == 2
    assert record["turns"][1]["observation"]["status"] == "success"


def test_early_submit_does_not_require_identical_sql(sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": "SELECT count(*) FROM employees"}, "call-1"),
        _tool_completion("submit_sql", {"sql": "SELECT 1"}, "call-2"),
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "submitted_sql"
    assert record["tool_sequence"] == ["execute_sql", "submit_sql"]
    assert not record["correct"]
    assert "sql_matches_previous" not in record


def test_early_submit_after_failed_execute_is_not_hard_rejected(sample_db, config, task) -> None:
    del sample_db
    sql = "SELECT missing FROM employees"
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": sql}, "call-1"),
        _tool_completion("submit_sql", {"sql": "SELECT count(*) FROM employees"}, "call-2"),
    )
    record = collect_trajectory(task, config, client)
    assert record["turns"][0]["observation"]["status"] == "error"
    assert record["status"] == "submitted_sql"
    assert record["correct"]
    assert len(record["turns"]) == 2


def test_first_turn_submit_is_not_hard_rejected(sample_db, config, task) -> None:
    del sample_db
    record = collect_trajectory(
        task,
        config,
        FakeClient(
            _tool_completion(
                "submit_sql", {"sql": "SELECT count(*) FROM employees"}, "call-1"
            )
        ),
    )
    assert record["status"] == "submitted_sql"
    assert record["correct"]
    assert record["tool_sequence"] == ["submit_sql"]


def test_intervening_tool_does_not_hard_reject_submit(sample_db, config, task) -> None:
    del sample_db
    sql = "SELECT count(*) FROM employees"
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": sql}, "call-1"),
        _tool_completion("list_tables", {}, "call-2"),
        _tool_completion("submit_sql", {"sql": sql}, "call-3"),
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "submitted_sql"
    assert record["correct"]
    assert record["tool_sequence"] == ["execute_sql", "list_tables", "submit_sql"]


def test_last_turn_submit_sql_needs_only_sql(sample_db, config, task) -> None:
    del sample_db
    two_turn_config = replace(config, max_turns=2)
    client = FakeClient(
        _tool_completion("list_tables", {}, "call-1"),
        _tool_completion("submit_sql", {"sql": "SELECT count(*) FROM employees"}, "call-2"),
    )
    record = collect_trajectory(task, two_turn_config, client)
    assert record["status"] == "submitted_sql"
    assert record["correct"]
    assert record["tool_sequence"] == ["list_tables", "submit_sql"]
    assert len(client.requests[1][1]) == 5


def test_last_turn_other_tool_is_not_hard_rejected(sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        _tool_completion("list_tables", {}, "call-1"),
        _tool_completion("inspect_tables", {"table_names": ["employees"]}, "call-2"),
        Completion(message={"role": "assistant", "content": "no submission"},
                   finish_reason="stop", request_id="text", model="fake-deepseek", usage={}),
    )
    record = collect_trajectory(task, replace(config, max_turns=2), client)
    assert record["status"] == "max_turns_without_submit"
    assert record["tool_sequence"] == ["list_tables", "inspect_tables"]


def test_tenth_turn_submit_sql_is_accepted(sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        *(_tool_completion("list_tables", {}, f"call-{index}") for index in range(1, 10)),
        _tool_completion(
            "submit_sql",
            {"sql": "SELECT count(*) FROM employees"},
            "call-10",
        ),
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "submitted_sql"
    assert record["correct"]
    assert len(record["turns"]) == 10
    assert record["turns"][-1]["arguments"] == {"sql": "SELECT count(*) FROM employees"}


def test_ten_exploration_calls_leave_one_submit_call(sample_db, config, task) -> None:
    del sample_db
    sql = "SELECT count(*) FROM employees"
    client = FakeClient(
        *(_tool_completion("list_tables", {}, f"call-{index}") for index in range(1, 10)),
        _tool_completion("execute_sql", {"sql": sql}, "call-10"),
        _tool_completion("submit_sql", {"sql": sql}, "call-11"),
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "submitted_sql" and record["correct"]
    assert record["exploration_turns"] == 10
    assert record["submission_budget"] == 1
    assert len(record["turns"]) == 11
    assert record["turns"][9]["observation"]["turns_remaining"] == 0
    assert [tool["function"]["name"] for tool in client.requests[-1][1]] == ["submit_sql"]


def test_multi_table_inspection_is_one_exploration_call(sample_db, config, task) -> None:
    del sample_db
    sql = "SELECT count(*) FROM employees"
    client = FakeClient(
        _tool_completion(
            "inspect_tables",
            {"table_names": ["employees", "departments", "missing"]},
            "call-1",
        ),
        _tool_completion("execute_sql", {"sql": sql}, "call-2"),
        _tool_completion("submit_sql", {"sql": sql}, "call-3"),
    )
    record = collect_trajectory(task, config, client)
    observed = record["turns"][0]["observation"]
    assert record["status"] == "submitted_sql" and record["correct"]
    assert record["exploration_turns"] == 2
    assert observed["requested_table_count"] == 3
    assert observed["returned_table_count"] == 3
    assert [table["status"] for table in observed["tables"]] == [
        "success", "success", "error"
    ]
    assert observed["tables"][2]["error_type"] == "unknown_table"


def test_submit_in_same_batched_response_is_rejected_without_execution(sample_db, config, task) -> None:
    del sample_db
    sql = "SELECT count(*) FROM employees"
    completion = _tool_completion("execute_sql", {"sql": sql}, "call-1")
    completion.message["tool_calls"].append(
        _tool_completion("submit_sql", {"sql": sql}, "call-2")
        .message["tool_calls"][0]
    )
    record = collect_trajectory(task, config, FakeClient(completion))
    assert record["status"] == "multiple_tool_calls"
    assert not record["correct"]
    assert record["tool_sequence"] == []
    assert record["final_sql"] is None
    assert record["exploration_turns"] == 0
    assert record["has_batched_calls"]
    assert len(record["messages"][-1]["tool_calls"]) == 2


def test_submit_sql_rejects_legacy_final_argument(sample_db, config, task) -> None:
    del sample_db
    record = collect_trajectory(
        task,
        replace(config, max_turns=1),
        FakeClient(_tool_completion("submit_sql", {"sql": "SELECT 1", "final": True}, "call-1")),
    )
    assert record["turns"][0]["observation"]["error_type"] == "invalid_arguments"
    assert record["status"] == "invalid_submission"
    assert record["final_sql"] is None
    assert record["tool_sequence"] == ["submit_sql"]


def test_collect_resumes_without_duplicate_api_calls(tmp_path, sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": "SELECT count(*) FROM employees"}, "call-1"),
        _tool_completion("submit_sql", {"sql": "SELECT count(*) FROM employees"}, "call-2"),
    )
    arguments = dict(
        output_dir=tmp_path / "out", samples_per_task=1,
        temperature=0.7, max_tokens=512,
    )
    first = collect([task], config, client, **arguments)
    second = collect([task], config, client, **arguments)
    assert first == {"created": 1, "skipped": 0, "correct": 1, "errors": 0}
    assert second == {"created": 0, "skipped": 1, "correct": 0, "errors": 0}
    saved = json.loads(next((tmp_path / "out" / "trajectories").glob("*.json")).read_text())
    assert saved["tool_sequence"] == ["execute_sql", "submit_sql"]
    assert len(client.requests) == 2


def test_collect_supports_parallel_workers(tmp_path, sample_db, config, task) -> None:
    del sample_db

    class ConcurrentFakeClient:
        model = "fake-deepseek"

        def __init__(self):
            self.counts = {}
            self.lock = Lock()

        def complete(self, messages, tools, *, temperature, max_tokens):
            del tools, temperature, max_tokens
            question = messages[1]["content"]
            with self.lock:
                call_index = self.counts.get(question, 0)
                self.counts[question] = call_index + 1
            if call_index == 0:
                return _tool_completion(
                    "execute_sql", {"sql": "SELECT count(*) FROM employees"}, question
                )
            return _tool_completion(
                "submit_sql", {"sql": "SELECT count(*) FROM employees"}, f"done-{question}"
            )

    tasks = [task, replace(task, task_id="sample_00001", question="Count employees again")]
    counts = collect(
        tasks,
        config,
        ConcurrentFakeClient(),
        output_dir=tmp_path / "parallel",
        samples_per_task=1,
        temperature=0.7,
        max_tokens=512,
        workers=2,
    )
    assert counts == {"created": 2, "skipped": 0, "correct": 2, "errors": 0}
    assert len(list((tmp_path / "parallel" / "trajectories").glob("*.json"))) == 2


def test_nonfinal_text_is_not_counted_as_a_verified_answer(sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": "SELECT count(*) FROM employees"}, "call-1"),
        Completion(message={"role": "assistant", "content": "I need to think"},
                   finish_reason="stop", request_id="text", model="fake-deepseek", usage={}),
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "unexpected_text"
    assert record["verification"] is None
    assert not record["correct"]


def test_explanation_ending_in_done_is_rejected(sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": "SELECT count(*) FROM employees"}, "call-1"),
        Completion(
            message={"role": "assistant", "content": "The count is 25.\nDONE"},
            finish_reason="stop",
            request_id="done",
            model="deepseek-v4.1-flash",
            usage={},
        ),
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "unexpected_text"
    assert not record["correct"]
    assert record["response_models"] == ["fake-deepseek", "deepseek-v4.1-flash"]


def test_done_prefix_followed_by_explanation_is_rejected(sample_db, config, task) -> None:
    del sample_db
    client = FakeClient(
        _tool_completion("execute_sql", {"sql": "SELECT count(*) FROM employees"}, "call-1"),
        Completion(
            message={"role": "assistant", "content": "DONE The count is 25."},
            finish_reason="stop",
            request_id="done",
            model="deepseek-flash",
            usage={},
        ),
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "unexpected_text"
    assert not record["correct"]


def test_tool_call_with_explanation_is_rejected(sample_db, config, task) -> None:
    del sample_db
    completion = _tool_completion("execute_sql", {"sql": "SELECT 1"}, "call-1")
    completion.message["content"] = "I will run this query."
    record = collect_trajectory(task, config, FakeClient(completion))
    assert record["status"] == "unexpected_text"
    assert record["tool_sequence"] == []


def test_multiple_tool_calls_in_one_response_stop_trajectory(sample_db, config, task) -> None:
    del sample_db
    completion = _tool_completion("list_tables", {}, "call-1")
    completion.message["tool_calls"].append(
        _tool_completion("inspect_tables", {"table_names": ["employees"]}, "call-2")
        .message["tool_calls"][0]
    )
    client = FakeClient(completion)
    record = collect_trajectory(task, config, client)
    assert record["status"] == "multiple_tool_calls"
    assert record["has_batched_calls"]
    assert record["tool_sequence"] == []
    assert record["turns"] == []
    assert record["exploration_turns"] == 0
    assert "received 2" in record["error"]
    assert len(client.requests) == 1


def test_multiple_calls_after_valid_step_do_not_execute_batch(sample_db, config, task) -> None:
    del sample_db
    completion = _tool_completion("inspect_tables", {"table_names": ["employees"]}, "call-2")
    completion.message["tool_calls"].append(
        _tool_completion("execute_sql", {"sql": "SELECT count(*) FROM employees"}, "call-3")
        .message["tool_calls"][0]
    )
    client = FakeClient(_tool_completion("list_tables", {}, "call-1"), completion)
    record = collect_trajectory(task, config, client)
    assert record["status"] == "multiple_tool_calls"
    assert record["tool_sequence"] == ["list_tables"]
    assert record["exploration_turns"] == 1
    assert len(record["messages"][-1]["tool_calls"]) == 2
    assert record["messages"][-2]["role"] == "tool"


def test_manifest_rejects_mixed_experiment_settings(tmp_path) -> None:
    output = tmp_path / "run"
    ensure_run_manifest(output, {"model": "deepseek-flash"})
    ensure_run_manifest(output, {"model": "deepseek-flash"})
    with pytest.raises(ValueError, match="different settings"):
        ensure_run_manifest(output, {"model": "deepseek-v4-pro"})


def test_deepseek_client_uses_native_tools_without_sequence_constraints(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return json.dumps({
                "id": "req-1", "model": "deepseek-flash",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": []}}],
                "usage": {"prompt_tokens": 2},
            }).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = DeepSeekClient("secret-for-test")
    completion = client.complete(
        [{"role": "user", "content": "question"}], api_tools(),
        temperature=0.7, max_tokens=512,
    )
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["payload"]["tool_choice"] == "required"
    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert completion.request_id == "req-1"
