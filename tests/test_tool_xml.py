from __future__ import annotations

import json
from dataclasses import replace

import pytest

from evaluation.run_tool_xml import load_initial_sql, summarize
from evaluation.tool_xml import ToolXmlEvaluator, build_system_prompt, parse_tool_response
from sql_agent.action_parser import ActionParseError
from sql_agent.model_adapter import Generation, ScriptedModelAdapter


def response(name: str, arguments: dict[str, object]) -> str:
    return (
        "<reasoning>I will use the available schema and observation.</reasoning>\n"
        f"<tool>{json.dumps({'name': name, 'arguments': arguments})}</tool>"
    )


def test_tool_response_requires_wrapped_single_action() -> None:
    action = parse_tool_response(response("execute_sql", {"sql": "SELECT 1"}))
    assert action.name == "execute_sql"
    assert action.arguments == {"sql": "SELECT 1"}
    for invalid in (
        "<reasoning>Try SQL</reasoning><sql>SELECT 1</sql>",
        response("execute_sql", {"sql": "SELECT 1"}) + "<observation>fake</observation>",
        response("submit_sql", {"sql": ""}),
        response("list_tables", {}),
        response("inspect_tables", {"table_names": ["employees"]}),
        response("inspect_values", {"table_name": "employees", "column_name": "id"}),
    ):
        with pytest.raises(ActionParseError):
            parse_tool_response(invalid)


def test_prompt_preserves_candidate_without_evidence_and_checks_columns() -> None:
    prompt = build_system_prompt(10)
    assert "Do not output any text outside these two blocks." in prompt
    assert "list_tables" not in prompt
    assert "inspect_tables" not in prompt
    assert "inspect_values" not in prompt
    assert "call submit_sql on EVERY turn" in prompt
    assert "Do not postpone submission until the last few turns" in prompt
    assert "Read the question carefully" in prompt
    assert "submit_sql immediately" in prompt
    assert "does not provide clear evidence" in prompt
    assert "keep the SQL unchanged and submit it" in prompt
    assert "singer.Country, COUNT(*)" in prompt
    assert "extra output column" in prompt


def test_full_schema_and_remaining_calls(sample_db, config, task) -> None:
    del sample_db
    model = ScriptedModelAdapter(
        [
            response("execute_sql", {"sql": "SELECT count(*) FROM employees"}),
            response("submit_sql", {"sql": "SELECT count(*) FROM employees"}),
        ]
    )
    evaluator = ToolXmlEvaluator(replace(config, max_turns=2), model)
    result = evaluator.evaluate(task)

    assert result["correct"] is True
    assert result["status"] == "submitted_sql"
    assert result["turns"][0]["observation"]["turns_remaining"] == 1
    assert result["turns"][0]["observation"]["columns"] == ["count(*)"]
    assert 'CREATE TABLE "employees"' in model.requests[0][1]["content"]
    assert "Exploratory calls remaining: 2" in model.requests[0][1]["content"]
    observations = [
        message["content"]
        for message in model.requests[1]
        if message["content"].startswith("<observation>")
    ]
    assert len(observations) == 1
    assert '"turns_remaining":1' in observations[0]
    assert 'CREATE TABLE "employees"' not in observations[0]
    assert f"<question>\n{task.question}\n</question>" in observations[0]
    assert "<database_schema>" not in observations[0]
    assert "<tool>" in build_system_prompt(2)


def test_budget_exhaustion_forces_last_proposed_sql(sample_db, config, task) -> None:
    del sample_db
    model = ScriptedModelAdapter(
        [
            response("execute_sql", {"sql": "SELECT count(*) FROM employees"}),
            response("execute_sql", {"sql": "SELECT 1"}),
            response("submit_sql", {"sql": "SELECT count(*) FROM employees"}),
        ]
    )
    result = ToolXmlEvaluator(replace(config, max_turns=1), model).evaluate(task)
    assert result["correct"] is False
    assert result["status"] == "forced_submit_sql"
    assert result["final_sql"] == "SELECT 1"
    assert result["turns"][0]["observation"]["turns_remaining"] == 0
    assert result["turns"][1]["forced_submission_reason"] == "wrong_tool_on_final_turn"
    assert result["turns"][2]["source"] == "evaluator_fallback"
    assert result["turns"][2]["tool"] == "submit_sql"
    assert len(model.requests) == 2
    assert any(
        "Final turn: no exploratory calls remain" in message["content"]
        for message in result["messages"]
    )


def test_final_invalid_action_falls_back_to_last_successful_sql(sample_db, config, task):
    del sample_db
    model = ScriptedModelAdapter(["not a tool call"])
    initial_sql = "SELECT count(*) FROM employees"
    result = ToolXmlEvaluator(replace(config, max_turns=1), model).evaluate(task, initial_sql)
    assert result["correct"] is True
    assert result["status"] == "forced_submit_sql"
    assert result["final_sql"] == initial_sql
    assert result["turns"][-1]["reason"] == "invalid_format"


def test_stop_after_tool_restores_closing_tag(sample_db, config, task) -> None:
    del sample_db

    class StoppedModel(ScriptedModelAdapter):
        def generate(self, messages, **kwargs):
            del messages, kwargs
            return Generation(
                text=response("submit_sql", {"sql": "SELECT count(*) FROM employees"})[:-7],
                finish_reason="stop",
            )

    result = ToolXmlEvaluator(config, StoppedModel([])).evaluate(task)
    assert result["correct"] is True
    assert result["turns"][0]["response"].endswith("</tool>")


def test_single_turn_submits_without_replay_or_observation(sample_db, config, task) -> None:
    del sample_db
    model = ScriptedModelAdapter(
        [response("submit_sql", {"sql": "SELECT count(*) FROM employees"})]
    )
    result = ToolXmlEvaluator(config, model).evaluate_single(task)
    assert result["correct"] is True
    assert result["status"] == "submitted_sql"
    assert len(model.requests) == 1
    assert len(result["turns"]) == 1
    assert result["initial_sql"] is None
    assert "Exploratory calls remaining: 0" in model.requests[0][1]["content"]
    assert 'CREATE TABLE "employees"' in model.requests[0][1]["content"]
    assert all(
        "<observation>" not in message["content"]
        for message in result["messages"]
        if message["role"] == "user"
    )


def test_single_turn_does_not_execute_or_fallback(sample_db, config, task) -> None:
    del sample_db
    model = ScriptedModelAdapter(
        [response("execute_sql", {"sql": "SELECT count(*) FROM employees"})]
    )
    result = ToolXmlEvaluator(config, model).evaluate_single(task)
    assert result["correct"] is False
    assert result["status"] == "missing_submission"
    assert result["final_sql"] is None
    assert len(result["turns"]) == 1


def test_summary_counts_completed_records() -> None:
    records = [
        {"difficulty": "easy", "correct": True, "status": "submitted_sql"},
        {"difficulty": "hard", "correct": False, "status": "missing_submission"},
    ]
    summary = summarize(records, 3)
    assert summary["execution_accuracy"] == 0.5
    assert summary["complete"] is False
    assert summary["by_difficulty"]["hard"] == {"count": 1, "correct": 0}


@pytest.mark.parametrize(
    "initial_sql", ["SELECT count(*) FROM employees", "SELECT bad FROM employees"]
)
def test_replay_baseline_sql_and_preserve_observation(sample_db, config, task, initial_sql):
    del sample_db
    model = ScriptedModelAdapter(
        [response("submit_sql", {"sql": "SELECT count(*) FROM employees"})]
    )
    result = ToolXmlEvaluator(replace(config, max_turns=1), model).evaluate(task, initial_sql)
    assert result["correct"] is True
    assert result["initial_sql"] == initial_sql
    assert len(model.requests) == 1
    first = result["turns"][0]
    assert first["source"] == "baseline_replay"
    assert first["arguments"]["sql"] == initial_sql
    assert first["observation"]["turns_remaining"] == 0
    assert first["observation"]["status"] == ("error" if "bad" in initial_sql else "success")
    assert any(message["content"].startswith("<observation>") for message in result["messages"])
    observation_message = result["messages"][3]["content"]
    assert 'CREATE TABLE "employees"' not in observation_message
    assert f"<question>\n{task.question}\n</question>" in observation_message
    assert task.reference_sql not in result["messages"][0]["content"]


def test_baseline_input_checks_dataset_and_hashes_sql(tmp_path, task):
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset_sha256": "dataset"}))
    (tmp_path / "trajectories").mkdir()
    path = tmp_path / "trajectories" / f"{task.task_id}.json"
    path.write_text(json.dumps({"task_id": task.task_id, "sql": "SELECT 1"}))
    with pytest.raises(ValueError, match="dataset hash"):
        load_initial_sql(tmp_path, [task], "other")
    candidates, provenance = load_initial_sql(tmp_path, [task], "dataset")
    assert candidates == {task.task_id: "SELECT 1"}
    path.write_text(json.dumps({"task_id": task.task_id, "sql": "SELECT 2"}))
    _, changed = load_initial_sql(tmp_path, [task], "dataset")
    assert provenance["sql_sha256"] != changed["sql_sha256"]
