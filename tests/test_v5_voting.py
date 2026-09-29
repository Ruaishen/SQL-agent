from __future__ import annotations

import json
from dataclasses import replace

from evaluation.v5_voting import V5VotingEvaluator, parse_v5_response, vote_sql
from sql_agent.model_adapter import ScriptedModelAdapter


def response(name: str, arguments: dict[str, object]) -> str:
    return (
        "<reasoning>I checked the schema.</reasoning>\n"
        f"<tool>{json.dumps({'name': name, 'arguments': arguments})}</tool>"
    )


def first_messages(question: str, turns: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "The archived v5 tool prompt"},
        {
            "role": "user",
            "content": (
                "<database_schema>\nTABLE employees (id INT)\n</database_schema>\n\n"
                f"<question>\n{question}\n</question>\n"
                f"Exploratory calls remaining: {turns}"
            ),
        },
    ]


def test_v5_parser_accepts_archived_inspection_tools() -> None:
    assert parse_v5_response(response("list_tables", {})).name == "list_tables"
    assert parse_v5_response(
        response("inspect_values", {"table_name": "employees", "column_name": "id"})
    ).name == "inspect_values"


def test_v5_replay_uses_original_context_and_stochastic_seed(sample_db, config, task) -> None:
    del sample_db
    config = replace(config, max_turns=2)
    model = ScriptedModelAdapter(
        [response("submit_sql", {"sql": "SELECT count(*) FROM employees"})]
    )
    result = V5VotingEvaluator(config, model).evaluate(
        task, first_messages(task.question, 2), "SELECT count(*) FROM employees", 3
    )
    assert result["correct"]
    assert result["status"] == "submitted_sql"
    assert len(model.requests) == 1
    assert "turns_remaining" in model.requests[0][3]["content"]
    assert "<database_schema>" in model.requests[0][3]["content"]
    assert f"<question>\n{task.question}\n</question>" in model.requests[0][3]["content"]
    assert model.settings[0]["temperature"] == 0.8
    assert model.settings[0]["seed"] == config.split_seed + 3


def test_v5_unknown_column_feedback_remains_generic(sample_db, config, task) -> None:
    del sample_db
    model = ScriptedModelAdapter(
        [response("submit_sql", {"sql": "SELECT count(*) FROM employees"})]
    )
    result = V5VotingEvaluator(config, model).evaluate(
        task, first_messages(task.question, config.max_turns), "SELECT T2.missing FROM employees", 0
    )
    assert result["correct"]
    observation = result["turns"][0]["observation"]
    assert observation["error_type"] == "unknown_column"
    assert observation["message"] == "Query references an unknown column"
    assert "T2.missing" not in model.requests[0][3]["content"]


def test_vote_groups_full_execution_results(sample_db, config, task) -> None:
    del sample_db
    candidates = [
        {"final_sql": "SELECT 1"},
        {"final_sql": "SELECT 2"},
        {"final_sql": "SELECT 2 AS other"},
        {"final_sql": "SELECT missing FROM employees"},
    ]
    voted = vote_sql(task, candidates, config)
    assert voted == {
        "selected_sample": 1,
        "final_sql": "SELECT 2",
        "votes": 2,
        "valid": 3,
    }
