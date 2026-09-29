from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from evaluation.logging import load_jsonl
from evaluation.replay import replay_record
from evaluation.run_eval import select_tasks
from evaluation.runner import BaselineMode, EvaluationRunner, RunnerConfig, extract_sql
from evaluation.schema import oracle_table_names, render_schema
from sql_agent.model_adapter import OpenAICompatibleAdapter, ScriptedModelAdapter
from sql_agent.models import VerifierResult
from sql_agent.prompts import (
    LEGACY_PROMPT_VERSION,
    PROMPT_VERSION,
    build_direct_sql_prompt,
    build_system_prompt,
)


def test_schema_rendering_and_oracle_tables(sample_db: Path, task) -> None:
    assert oracle_table_names(task.reference_sql) == {"employees"}
    full = render_schema(sample_db)
    oracle = render_schema(sample_db, {"employees"})
    assert 'CREATE TABLE "departments"' in full
    assert 'CREATE TABLE "employees"' in oracle
    assert 'CREATE TABLE "departments"' not in oracle
    assert '"name" TEXT NOT NULL, -- examples: ["Engineering", "Sales"]' in full
    assert 'PRIMARY KEY ("id")' in oracle
    assert 'FOREIGN KEY ("department_id") REFERENCES "departments" ("id")' in oracle


def test_deterministic_stratified_selection(task) -> None:
    tasks = []
    for difficulty in ("easy", "medium", "hard", "extra"):
        for index in range(3):
            tasks.append(
                type(task)(
                    task_id=f"{difficulty}-{index}",
                    source=task.source,
                    db_id=task.db_id,
                    db_path=task.db_path,
                    question=task.question,
                    reference_sql=task.reference_sql,
                    difficulty=difficulty,
                    split=task.split,
                )
            )
    first = select_tasks(tasks, limit=0, per_difficulty=2, seed=42)
    second = select_tasks(tasks, limit=0, per_difficulty=2, seed=42)
    assert [item.task_id for item in first] == [item.task_id for item in second]
    assert {
        difficulty: sum(item.difficulty == difficulty for item in first)
        for difficulty in {"easy", "medium", "hard", "extra"}
    } == {"easy": 2, "medium": 2, "hard": 2, "extra": 2}


def test_extract_sql_accepts_plain_fenced_and_sql_json() -> None:
    sql = "SELECT count(*) FROM employees"
    assert extract_sql(sql) == sql
    assert extract_sql(f"```sql\n{sql}\n```") == sql
    assert extract_sql(json.dumps({"sql": sql})) == sql
    wrong = VerifierResult(0.0, False, True, True, True, False, False, False, "wrong_result")
    classify = EvaluationRunner._failure_category
    assert classify(wrong, "SELECT count(*) FROM departments", sql) == "wrong_table"
    assert classify(wrong, "SELECT sum(id) FROM employees", sql) == "aggregation_error"
    assert (
        classify(
            wrong,
            "SELECT count(*) FROM employees WHERE id > 10",
            sql,
        )
        == "condition_error"
    )
    assert (
        classify(
            wrong,
            "SELECT count(*) FROM employees JOIN departments",
            "SELECT count(*) FROM employees JOIN departments ON department_id = departments.id",
        )
        == "join_error"
    )


def test_turn_budget_prompt_contract() -> None:
    schema_messages = build_direct_sql_prompt("How many employees?", "TABLE employees (id INT)")
    no_schema_messages = build_direct_sql_prompt("How many employees?", None)
    agent_prompt = build_system_prompt(10)

    assert PROMPT_VERSION == "baseline_v4_inspect_values"
    assert "Return SQL only" in schema_messages[0]["content"]
    assert "<database_schema>" in schema_messages[1]["content"]
    assert "TABLE employees" in schema_messages[1]["content"]
    assert "<database_schema>" not in no_schema_messages[1]["content"]
    assert "No database schema is available" in no_schema_messages[0]["content"]
    assert "Start with list_tables" in agent_prompt
    assert "inspect_table" in agent_prompt
    assert "inspect_values" in agent_prompt
    assert "most recent execute_sql query is the final answer" in agent_prompt
    assert "final allowed turn for execute_sql" in agent_prompt
    assert "<think>" not in agent_prompt
    assert "at most 10 assistant turns" in agent_prompt
    assert "turns_remaining" in agent_prompt
    legacy = build_system_prompt(6, prompt_version=LEGACY_PROMPT_VERSION)
    assert "Finish within 6 turns" in legacy
    assert "turns_remaining" not in legacy
    assert "inspect_values" in legacy


def test_all_baselines_logging_summary_and_replay(tmp_path, sample_db, config, task) -> None:
    del sample_db
    correct = "SELECT count(*) FROM employees"
    wrong = "SELECT count(*) FROM departments"
    model = ScriptedModelAdapter(
        [
            correct,
            wrong,
            json.dumps({"tool": "execute_sql", "arguments": {"sql": wrong}}),
            json.dumps({"tool": "execute_sql", "arguments": {"sql": correct}}),
            correct,
        ]
    )
    config = replace(config, max_turns=2)
    runner = EvaluationRunner(config, model, RunnerConfig(seed=7))
    log_path = tmp_path / "trajectories.jsonl"
    summary_path = tmp_path / "summary.json"
    modes = list(BaselineMode)
    summary = runner.run(
        [task], modes, log_path=log_path, summary_path=summary_path, run_id="test-run"
    )
    records = load_jsonl(log_path)

    assert len(records) == 4
    assert summary["overall"]["successes"] == 3
    assert summary["overall"]["execution_accuracy"] == 0.75
    assert summary["overall"]["recovery_rate"] == 0.5
    assert summary["overall"]["failure_categories"] == {"wrong_table": 1}
    assert summary["prompt_version"] == "baseline_v4_inspect_values"
    assert summary["parameters"] == {
        "agent_max_turns": config.max_turns,
        "enable_thinking": False,
        "max_tokens": 512,
        "min_p": 0.0,
        "seed": 7,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "workers": 1,
    }
    assert records[2]["mode"] == "multi_turn_agent"
    assert records[2]["first_sql_correct"] is False
    assert records[2]["recovered_after_error"] is True
    assert records[2]["tool_calls"] == ["execute_sql", "execute_sql"]
    assert 'CREATE TABLE "departments"' in model.requests[0][1]["content"]
    assert "<database_schema>" not in model.requests[1][1]["content"]
    assert "No database schema is available" in model.requests[1][0]["content"]
    assert 'CREATE TABLE "employees"' in model.requests[-1][1]["content"]
    assert 'CREATE TABLE "departments"' not in model.requests[-1][1]["content"]
    assert all("reference_sql" not in row for row in records)
    assert all(replay_record(row, task, config)["matches"] for row in records)
    assert json.loads(summary_path.read_text())["run_id"] == "test-run"


def test_openai_compatible_adapter_request(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "SELECT 1"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenAICompatibleAdapter(
        "http://baseline.test:8000/v1", "Qwen3-1.7B", api_key="secret", timeout_seconds=9
    )
    result = adapter.generate(
        [{"role": "user", "content": "question"}],
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=64,
        seed=42,
        enable_thinking=False,
    )
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["seed"] == 42
    assert captured["payload"]["temperature"] == 0.7
    assert captured["payload"]["top_p"] == 0.8
    assert captured["payload"]["top_k"] == 20
    assert captured["payload"]["min_p"] == 0.0
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 9
    assert result.text == "SELECT 1"
    assert result.prompt_tokens == 5 and result.completion_tokens == 2


def test_local_model_endpoint_bypasses_proxy() -> None:
    adapter = OpenAICompatibleAdapter("http://127.0.0.1:8000/v1", "Qwen3-1.7B")
    assert adapter._opener is not None
