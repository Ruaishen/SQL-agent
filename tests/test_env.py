from __future__ import annotations

import sqlite3
from dataclasses import replace

from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.prompts import LEGACY_PROMPT_VERSION
from tests.conftest import file_sha256


def test_legacy_prompt_omits_turns_remaining(sample_db, config: EnvConfig, task) -> None:
    env = SQLAgentEnv(config, prompt_version=LEGACY_PROMPT_VERSION)
    try:
        reset = env.reset(task)
        assert reset["max_turns"] == config.max_turns
        assert "turns_remaining" not in reset
        observation, _ = env.step({"tool": "list_tables", "arguments": {}})
        assert "turns_remaining" not in observation
    finally:
        env.close()


def test_tools_observations_and_hidden_gold(sample_db, config: EnvConfig, task) -> None:
    before = file_sha256(sample_db)
    env = SQLAgentEnv(config)
    try:
        reset = env.reset(task)
        assert reset["max_turns"] == config.max_turns
        assert reset["turns_remaining"] == config.max_turns
        assert task.reference_sql not in str(reset)
        tables, done = env.step({"tool": "list_tables", "arguments": {}})
        assert tables["turns_remaining"] == config.max_turns - 1
        assert not done and tables["tables"] == [
            "binary_data",
            "departments",
            "employees",
            "invalid_text",
        ]
        schema, _ = env.step({"tool": "inspect_table", "arguments": {"table_name": "employees"}})
        assert schema["primary_key"] == ["id"]
        assert schema["foreign_keys"] == [
            {"column": "department_id", "references": "departments.id"}
        ]
        values, _ = env.step(
            {"tool": "inspect_values", "arguments": {"table_name": "employees", "column_name": "department_id"}}
        )
        assert values["values"] == [1, 2]
        assert values["returned_value_count"] == 2
    finally:
        env.close()
    assert file_sha256(sample_db) == before


def test_reserved_submission_does_not_consume_exploration_turn(sample_db, config, task) -> None:
    del sample_db
    env = SQLAgentEnv(replace(config, max_turns=2), reserve_final_submission=True)
    try:
        env.reset(task)
        schema, done = env.step(
            {"tool": "inspect_tables", "arguments": {"table_names": ["employees", "departments"]}}
        )
        assert not done and schema["turns_remaining"] == 1
        assert [table["table"] for table in schema["tables"]] == ["employees", "departments"]
        executed, done = env.step(
            {"tool": "execute_sql", "arguments": {"sql": "SELECT count(*) FROM employees"}}
        )
        assert not done and executed["turns_remaining"] == 0
        assert "verification" not in executed
        blocked, done = env.step({"tool": "list_tables", "arguments": {}})
        assert not done and blocked["error_type"] == "exploration_budget_exhausted"
        submitted, done = env.submit_sql({"sql": "SELECT count(*) FROM employees"})
        assert done and submitted["verification"]["correct"]
        assert submitted["termination_reason"] == "submit_sql"
        assert env.turn == 2
    finally:
        env.close()


def test_execute_truncation_and_token_limit(sample_db, config: EnvConfig, task) -> None:
    assert config.max_turns == 10
    assert config.execute_rows_max == 50
    env = SQLAgentEnv(config)
    try:
        env.reset(task)
        observation, _ = env.step(
            {"tool": "execute_sql", "arguments": {"sql": "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 55) SELECT x FROM n"}}
        )
        assert observation["returned_row_count"] == 50
        assert observation["rows"] == [[index] for index in range(1, 51)]
        assert observation["total_row_count"] is None
        assert observation["truncated"]
        assert "row_limit" in observation["truncation_reasons"]
        assert env.token_counter.count_json(observation) <= config.max_observation_tokens
        blob, _ = env.step(
            {"tool": "execute_sql", "arguments": {"sql": "SELECT value FROM binary_data"}}
        )
        assert blob["rows"] == [["0x00ff"]]
    finally:
        env.close()


def test_inspect_values_is_distinct_bounded_and_checks_column(sample_db, config: EnvConfig, task) -> None:
    with sqlite3.connect(sample_db) as connection:
        connection.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?, ?)",
            [(index, f"employee-{index:02d}", 1, 1000.0 + index, "duplicate") for index in range(26, 56)],
        )
    env = SQLAgentEnv(config)
    try:
        env.reset(task)
        values, done = env.step(
            {"tool": "inspect_values", "arguments": {"table_name": "employees", "column_name": "name"}}
        )
        assert not done
        assert values["values"] == [f"employee-{index:02d}" for index in range(1, 51)]
        assert values["returned_value_count"] == 50
        assert values["truncated"]
        assert "value_limit" in values["truncation_reasons"]
        missing, _ = env.step(
            {"tool": "inspect_values", "arguments": {"table_name": "employees", "column_name": "missing"}}
        )
        assert missing["error_type"] == "unknown_column"
    finally:
        env.close()


def test_invalid_json_consumes_turn_without_crashing(sample_db, config: EnvConfig, task) -> None:
    env = SQLAgentEnv(replace(config, max_turns=2))
    try:
        env.reset(task)
        first, done = env.step("not-json")
        assert first["error_type"] == "invalid_json" and not done
        assert first["turns_remaining"] == 1
        second, done = env.step("not-json")
        assert done and second["termination_reason"] == "max_turns"
        assert second["turns_remaining"] == 0
        after, done = env.step({"tool": "list_tables", "arguments": {}})
        assert done and after["error_type"] == "episode_done"
    finally:
        env.close()


def test_non_json_mapping_does_not_crash(sample_db, config: EnvConfig, task) -> None:
    env = SQLAgentEnv(config)
    try:
        env.reset(task)
        observation, done = env.step({"tool": "execute_sql", "arguments": {"sql": object()}})
        assert not done
        assert observation["error_type"] == "invalid_arguments"
    finally:
        env.close()


def test_last_execute_sql_is_verified_at_turn_limit(sample_db, config: EnvConfig, task) -> None:
    env = SQLAgentEnv(replace(config, max_turns=2))
    try:
        env.reset(task)
        first, done = env.step(
            {"tool": "execute_sql", "arguments": {"sql": "SELECT COUNT(id) FROM departments"}}
        )
        assert not done and "verification" not in first
        observation, done = env.step(
            {"tool": "execute_sql", "arguments": {"sql": "SELECT COUNT(id) FROM employees"}}
        )
        assert done
        assert observation["reward"] == 1.0
        assert observation["verification"]["correct"]
        assert observation["termination_reason"] == "final_execute_sql"
    finally:
        env.close()


def test_latest_execute_remains_final_after_non_sql_action(sample_db, config: EnvConfig, task) -> None:
    env = SQLAgentEnv(replace(config, max_turns=2))
    try:
        env.reset(task)
        env.step({"tool": "execute_sql", "arguments": {"sql": "SELECT COUNT(*) FROM employees"}})
        observation, done = env.step({"tool": "list_tables", "arguments": {}})
        assert done
        assert observation["verification"]["correct"]
        assert observation["termination_reason"] == "final_execute_sql"
    finally:
        env.close()
