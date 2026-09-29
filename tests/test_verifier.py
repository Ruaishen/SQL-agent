from __future__ import annotations

from sql_agent.config import EnvConfig
from sql_agent.verifier import ExecutionVerifier


def test_oracle_and_equivalent_sql_pass(sample_db, config: EnvConfig) -> None:
    verifier = ExecutionVerifier(sample_db, "SELECT count(*) FROM employees", config)
    assert verifier.verify("SELECT COUNT(id) FROM employees").correct


def test_invalid_and_wrong_sql_fail(sample_db, config: EnvConfig) -> None:
    verifier = ExecutionVerifier(sample_db, "SELECT count(*) FROM employees", config)
    invalid = verifier.verify("")
    wrong = verifier.verify("SELECT count(*) FROM employees WHERE id < 5")
    assert not invalid.correct and not invalid.agent_sql_valid
    assert not wrong.correct and wrong.agent_sql_executable


def test_order_semantics(sample_db, config: EnvConfig) -> None:
    unordered = ExecutionVerifier(sample_db, "SELECT id FROM employees WHERE id <= 3", config)
    ordered = ExecutionVerifier(
        sample_db, "SELECT id FROM employees WHERE id <= 3 ORDER BY id", config
    )
    reverse = "SELECT id FROM employees WHERE id <= 3 ORDER BY id DESC"
    assert unordered.verify(reverse).correct
    result = ordered.verify(reverse)
    assert not result.correct and result.order_sensitive


def test_duplicate_rows_are_preserved(sample_db, config: EnvConfig) -> None:
    verifier = ExecutionVerifier(
        sample_db, "SELECT department_id FROM employees WHERE id <= 4", config
    )
    assert not verifier.verify(
        "SELECT DISTINCT department_id FROM employees WHERE id <= 4"
    ).correct


def test_float_tolerance(sample_db, config: EnvConfig) -> None:
    verifier = ExecutionVerifier(sample_db, "SELECT 1.0", config)
    assert verifier.verify("SELECT 1.0000005").correct
    assert not verifier.verify("SELECT 1.01").correct
