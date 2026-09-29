from __future__ import annotations

import sqlite3

import pytest

from sql_agent.config import EnvConfig
from sql_agent.sandbox import SQLSandbox, SQLSandboxError, connect_readonly
from tests.conftest import file_sha256


def test_read_queries_and_set_operations(sample_db, config: EnvConfig) -> None:
    sandbox = SQLSandbox(config)
    connection = connect_readonly(sample_db)
    try:
        result = sandbox.execute(connection, "SELECT count(*) FROM employees", row_limit=20)
        assert result.result.rows == ((25,),)
        cte = sandbox.execute(
            connection,
            "WITH selected AS (SELECT id FROM employees WHERE id <= 2) SELECT id FROM selected",
            row_limit=20,
        )
        assert cte.result.rows == ((1,), (2,))
        union = sandbox.execute(connection, "SELECT 1 UNION SELECT 2", row_limit=20)
        assert union.result.rows == ((1,), (2,))
        invalid_utf8 = sandbox.execute(connection, "SELECT value FROM invalid_text", row_limit=20)
        assert invalid_utf8.result.rows == (("�",),)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO employees(name) VALUES ('bad')",
        "UPDATE employees SET name = 'bad'",
        "DELETE FROM employees",
        "DROP TABLE employees",
        "CREATE TABLE bad(id INTEGER)",
        "PRAGMA table_info(employees)",
        "ATTACH DATABASE '/tmp/other.sqlite' AS other",
        "SELECT 1; SELECT 2",
        "SELECT * FROM sqlite_master",
        "SELECT load_extension('bad')",
    ],
)
def test_forbidden_sql_does_not_modify_database(sample_db, config: EnvConfig, sql: str) -> None:
    before = file_sha256(sample_db)
    sandbox = SQLSandbox(config)
    connection = connect_readonly(sample_db)
    try:
        with pytest.raises(SQLSandboxError):
            sandbox.execute(connection, sql, row_limit=20)
    finally:
        connection.close()
    assert file_sha256(sample_db) == before


def test_row_limit_boundary(sample_db, config: EnvConfig) -> None:
    sandbox = SQLSandbox(config)
    connection = connect_readonly(sample_db)
    try:
        twenty = sandbox.execute(
            connection, "SELECT id FROM employees WHERE id <= 20", row_limit=20
        )
        twenty_one = sandbox.execute(
            connection, "SELECT id FROM employees WHERE id <= 21", row_limit=20
        )
    finally:
        connection.close()
    assert len(twenty.result.rows) == 20 and not twenty.has_more
    assert len(twenty_one.result.rows) == 20 and twenty_one.has_more


def test_recursive_query_times_out(sample_db, config: EnvConfig) -> None:
    sandbox = SQLSandbox(config)
    connection = connect_readonly(sample_db)
    sql = (
        "WITH RECURSIVE counter(x) AS (VALUES(1) UNION ALL "
        "SELECT x + 1 FROM counter) SELECT sum(x) FROM counter"
    )
    try:
        with pytest.raises(SQLSandboxError) as captured:
            sandbox.execute(connection, sql, row_limit=20, timeout_seconds=0.001)
    finally:
        connection.close()
    assert captured.value.error_type == "timeout"


def test_unterminated_string_is_a_parse_error(sample_db, config: EnvConfig) -> None:
    connection = connect_readonly(sample_db)
    try:
        with pytest.raises(SQLSandboxError) as captured:
            SQLSandbox(config).execute(
                connection,
                "SELECT * FROM employees WHERE name = 'unterminated",
                row_limit=10,
            )
    finally:
        connection.close()
    assert captured.value.error_type == "parse_error"


def test_unknown_column_error_names_column(sample_db, config: EnvConfig) -> None:
    connection = connect_readonly(sample_db)
    try:
        with pytest.raises(SQLSandboxError) as captured:
            SQLSandbox(config).execute(
                connection, "SELECT T2.missing FROM employees AS T2", row_limit=10
            )
    finally:
        connection.close()
    assert captured.value.error_type == "unknown_column"
    assert captured.value.message == "Query references an unknown column: T2.missing"


def test_readonly_connection_rejects_direct_write(sample_db) -> None:
    connection = connect_readonly(sample_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("UPDATE employees SET name = 'bad'")
    finally:
        connection.close()
