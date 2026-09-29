from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sqlglot import exp, parse
from sqlglot.errors import ParseError, TokenError

from sql_agent.config import EnvConfig
from sql_agent.models import QueryResult


class SQLSandboxError(ValueError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


@dataclass(frozen=True, slots=True)
class BoundedQueryResult:
    result: QueryResult
    has_more: bool
    approximate_bytes: int


def _readonly_uri(path: Path) -> str:
    return f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"


def _decode_sqlite_text(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {path}")
    connection = sqlite3.connect(_readonly_uri(path), uri=True, check_same_thread=False)
    connection.text_factory = _decode_sqlite_text
    connection.execute("PRAGMA query_only=ON")
    connection.enable_load_extension(False)
    return connection


def _sqlite_authorizer_codes(names: tuple[str, ...]) -> frozenset[int]:
    return frozenset(
        code for name in names if (code := getattr(sqlite3, name, None)) is not None
    )


class SQLSandbox:
    _DENIED_ACTIONS = _sqlite_authorizer_codes(
        (
            "SQLITE_INSERT",
            "SQLITE_UPDATE",
            "SQLITE_DELETE",
            "SQLITE_ALTER_TABLE",
            "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TABLE",
            "SQLITE_CREATE_TEMP_INDEX",
            "SQLITE_CREATE_TEMP_TABLE",
            "SQLITE_CREATE_TEMP_TRIGGER",
            "SQLITE_CREATE_TEMP_VIEW",
            "SQLITE_CREATE_TRIGGER",
            "SQLITE_CREATE_VIEW",
            "SQLITE_CREATE_VTABLE",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TEMP_INDEX",
            "SQLITE_DROP_TEMP_TABLE",
            "SQLITE_DROP_TEMP_TRIGGER",
            "SQLITE_DROP_TEMP_VIEW",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_DROP_VIEW",
            "SQLITE_DROP_VTABLE",
            "SQLITE_ATTACH",
            "SQLITE_DETACH",
            "SQLITE_TRANSACTION",
            "SQLITE_SAVEPOINT",
            "SQLITE_PRAGMA",
            "SQLITE_REINDEX",
            "SQLITE_ANALYZE",
        )
    )
    _DENIED_FUNCTIONS = frozenset(
        {"load_extension", "readfile", "writefile", "fts3_tokenizer", "edit", "shell"}
    )

    def __init__(self, config: EnvConfig):
        self.config = config

    def validate_sql(self, sql: str) -> exp.Query:
        if not isinstance(sql, str) or not sql.strip():
            raise SQLSandboxError("empty_sql", "SQL must be a non-empty string")
        if "\x00" in sql:
            raise SQLSandboxError("parse_error", "SQL cannot contain NUL characters")
        if len(sql) > self.config.max_sql_chars:
            raise SQLSandboxError("sql_too_long", "SQL exceeds the configured length limit")
        try:
            statements = [
                statement for statement in parse(sql, read="sqlite") if statement is not None
            ]
        except (ParseError, TokenError) as exc:
            raise SQLSandboxError("parse_error", "SQL could not be parsed") from exc
        if len(statements) != 1:
            raise SQLSandboxError("multiple_statements", "Exactly one SQL statement is required")
        statement = statements[0]
        if not isinstance(statement, exp.Query):
            raise SQLSandboxError("forbidden_sql", "Only read-only SELECT queries are allowed")
        if statement.find(exp.Into) is not None:
            raise SQLSandboxError("forbidden_sql", "SELECT INTO is not allowed")
        return statement

    def _authorizer(
        self,
        action: int,
        parameter1: str | None,
        parameter2: str | None,
        _database: str | None,
        _source: str | None,
    ) -> int:
        if action in self._DENIED_ACTIONS:
            return sqlite3.SQLITE_DENY
        if (
            action == sqlite3.SQLITE_READ
            and parameter1
            and parameter1.casefold().startswith("sqlite_")
        ):
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = (parameter2 or parameter1 or "").casefold()
            if function_name in self._DENIED_FUNCTIONS:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def execute(
        self,
        connection: sqlite3.Connection,
        sql: str,
        *,
        row_limit: int,
        timeout_seconds: float | None = None,
        byte_limit: int | None = None,
    ) -> BoundedQueryResult:
        self.validate_sql(sql)
        timeout = timeout_seconds or self.config.query_timeout_seconds
        deadline = time.monotonic() + timeout
        connection.set_authorizer(self._authorizer)
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1_000)
        cursor: sqlite3.Cursor | None = None
        try:
            cursor = connection.execute(sql)
            columns = tuple(description[0] for description in (cursor.description or ()))
            rows: list[tuple[Any, ...]] = []
            approximate_bytes = 0
            has_more = False
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                if len(rows) >= row_limit:
                    has_more = True
                    break
                normalized_row = tuple(row)
                approximate_bytes += sum(
                    len(value) if isinstance(value, bytes) else len(str(value)) for value in row
                )
                if byte_limit is not None and approximate_bytes > byte_limit:
                    raise SQLSandboxError("result_too_large", "Query result exceeds the byte limit")
                rows.append(normalized_row)
            return BoundedQueryResult(
                result=QueryResult(columns=columns, rows=tuple(rows)),
                has_more=has_more,
                approximate_bytes=approximate_bytes,
            )
        except SQLSandboxError:
            raise
        except sqlite3.DatabaseError as exc:
            raise self._map_database_error(exc) from exc
        finally:
            if cursor is not None:
                cursor.close()
            connection.set_progress_handler(None, 0)
            connection.set_authorizer(None)

    @staticmethod
    def _map_database_error(exc: sqlite3.DatabaseError) -> SQLSandboxError:
        raw_message = str(exc)
        message = raw_message.casefold()
        if "interrupted" in message:
            return SQLSandboxError("timeout", "Query exceeded the execution deadline")
        if "not authorized" in message or "authorization denied" in message:
            return SQLSandboxError("forbidden_sql", "Query attempted a forbidden operation")
        if "no such table" in message:
            return SQLSandboxError("unknown_table", "Query references an unknown table")
        if "no such column" in message:
            column = raw_message.partition(":")[2].strip()
            detail = f": {column}" if column else ""
            return SQLSandboxError("unknown_column", f"Query references an unknown column{detail}")
        if "ambiguous column" in message:
            return SQLSandboxError("ambiguous_column", "Query references an ambiguous column")
        if "syntax error" in message or "incomplete input" in message:
            return SQLSandboxError("syntax_error", "SQLite rejected the SQL syntax")
        return SQLSandboxError("execution_error", "SQLite could not execute the query")
