from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, TokenError

from sql_agent.config import EnvConfig
from sql_agent.models import QueryResult, VerifierResult
from sql_agent.sandbox import SQLSandbox, SQLSandboxError, connect_readonly


class ExecutionVerifier:
    def __init__(self, db_path: Path, reference_sql: str, config: EnvConfig):
        self.db_path = db_path
        self.reference_sql = reference_sql
        self.config = config
        self.sandbox = SQLSandbox(config)

    def verify(self, submitted_sql: str) -> VerifierResult:
        order_sensitive = self._is_order_sensitive(self.reference_sql)
        try:
            self.sandbox.validate_sql(submitted_sql)
        except SQLSandboxError as exc:
            return self._failure(
                order_sensitive=order_sensitive,
                valid=False,
                executable=False,
                error=exc.error_type,
            )
        try:
            predicted = self._execute_full(submitted_sql)
        except SQLSandboxError as exc:
            return self._failure(
                order_sensitive=order_sensitive,
                valid=True,
                executable=False,
                error=exc.error_type,
            )
        if submitted_sql == self.reference_sql:
            expected = predicted
        else:
            try:
                expected = self._execute_full(self.reference_sql)
            except SQLSandboxError as exc:
                return self._failure(
                    order_sensitive=order_sensitive,
                    valid=True,
                    executable=True,
                    error=f"verifier_reference_{exc.error_type}",
                )
        column_match = len(predicted.columns) == len(expected.columns)
        row_count_match = len(predicted.rows) == len(expected.rows)
        value_match = column_match and row_count_match and self._compare_rows(
            predicted.rows, expected.rows, order_sensitive=order_sensitive
        )
        correct = column_match and row_count_match and value_match
        return VerifierResult(
            reward=1.0 if correct else 0.0,
            correct=correct,
            agent_sql_valid=True,
            agent_sql_executable=True,
            column_match=column_match,
            row_count_match=row_count_match,
            value_match=value_match,
            order_sensitive=order_sensitive,
            error=None if correct else "wrong_result",
        )

    def _execute_full(self, sql: str) -> QueryResult:
        connection = connect_readonly(self.db_path)
        try:
            bounded = self.sandbox.execute(
                connection,
                sql,
                row_limit=self.config.verifier_max_rows,
                timeout_seconds=self.config.verifier_timeout_seconds,
                byte_limit=self.config.verifier_max_bytes,
            )
            if bounded.has_more:
                raise SQLSandboxError("result_too_large", "Verifier row limit exceeded")
            return bounded.result
        finally:
            connection.close()

    @staticmethod
    def _is_order_sensitive(sql: str) -> bool:
        try:
            statement = parse_one(sql, read="sqlite")
        except (ParseError, TokenError):
            return False
        return isinstance(statement, exp.Query) and statement.args.get("order") is not None

    @classmethod
    def _compare_rows(
        cls,
        predicted: tuple[tuple[Any, ...], ...],
        expected: tuple[tuple[Any, ...], ...],
        *,
        order_sensitive: bool,
    ) -> bool:
        if order_sensitive:
            return all(
                cls._row_equal(left, right)
                for left, right in zip(predicted, expected, strict=True)
            )
        try:
            if Counter(predicted) == Counter(expected):
                return True
        except TypeError:
            pass
        remaining = list(expected)
        for row in predicted:
            match = next(
                (
                    index
                    for index, candidate in enumerate(remaining)
                    if cls._row_equal(row, candidate)
                ),
                None,
            )
            if match is None:
                return False
            remaining.pop(match)
        return not remaining

    @classmethod
    def _row_equal(cls, left: Iterable[Any], right: Iterable[Any]) -> bool:
        left_values = tuple(left)
        right_values = tuple(right)
        return len(left_values) == len(right_values) and all(
            cls._value_equal(a, b) for a, b in zip(left_values, right_values, strict=True)
        )

    @staticmethod
    def _value_equal(left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is right
        if isinstance(left, Real) and isinstance(right, Real):
            return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
        return type(left) is type(right) and left == right

    @staticmethod
    def _failure(
        *, order_sensitive: bool, valid: bool, executable: bool, error: str
    ) -> VerifierResult:
        return VerifierResult(
            reward=0.0,
            correct=False,
            agent_sql_valid=valid,
            agent_sql_executable=executable,
            column_match=False,
            row_count_match=False,
            value_match=False,
            order_sensitive=order_sensitive,
            error=error,
        )
