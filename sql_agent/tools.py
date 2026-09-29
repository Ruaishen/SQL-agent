from __future__ import annotations

import sqlite3
from typing import Any

from sql_agent.config import EnvConfig
from sql_agent.sandbox import SQLSandbox, SQLSandboxError
from sql_agent.truncation import ObservationLimiter


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class SQLTools:
    def __init__(
        self,
        connection: sqlite3.Connection,
        config: EnvConfig,
        sandbox: SQLSandbox,
        limiter: ObservationLimiter,
    ):
        self.connection = connection
        self.config = config
        self.sandbox = sandbox
        self.limiter = limiter

    def list_tables(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name COLLATE BINARY"
        ).fetchall()
        tables = [row[0] for row in rows]
        reasons: list[str] = []
        if len(tables) > self.config.max_tables:
            tables = tables[: self.config.max_tables]
            reasons.append("table_limit")
        return self.limiter.limit(
            {"status": "success", "tables": tables, "truncation_reasons": reasons}
        )

    def inspect_table(self, table_name: str) -> dict[str, Any]:
        return self.limiter.limit(self._inspect_table_data(table_name))

    def inspect_tables(self, table_names: list[str]) -> dict[str, Any]:
        tables: list[dict[str, Any]] = []
        for table_name in table_names:
            try:
                tables.append(self._inspect_table_data(table_name))
            except SQLSandboxError as exc:
                tables.append(
                    {
                        "status": "error",
                        "table": table_name,
                        "error_type": exc.error_type,
                        "message": exc.message,
                    }
                )
        return self.limiter.limit(
            {
                "status": "success",
                "tables": tables,
                "requested_table_count": len(table_names),
                "returned_table_count": len(tables),
            }
        )

    def _inspect_table_data(self, table_name: str) -> dict[str, Any]:
        self._require_table(table_name)
        columns_raw = self.connection.execute(
            "SELECT * FROM pragma_table_info(?)", (table_name,)
        ).fetchall()
        fks_raw = self.connection.execute(
            "SELECT * FROM pragma_foreign_key_list(?)", (table_name,)
        ).fetchall()
        columns = [
            {"name": row[1], "type": row[2] or "", "nullable": not bool(row[3])}
            for row in columns_raw
        ]
        primary_key = [row[1] for row in sorted(columns_raw, key=lambda row: row[5]) if row[5] > 0]
        foreign_keys = [
            {"column": row[3], "references": f"{row[2]}.{row[4]}"}
            for row in sorted(fks_raw, key=lambda row: (row[2], row[3], row[4]))
        ]
        reasons: list[str] = []
        if len(columns) > self.config.max_columns:
            columns = columns[: self.config.max_columns]
            reasons.append("column_limit")
        if len(foreign_keys) > self.config.max_foreign_keys:
            foreign_keys = foreign_keys[: self.config.max_foreign_keys]
            reasons.append("foreign_key_limit")
        return {
            "status": "success",
            "table": table_name,
            "columns": columns,
            "primary_key": primary_key,
            "foreign_keys": foreign_keys,
            "truncated": bool(reasons),
            "truncation_reasons": reasons,
        }

    def inspect_values(self, table_name: str, column_name: str) -> dict[str, Any]:
        self._require_table(table_name)
        table_info = self.connection.execute(
            "SELECT * FROM pragma_table_info(?)", (table_name,)
        ).fetchall()
        column = next(
            (str(row[1]) for row in table_info if str(row[1]).casefold() == column_name.casefold()),
            None,
        )
        if column is None:
            raise SQLSandboxError("unknown_column", "Tool references an unknown column")
        sql = (
            f"SELECT DISTINCT {quote_identifier(column)} FROM {quote_identifier(table_name)} "
            "ORDER BY 1"
        )
        bounded = self.sandbox.execute(
            self.connection,
            sql,
            row_limit=self.config.inspect_values_max,
            timeout_seconds=self.config.query_timeout_seconds,
        )
        reasons = ["value_limit"] if bounded.has_more else []
        return self.limiter.limit(
            {
                "status": "success",
                "table": table_name,
                "column": column,
                "values": [row[0] for row in bounded.result.rows],
                "returned_value_count": len(bounded.result.rows),
                "truncation_reasons": reasons,
            }
        )

    def execute_sql(self, sql: str) -> dict[str, Any]:
        bounded = self.sandbox.execute(
            self.connection,
            sql,
            row_limit=self.config.execute_rows_max,
            timeout_seconds=self.config.query_timeout_seconds,
        )
        reasons = ["row_limit"] if bounded.has_more else []
        return self.limiter.limit(
            {
                "status": "success",
                "columns": list(bounded.result.columns),
                "rows": [list(row) for row in bounded.result.rows],
                "returned_row_count": len(bounded.result.rows),
                "total_row_count": None if bounded.has_more else len(bounded.result.rows),
                "truncation_reasons": reasons,
            }
        )

    def error(self, error: SQLSandboxError) -> dict[str, Any]:
        return self.limiter.limit(
            {"status": "error", "error_type": error.error_type, "message": error.message}
        )

    def _require_table(self, table_name: str) -> None:
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = ? AND name NOT LIKE 'sqlite_%'",
            (table_name,),
        ).fetchone()
        if row is None:
            raise SQLSandboxError("unknown_table", "Tool references an unknown table")
