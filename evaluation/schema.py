from __future__ import annotations

import json
from pathlib import Path

from sqlglot import exp, parse_one

from sql_agent.sandbox import connect_readonly
from sql_agent.tools import quote_identifier

SCHEMA_FORMAT = "sqlite_ddl_example_values_v1"
MAX_EXAMPLE_VALUES = 2
MAX_EXAMPLE_CHARS = 40


def oracle_table_names(reference_sql: str) -> set[str]:
    statement = parse_one(reference_sql, read="sqlite")
    return {table.name.casefold() for table in statement.find_all(exp.Table) if table.name}


def render_schema(db_path: Path, only_tables: set[str] | None = None) -> str:
    connection = connect_readonly(db_path)
    try:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name COLLATE BINARY"
            )
            if only_tables is None or row[0].casefold() in only_tables
        ]
        blocks: list[str] = []
        for name in names:
            columns = connection.execute("SELECT * FROM pragma_table_info(?)", (name,)).fetchall()
            foreign_keys = connection.execute(
                "SELECT * FROM pragma_foreign_key_list(?)", (name,)
            ).fetchall()
            definitions: list[tuple[str, str]] = []
            for column in columns:
                quoted_column = quote_identifier(column[1])
                definition = quoted_column
                if column[2]:
                    definition += f" {column[2]}"
                if column[3]:
                    definition += " NOT NULL"
                examples = connection.execute(
                    f"SELECT DISTINCT {quoted_column} FROM {quote_identifier(name)} "
                    f"WHERE {quoted_column} IS NOT NULL AND {quoted_column} != '' "
                    f"ORDER BY {quoted_column} LIMIT ?",
                    (MAX_EXAMPLE_VALUES,),
                ).fetchall()
                values = [
                    value[:MAX_EXAMPLE_CHARS] if isinstance(value, str) else value
                    for (value,) in examples
                    if not isinstance(value, bytes)
                ]
                annotation = f" -- examples: {json.dumps(values, ensure_ascii=False)}" if values else ""
                definitions.append((definition, annotation))
            primary_key = [
                quote_identifier(column[1])
                for column in sorted(columns, key=lambda item: item[5])
                if column[5]
            ]
            if primary_key:
                definitions.append((f"PRIMARY KEY ({', '.join(primary_key)})", ""))
            grouped_keys: dict[int, list[tuple]] = {}
            for foreign_key in foreign_keys:
                grouped_keys.setdefault(foreign_key[0], []).append(foreign_key)
            for group in grouped_keys.values():
                parts = sorted(group, key=lambda item: item[1])
                source_columns = ", ".join(quote_identifier(part[3]) for part in parts)
                target_columns = [part[4] for part in parts]
                target = quote_identifier(parts[0][2])
                if all(target_columns):
                    target += " (" + ", ".join(quote_identifier(column) for column in target_columns) + ")"
                definitions.append((f"FOREIGN KEY ({source_columns}) REFERENCES {target}", ""))
            lines = [
                f"  {definition}{',' if index < len(definitions) - 1 else ''}{annotation}"
                for index, (definition, annotation) in enumerate(definitions)
            ]
            blocks.append(f"CREATE TABLE {quote_identifier(name)} (\n" + "\n".join(lines) + "\n);")
        return "\n\n".join(blocks)
    finally:
        connection.close()
