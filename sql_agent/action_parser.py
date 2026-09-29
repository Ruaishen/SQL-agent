from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArguments(_StrictModel):
    pass


class TableArguments(_StrictModel):
    table_name: str = Field(min_length=1, max_length=256)


class TablesArguments(_StrictModel):
    table_names: list[str] = Field(min_length=1, max_length=8)

    @field_validator("table_names")
    @classmethod
    def distinct_table_names(cls, names: list[str]) -> list[str]:
        if any(not name.strip() or len(name) > 256 for name in names):
            raise ValueError("table names must contain 1 to 256 non-blank characters")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("table names must be distinct")
        return names


class ColumnArguments(TableArguments):
    column_name: str = Field(min_length=1, max_length=256)


class SQLArguments(_StrictModel):
    sql: str = Field(min_length=1, max_length=32768)


class ListTablesAction(_StrictModel):
    tool: Literal["list_tables"]
    arguments: NoArguments


class InspectTableAction(_StrictModel):
    tool: Literal["inspect_table"]
    arguments: TableArguments


class InspectTablesAction(_StrictModel):
    tool: Literal["inspect_tables"]
    arguments: TablesArguments


class InspectValuesAction(_StrictModel):
    tool: Literal["inspect_values"]
    arguments: ColumnArguments


class ExecuteSQLAction(_StrictModel):
    tool: Literal["execute_sql"]
    arguments: SQLArguments


Action = Annotated[
    ListTablesAction
    | InspectTableAction
    | InspectTablesAction
    | InspectValuesAction
    | ExecuteSQLAction,
    Field(discriminator="tool"),
]
_ACTION_ADAPTER = TypeAdapter(Action)


class ActionParseError(ValueError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


def parse_action(raw: str | Mapping[str, Any]) -> Action:
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActionParseError("invalid_json", f"Invalid JSON at character {exc.pos}") from exc
    elif isinstance(raw, Mapping):
        value = dict(raw)
    else:
        raise ActionParseError("invalid_action", "Action must be a JSON object")
    if not isinstance(value, dict):
        raise ActionParseError("invalid_action", "Action must be a JSON object")
    try:
        return _ACTION_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        issue = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in issue["loc"])
        message = f"{location}: {issue['msg']}" if location else issue["msg"]
        raise ActionParseError("invalid_arguments", message) from exc


TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_tables",
        "description": "List user-visible tables in the current SQLite database.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "inspect_table",
        "description": "Inspect columns, primary key, and foreign keys for one table.",
        "parameters": {
            "type": "object",
            "properties": {"table_name": {"type": "string"}},
            "required": ["table_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_tables",
        "description": "Inspect columns, primary keys, and foreign keys for 1 to 8 tables in one call.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                    "uniqueItems": True,
                }
            },
            "required": ["table_names"],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_values",
        "description": "Read distinct values stored in one column of one table, up to the configured limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "column_name": {"type": "string"},
            },
            "required": ["table_name", "column_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "execute_sql",
        "description": "Execute one read-only SQLite query and inspect a bounded result prefix.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
)
