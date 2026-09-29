from __future__ import annotations

import pytest

from sql_agent.action_parser import (
    ActionParseError,
    InspectTablesAction,
    InspectValuesAction,
    parse_action,
)


def test_valid_action_and_default() -> None:
    action = parse_action(
        '{"tool":"inspect_values","arguments":{"table_name":"employees","column_name":"name"}}'
    )
    assert isinstance(action, InspectValuesAction)
    assert action.arguments.column_name == "name"


def test_inspect_tables_accepts_distinct_names() -> None:
    action = parse_action(
        {"tool": "inspect_tables", "arguments": {"table_names": ["employees", "departments"]}}
    )
    assert isinstance(action, InspectTablesAction)
    assert action.arguments.table_names == ["employees", "departments"]


@pytest.mark.parametrize(
    ("raw", "error_type"),
    [
        ("not-json", "invalid_json"),
        ("[]", "invalid_action"),
        ({"tool": "unknown", "arguments": {}}, "invalid_arguments"),
        ({"tool": "list_tables", "arguments": {}, "extra": 1}, "invalid_arguments"),
        (
            {"tool": "inspect_values", "arguments": {"table_name": "x"}},
            "invalid_arguments",
        ),
        (
            {"tool": "inspect_values", "arguments": {"table_name": "x", "column_name": "c", "limit": 6}},
            "invalid_arguments",
        ),
        ({"tool": "submit", "arguments": {"sql": "SELECT 1"}}, "invalid_arguments"),
        ({"tool": "sample_rows", "arguments": {"table_name": "x"}}, "invalid_arguments"),
        ({"tool": "inspect_tables", "arguments": {"table_names": []}}, "invalid_arguments"),
        (
            {"tool": "inspect_tables", "arguments": {"table_names": ["employees", "EMPLOYEES"]}},
            "invalid_arguments",
        ),
    ],
)
def test_invalid_actions(raw: object, error_type: str) -> None:
    with pytest.raises(ActionParseError) as captured:
        parse_action(raw)  # type: ignore[arg-type]
    assert captured.value.error_type == error_type
