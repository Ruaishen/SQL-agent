from __future__ import annotations

from collections import deque

import pytest

from sft.collect_reasoning import build_prompt, collect_trajectory, parse_response
from sql_agent.deepseek import Completion, DeepSeekClient


class FakeTextClient:
    model = "fake-deepseek"

    def __init__(self, *responses: str):
        self.responses = deque(responses)
        self.requests = []

    def complete_text(self, messages, *, temperature, max_tokens):
        self.requests.append([message.copy() for message in messages])
        return Completion(
            message={"role": "assistant", "content": self.responses.popleft()},
            finish_reason="stop",
            request_id=f"request-{len(self.requests)}",
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


def test_prompt_uses_one_tool_tag_for_every_action() -> None:
    prompt = build_prompt(10)
    for name in ("list_tables", "inspect_tables", "inspect_values", "execute_sql", "submit_sql"):
        assert f'<tool>{{"name":"{name}","arguments":' in prompt
    assert "<sql>" not in prompt
    assert "<solution>" not in prompt
    assert "exactly ONE action block" in prompt


def test_text_client_stops_after_tool_block(monkeypatch) -> None:
    client = DeepSeekClient("test-key")

    def fake_request(payload):
        assert "</tool>" in payload["stop"]
        assert "</｜｜DSML｜｜ parameter>" in payload["stop"]
        return Completion(
            message={
                "role": "assistant",
                "content": '<reasoning>Look.</reasoning>'
                '<tool>{"name":"list_tables","arguments":{}}',
            },
            finish_reason="stop",
            request_id="request-1",
            model="fake-deepseek",
            usage={},
        )

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.complete_text([], temperature=0.7, max_tokens=256)
    assert result.message["content"].endswith("</tool>")


def test_parser_accepts_one_tool_and_rejects_extra_actions() -> None:
    reasoning, action = parse_response(
        "<reasoning>Count employees after checking the schema.</reasoning>"
        '<tool>{"name":"execute_sql","arguments":{"sql":"SELECT count(*) FROM employees"}}</tool>'
    )
    assert reasoning.startswith("Count employees")
    assert action == {
        "tool": "execute_sql",
        "arguments": {"sql": "SELECT count(*) FROM employees"},
    }
    with pytest.raises(ValueError):
        parse_response(
            '<reasoning>Try both.</reasoning><tool>{"name":"list_tables","arguments":{}}</tool>'
            '<tool>{"name":"inspect_tables","arguments":{"table_names":["employees"]}}</tool>'
        )
    with pytest.raises(ValueError, match="Invalid reasoning/tool blocks"):
        parse_response("<reasoning>Done.</reasoning><solution>SELECT 1</solution>")
    with pytest.raises(ValueError, match="Invalid reasoning/tool blocks"):
        parse_response('<reasoning>Explore.</reasoning><tool name="list_tables">{}</tool>')
    with pytest.raises(ValueError, match="Invalid reasoning/tool blocks"):
        parse_response(
            '<reasoning>Explore.</reasoning>'
            '<tool>{"name":"list_tables","arguments":{}}</｜｜DSML｜｜ parameter>'
        )


def test_tagged_trajectory_executes_and_returns_observations(sample_db, config, task) -> None:
    del sample_db
    sql = "SELECT count(*) FROM employees"
    client = FakeTextClient(
        '<reasoning>Find available tables.</reasoning>'
        '<tool>{"name":"list_tables","arguments":{}}</tool>',
        "<reasoning>Inspect the employees table.</reasoning>"
        '<tool>{"name":"inspect_tables","arguments":{"table_names":["employees"]}}</tool>',
        f'<reasoning>Count employee rows.</reasoning>'
        f'<tool>{{"name":"execute_sql","arguments":{{"sql":"{sql}"}}}}</tool>',
        f"<reasoning>The count is 25. Submit the tested query.</reasoning>"
        f'<tool>{{"name":"submit_sql","arguments":{{"sql":"{sql}"}}}}</tool>',
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "submitted_sql"
    assert record["correct"]
    assert record["tool_sequence"] == ["list_tables", "inspect_tables", "execute_sql", "submit_sql"]
    assert record["turns"][2]["observation"]["rows"] == [[25]]
    assert client.requests[1][-1]["content"].startswith("<observation>")
    assert client.requests[1][-1]["content"].endswith("</observation>")
    assert all(turn["reasoning"] for turn in record["turns"])


def test_invalid_tagged_response_is_retried_without_execution(sample_db, config, task) -> None:
    del sample_db
    sql = "SELECT count(*) FROM employees"
    client = FakeTextClient(
        '<reasoning>Explore.</reasoning><tool>{"name":"list_tables","arguments":{}}</tool>'
        '<tool>{"name":"inspect_tables","arguments":{"table_names":["employees"]}}</tool>',
        '<reasoning>List tables first.</reasoning>'
        '<tool>{"name":"list_tables","arguments":{}}</tool>',
        f"<reasoning>Submit the query.</reasoning>"
        f'<tool>{{"name":"submit_sql","arguments":{{"sql":"{sql}"}}}}</tool>',
    )
    record = collect_trajectory(task, config, client)
    assert record["status"] == "submitted_sql"
    assert record["correct"]
    assert record["turns"][0]["response_index"] == 2
    assert len(record["rejected_responses"]) == 1
    assert record["tool_sequence"] == ["list_tables", "submit_sql"]
