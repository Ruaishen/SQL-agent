import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from evaluation.run_reasoning_sft import evaluate_batch


@pytest.fixture(autouse=True)
def scripted_sampling_parameters(monkeypatch):
    # These tests exercise SQLite episodes with a scripted LLM, not vLLM execution.
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, prompts, params, use_tqdm):
        self.calls += 1
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text=next(self.responses), finish_reason="stop")]
            )
            for prompt in prompts
        ]


class Tokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append([dict(message) for message in messages])
        return "prompt"


def response(name, arguments):
    action = json.dumps({"name": name, "arguments": arguments})
    return f"<reasoning>Use the requested tool.</reasoning><tool>{action}</tool>"


@pytest.mark.parametrize(
    "last_sql,correct",
    [
        ("SELECT count(*) FROM employees", True),
        ("SELECT count(*) FROM nonexistent_table", False),
    ],
)
def test_budget_forces_latest_executed_sql_even_if_it_failed(
    sample_db, config, task, last_sql, correct
):
    llm = ScriptedLLM(
        [
            response("execute_sql", {"sql": "SELECT count(*) FROM employees"}),
            response("execute_sql", {"sql": last_sql}),
            "invalid response",
        ]
    )
    records = evaluate_batch([task], llm, Tokenizer(), "prompt", replace(config, max_turns=2), 512)
    record = records[0]
    assert llm.calls == 3
    assert record["status"] == "forced_submit_sql"
    assert record["forced_submission"] is True
    assert record["final_sql"] == last_sql
    assert record["correct"] is correct
    assert record["turns"][-1]["source"] == "evaluator_fallback"
    assert record["turns"][-1]["tool"] == "submit_sql"


def test_other_last_tool_does_not_erase_previous_sql(sample_db, config, task):
    sql = "SELECT count(*) FROM employees"
    llm = ScriptedLLM(
        [
            response("execute_sql", {"sql": sql}),
            response("list_tables", {}),
            response("execute_sql", {"sql": "SELECT 0"}),
        ]
    )
    record = evaluate_batch([task], llm, Tokenizer(), "prompt", replace(config, max_turns=2), 512)[
        0
    ]
    assert record["final_sql"] == sql
    assert record["correct"] is True
    assert llm.calls == 3


def test_budget_without_executed_sql_does_not_invent_submission(sample_db, config, task):
    llm = ScriptedLLM([response("list_tables", {}), "invalid response"])
    record = evaluate_batch([task], llm, Tokenizer(), "prompt", replace(config, max_turns=1), 512)[
        0
    ]
    assert record["status"] == "missing_submission"
    assert record["final_sql"] is None
    assert record["forced_submission"] is False
    assert len(record["turns"]) == 2
    assert llm.calls == 2


def test_model_submission_before_budget_is_preserved(sample_db, config, task):
    llm = ScriptedLLM([response("submit_sql", {"sql": "SELECT count(*) FROM employees"})])
    record = evaluate_batch([task], llm, Tokenizer(), "prompt", replace(config, max_turns=1), 512)[
        0
    ]
    assert record["status"] == "submitted_sql"
    assert record["correct"] is True
    assert record["forced_submission"] is False
    assert llm.calls == 1


def test_early_format_error_is_not_replaced_with_fallback(sample_db, config, task):
    llm = ScriptedLLM(
        [
            response("execute_sql", {"sql": "SELECT count(*) FROM employees"}),
            "invalid response",
        ]
    )
    record = evaluate_batch([task], llm, Tokenizer(), "prompt", replace(config, max_turns=3), 512)[
        0
    ]
    assert record["status"] == "invalid_response"
    assert record["forced_submission"] is False
    assert record["final_sql"] is None


def test_remaining_rounds_and_final_submission_reminder(sample_db, config, task):
    sql = "SELECT count(*) FROM employees"
    llm = ScriptedLLM(
        [
            response("execute_sql", {"sql": sql}),
            response("list_tables", {}),
            response("submit_sql", {"sql": sql}),
        ]
    )
    tokenizer = Tokenizer()
    record = evaluate_batch([task], llm, tokenizer, "prompt", replace(config, max_turns=2), 512)[0]
    for index, messages in enumerate(tokenizer.messages):
        content = messages[-1]["content"]
        assert f"当前剩余轮数：{3 - index}" in content
        assert f"剩余探索轮数：{2 - index}" in content
        assert ("请在此轮使用submit_sql工具提交SQL" in content) == (index == 2)
    assert tokenizer.messages[0][-1]["content"].startswith(task.question)
    assert tokenizer.messages[1][-1]["content"].startswith("<observation>")
    assert llm.calls == 3
    assert record["status"] == "submitted_sql"
    assert record["correct"] is True
    assert record["forced_submission"] is False
