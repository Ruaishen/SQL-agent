from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from evaluation.run_reasoning_sft import evaluate_batch
from experience_memory.evolve import run_round, write_json
from experience_memory.reflect import reflect
from experience_memory.retrieve import (
    EmbeddingEncoder,
    EmbeddingRetriever,
    render_context,
    retrieve,
)
from experience_memory.store import FIELDS, MemoryStore
from sql_agent.deepseek import Completion, DeepSeekClient
from sql_agent.verifier import ExecutionVerifier

EXPERIENCE = "统计实体数量时，先确认需要统计的实体，检查选取的表与题目是否一致。"


def evidence(task, sql, correct):
    return {
        "task_id": task.task_id,
        "final_sql": sql,
        "correct": correct,
        "status": "submitted_sql",
        "forced_submission": False,
    }


def setup_evidence(tmp_path, task):
    initial_path, retry_path = tmp_path / "initial.json", tmp_path / "retry.json"
    write_json(initial_path, evidence(task, "SELECT count(*) FROM departments", False))
    write_json(retry_path, evidence(task, task.reference_sql, True))
    return initial_path, retry_path


def test_append_reexecutes_and_stores_only_requested_fields(tmp_path, sample_db, task, config):
    task = replace(task, split="train")
    store = MemoryStore(tmp_path / "mem.sqlite")
    initial_path, retry_path = setup_evidence(tmp_path, task)
    verifier = ExecutionVerifier(sample_db, task.reference_sql, config)
    arguments = dict(
        task=task,
        experience=EXPERIENCE,
        initial_path=initial_path,
        retry_path=retry_path,
        verifier=verifier,
    )
    assert store.append_verified(**arguments)
    assert not store.append_verified(**arguments)
    (record,) = store.all()
    assert set(record) == FIELDS
    assert record["sql_before"] == "SELECT count(*) FROM departments"
    assert record["sql_after"] == task.reference_sql
    # Equal experiences from another evidence pair remain independent records.
    second_path = tmp_path / "second_retry.json"
    write_json(second_path, evidence(task, task.reference_sql, True))
    assert store.append_verified(**{**arguments, "retry_path": second_path})
    assert len(store.all()) == 2


@pytest.mark.parametrize(
    "case",
    [
        "initial_true",
        "retry_false",
        "string_false",
        "wrong_sql",
        "initial_correct_sql",
        "wrong_task",
        "holdout",
        "forced",
    ],
)
def test_admission_rejects_invalid_evidence(case, tmp_path, sample_db, task, config):
    task = replace(task, split="train")
    store = MemoryStore(tmp_path / "mem.sqlite")
    initial_path, retry_path = setup_evidence(tmp_path, task)
    initial = json.loads(initial_path.read_text())
    retry = json.loads(retry_path.read_text())
    if case == "initial_true":
        initial["correct"] = True
    elif case == "retry_false":
        retry["correct"] = False
    elif case == "string_false":
        initial["correct"] = "false"
    elif case == "wrong_sql":
        retry["final_sql"] = "SELECT count(*) FROM departments"
    elif case == "initial_correct_sql":
        initial["final_sql"] = task.reference_sql
    elif case == "wrong_task":
        retry["task_id"] = "another"
    elif case == "holdout":
        task = replace(task, split="internal_holdout")
    elif case == "forced":
        retry["forced_submission"] = True
    write_json(initial_path, initial)
    write_json(retry_path, retry)
    with pytest.raises(ValueError):
        store.append_verified(
            task=task,
            experience=EXPERIENCE,
            initial_path=initial_path,
            retry_path=retry_path,
            verifier=ExecutionVerifier(sample_db, task.reference_sql, config),
        )
    assert store.all() == []


def test_retrieval_text_only_and_budget():
    memory = {
        "experience": EXPERIENCE,
        "sql_before": "OLD SQL",
        "sql_after": "NEW SQL",
        "source_task_id": "do_not_inject",
    }
    unrelated = {**memory, "experience": "排序结果时检查升序和降序"}
    class Encoder:
        batch_size = 8

        def encode(self, texts, *, is_query):
            if not is_query:
                assert texts == [unrelated["experience"], EXPERIENCE]
                return [[0, 1], [1, 0]]
            assert texts == ["How many entities?"]
            return [[1, 0]]

    assert retrieve("How many entities?", [unrelated, memory], encoder=Encoder()) == [memory]
    context = render_context([memory], max_tokens=2000, count_tokens=len)
    assert EXPERIENCE in context and "OLD SQL" in context and "NEW SQL" in context
    assert "do_not_inject" not in context
    assert render_context([memory], max_tokens=1, count_tokens=len) == ""
    assert retrieve("How many?", [memory], top_k=0) == []


class Teacher:
    def __init__(self, finish_reason="stop", content=None):
        self.messages = []
        self.finish_reason = finish_reason
        self.content = content or json.dumps({"experience": EXPERIENCE}, ensure_ascii=False)

    def complete_reflection(self, messages, **kwargs):
        self.messages.append(messages)
        return Completion({"content": self.content}, self.finish_reason, None, None, {})


@pytest.mark.parametrize(
    "finish_reason,content",
    [
        ("length", None),
        ("stop", "not json"),
        ("stop", '{"experience":"ok","sql":"SELECT 1"}'),
        ("stop", '{"experience":""}'),
    ],
)
def test_bad_reflections_are_rejected(task, finish_reason, content):
    with pytest.raises(ValueError):
        reflect(Teacher(finish_reason, content), task=task, schema="schema", initial={})


def test_reflection_api_uses_json_without_tool_stop_markers(monkeypatch):
    client = DeepSeekClient("runtime-only-key", model="configured-model")
    payloads = []
    monkeypatch.setattr(client, "_request", lambda payload: payloads.append(payload))
    client.complete_reflection([{"role": "user", "content": "data"}])
    assert "stop" not in payloads[0] and "tools" not in payloads[0]
    assert payloads[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("retry_succeeds", [True, False])
def test_round_full_student_execution_and_resume(
    tmp_path,
    sample_db,
    task,
    config,
    monkeypatch,
    retry_succeeds,
):
    task = replace(task, split="train")
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    # Fixture deliberately includes invalid UTF-8 cells; use schema DDL for this test.
    monkeypatch.setattr(
        "experience_memory.evolve.render_schema", lambda path: "CREATE TABLE employees(id INT)"
    )
    store = MemoryStore(tmp_path / "mem.sqlite")
    teacher = Teacher()
    student_inputs = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            student_inputs.append(messages)
            return json.dumps(messages)

    class LLM:
        calls = 0

        def generate(self, prompts, params, **kwargs):
            self.calls += 1
            assert store.all() == []  # No publication while the round is running.
            sql = (
                task.reference_sql
                if self.calls == 2 and retry_succeeds
                else "SELECT count(*) FROM departments"
            )
            text = (
                "<reasoning>Count the requested entities.</reasoning><tool>"
                + json.dumps({"name": "submit_sql", "arguments": {"sql": sql}})
                + "</tool>"
            )
            return [SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason="stop")])]

    llm, tokenizer = LLM(), Tokenizer()

    def student(batch, contexts):
        return evaluate_batch(
            batch, llm, tokenizer, "system", config, 512, memory_contexts=contexts
        )

    arguments = dict(
        tasks=[task],
        student=student,
        teacher=teacher,
        config=config,
        store=store,
        output_dir=tmp_path / "round",
    )
    counts = run_round(**arguments)
    assert counts["errors"] == 0
    assert counts["appended"] == int(retry_succeeds)
    assert len(store.all()) == int(retry_succeeds)
    assert EXPERIENCE not in student_inputs[0][1]["content"]
    assert EXPERIENCE in student_inputs[1][1]["content"]
    assert task.reference_sql not in json.dumps(student_inputs)
    assert task.reference_sql in teacher.messages[0][1]["content"]
    resumed = run_round(**arguments)
    assert resumed["appended"] == 0
    assert resumed["already_present"] == int(retry_succeeds)
    assert llm.calls == 2 and len(teacher.messages) == 1
    assert json.loads((tmp_path / "round" / "memory_snapshot.json").read_text()) == []


def test_round_skips_correct_initial_without_teacher(tmp_path, sample_db, task, config):
    task = replace(task, split="train")
    teacher = Teacher()
    counts = run_round(
        tasks=[task],
        student=lambda tasks, contexts: [evidence(task, task.reference_sql, True)],
        teacher=teacher,
        config=config,
        store=MemoryStore(tmp_path / "mem.sqlite"),
        output_dir=tmp_path / "round",
    )
    assert counts["initial_correct"] == 1 and counts["appended"] == 0
    assert not teacher.messages


def test_embedding_batch_cache_cross_language_and_stable_ties():
    calls = []
    memories = [{"experience": "按实体去重后计数"}, {"experience": "检查聚合粒度"}]

    class Encoder:
        batch_size = 1

        def encode(self, texts, *, is_query):
            calls.append((texts, is_query))
            return [[2, 0] for _ in texts]

    retriever = EmbeddingRetriever(memories, Encoder())
    assert retriever.search(["How many unique entities?", "统计唯一实体"], 1) == [
        [memories[0]], [memories[0]]
    ]
    assert retriever.search(["another question"], 2) == [memories]
    assert sum(not is_query for _, is_query in calls) == 1
    assert retriever.search(["anything"], 0) == [[]]
    with pytest.raises(ValueError):
        retriever.search(["anything"], -1)


@pytest.mark.parametrize("values", [[[float("nan"), 1]], [[0, 0]], [[1]], [1, 2]])
def test_invalid_embeddings_fail_without_lexical_fallback(values):
    class Encoder:
        batch_size = 8

        def encode(self, texts, *, is_query):
            return [[1, 0]] if is_query else values

    with pytest.raises(ValueError):
        retrieve("question", [{"experience": "advice"}], encoder=Encoder())


def test_embedding_encoder_freezes_model_and_uses_last_token(monkeypatch):
    import torch
    from transformers import AutoModel, AutoTokenizer

    captured = []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            captured.append((texts, kwargs))
            return Batch(input_ids=torch.ones((len(texts), 2), dtype=torch.long))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, input_ids):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(last_hidden_state=torch.tensor(
                [[[9., 9.], [3., 4.]]] * len(input_ids)
            ))

    model = Model()
    loads = []
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **kw: loads.append(kw) or model)
    encoder = EmbeddingEncoder(batch_size=1)
    assert torch.allclose(encoder.encode(["经验"], is_query=False), torch.tensor([[.6, .8]]))
    encoder.encode(["How many?"], is_query=True)
    assert len(loads) == 1 and not model.training and not model.weight.requires_grad
    assert captured[0][0] == ["经验"]
    assert captured[1][0][0].startswith("Instruct:")
    assert captured[1][0][0].endswith("Query:How many?")
    assert captured[0][1]["max_length"] == 2048


def test_default_embedding_retrieval_takes_ten():
    memories = [{"experience": str(i)} for i in range(12)]

    class Encoder:
        batch_size = 8

        def encode(self, texts, *, is_query):
            return [[1, 0] for _ in texts]

    assert retrieve("question", memories, encoder=Encoder()) == memories[:10]
