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
    BM25Retriever,
    EmbeddingEncoder,
    EmbeddingRetriever,
    SqlMemoryRetriever,
    render_context,
    retrieve,
)
from experience_memory.store import FIELDS, MemoryStore, first_executed_sql
from experience_memory.backfill_first_sql import backfill_first_sql
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
    assert record["question"] == task.question
    assert record["experience"] == EXPERIENCE
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
        "question": "How many distinct entities?",
        "experience": EXPERIENCE,
        "sql_before": "OLD SQL",
        "sql_after": "NEW SQL",
        "source_task_id": "do_not_inject",
    }
    unrelated = {**memory, "question": "Which entities come first?",
                 "experience": "排序结果时检查升序和降序"}
    class Encoder:
        batch_size = 8

        def encode(self, texts, *, is_query):
            if not is_query:
                assert texts == [unrelated["question"], memory["question"]]
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
        ("stop", '{"experience":"ok","type":"json_object","sql":"SELECT 1"}'),
        ("stop", '{"experience":"ok","type":"other"}'),
        ("stop", '{"experience":""}'),
    ],
)
def test_bad_reflections_are_rejected(task, finish_reason, content):
    with pytest.raises(ValueError):
        reflect(Teacher(finish_reason, content), task=task, schema="schema", initial={})


def test_flash_response_format_marker_is_ignored(task):
    teacher = Teacher(content=json.dumps({"experience": EXPERIENCE, "type": "json_object"}))
    result = reflect(teacher, task=task, schema="schema", initial={})
    assert result["experience"] == EXPERIENCE
    assert json.loads(result["teacher_response"]["content"])["type"] == "json_object"


def test_reflection_reasks_for_strict_json_without_accepting_extra_fields(task):
    class CorrectingTeacher:
        def __init__(self):
            self.calls = []

        def complete_reflection(self, messages, **kwargs):
            self.calls.append(json.loads(json.dumps(messages)))
            payload = ({"experience": "invalid candidate", "analysis_of_existing_experience": "extra"}
                       if len(self.calls) == 1 else {"experience": EXPERIENCE})
            return Completion({"content": json.dumps(payload)}, "stop", None, None, {})

    teacher = CorrectingTeacher()
    result = reflect(teacher, task=task, schema="schema", initial={})
    assert len(teacher.calls) == 2
    assert teacher.calls[-1][-1]["role"] == "user"
    assert "只有 experience 一个字段" in teacher.calls[-1][-1]["content"]
    assert result["experience"] == EXPERIENCE
    assert len(result["teacher_attempts"]) == 2
    assert json.loads(result["teacher_response"]["content"]) == {"experience": EXPERIENCE}


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
    memories = [{"question": "Count distinct entities", "experience": "按实体去重后计数"},
                {"question": "Aggregate by entity", "experience": "检查聚合粒度"}]

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
        retrieve("question", [{"question": "source", "experience": "advice"}], encoder=Encoder())


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


def test_default_embedding_retrieval_takes_five():
    memories = [{"question": str(i), "experience": str(i)} for i in range(12)]

    class Encoder:
        batch_size = 8

        def encode(self, texts, *, is_query):
            return [[1, 0] for _ in texts]

    assert retrieve("question", memories, encoder=Encoder()) == memories[:5]


def test_tfidf_retrieval_uses_full_experience_only():
    memories = [
        {"experience": "原始问题：COUNT employees\n通用经验：检查去重", "sql_after": "SUM salary"},
        {"experience": "原始问题：SUM salary\n通用经验：检查聚合", "sql_after": "COUNT employees"},
    ]
    assert retrieve("count EMPLOYEES", memories, retriever="tfidf") == [memories[0]]
    assert retrieve("salary", memories, retriever="tfidf") == [memories[1]]
    assert retrieve("去重", memories, retriever="tfidf") == [memories[0]]


def test_tfidf_no_overlap_limits_and_stable_ties():
    memories = [{"experience": "same query"} for _ in range(7)]
    assert retrieve("same", memories, retriever="tfidf") == memories[:5]
    assert retrieve("same", memories, top_k=2, retriever="tfidf") == memories[:2]
    assert retrieve("unknown", memories, retriever="tfidf") == []
    assert retrieve("same", memories, top_k=0, retriever="tfidf") == []
    assert retrieve("same", [], retriever="tfidf") == []
    with pytest.raises(ValueError):
        retrieve("same", memories, top_k=-1, retriever="tfidf")


def test_tfidf_never_loads_embedding(monkeypatch):
    monkeypatch.setattr(
        "experience_memory.retrieve.EmbeddingEncoder",
        lambda *args, **kwargs: pytest.fail("TF-IDF must not load an embedding model"),
    )
    memory = {"experience": "rare query"}
    assert retrieve("rare", [memory], retriever="tfidf") == [memory]


def test_tfidf_smoothed_idf_and_l2_cosine():
    import math

    from experience_memory.retrieve import TfidfRetriever

    retriever = TfidfRetriever([{"experience": "common rare rare"}, {"experience": "common"}])
    rare_idf = 1 + math.log(3 / 2)
    assert retriever.idf == {"common": 1, "rare": rare_idf}
    expected_norm = math.sqrt(1 + (2 * rare_idf) ** 2)
    assert retriever.vectors[0]["rare"] == pytest.approx(2 * rare_idf / expected_norm)
    assert retriever._vector({"rare": 1, "unseen": 100}) == {"rare": 1}


def test_separated_question_keeps_student_context_identical():
    memory = {"question": "Full original question?", "experience": "General advice",
              "sql_before": "SELECT 1", "sql_after": "SELECT 2"}
    old = {key: value for key, value in memory.items() if key != "question"}
    old["experience"] = "原始问题：Full original question?\n通用经验：General advice"
    assert render_context([memory], max_tokens=1000, count_tokens=len) == render_context(
        [old], max_tokens=1000, count_tokens=len
    )


def test_question_retrieval_rejects_missing_question():
    with pytest.raises(ValueError, match="question"):
        retrieve("question", [{"experience": "advice"}])


@pytest.mark.parametrize("top_k", [5, 10])
def test_sql_tfidf_top_k_scores_and_stable_order(top_k, monkeypatch):
    monkeypatch.setattr("experience_memory.retrieve.EmbeddingEncoder",
                        lambda *args, **kwargs: pytest.fail("SQL TF-IDF must not load embeddings"))
    memories = [{"memory_id": str(i), "question": "unindexed_question",
                 "experience": "unindexed_advice", "sql_before": "SELECT SUM(salary) FROM employees",
                 "sql_after": "SELECT COUNT(*) FROM departments"} for i in range(12)]
    retriever = SqlMemoryRetriever(memories, retriever="tfidf", top_k=top_k,
                                  max_tokens=10000, count_tokens=len)
    retriever.prepare()
    assert "unindexed_advice" not in retriever.retriever.idf
    assert "departments" not in retriever.retriever.idf
    result, empty = retriever(["SELECT SUM(salary) FROM employees", "unindexed_advice"])
    assert result["memory_ids"] == [str(i) for i in range(top_k)]
    assert result["scores"] == pytest.approx([1.0] * top_k)
    assert result["query_sql"] == "SELECT SUM(salary) FROM employees"
    assert result["context"].count("修改前 SQL：") == top_k
    assert empty["memory_ids"] == [] and empty["scores"] == [] and empty["context"] == ""


def test_sql_tfidf_uses_before_sql_instead_of_question_or_experience():
    memories = [
        {"memory_id": "a", "question": "count departments", "experience": "count departments",
         "sql_before": "SELECT SUM(salary) FROM employees", "sql_after": "SELECT COUNT(*) FROM departments"},
        {"memory_id": "b", "question": "sum salary employees", "experience": "sum salary employees",
         "sql_before": "SELECT COUNT(*) FROM departments", "sql_after": "SELECT SUM(salary) FROM employees"},
    ]
    retriever = SqlMemoryRetriever(memories, retriever="tfidf", top_k=1,
                                  max_tokens=10000, count_tokens=len)
    result, = retriever(["SELECT SUM(salary) FROM employees"])
    assert result["memory_ids"] == ["a"]
    assert result["scores"] == pytest.approx([1.0])


def test_sql_retrieval_indexes_before_sql_and_preserves_scores():
    memories = [{"memory_id": "a", "question": "source", "experience": "advice",
                 "sql_before": "SELECT wrong", "sql_after": "SELECT right"}]
    calls = []

    class Encoder:
        batch_size = 8

        def encode(self, texts, *, is_query):
            calls.append((texts, is_query))
            return [[1, 0] for _ in texts]

    retriever = SqlMemoryRetriever(memories, Encoder(), top_k=5, max_tokens=1000,
                                  count_tokens=len)
    retriever.prepare()
    (result,) = retriever(["SELECT candidate"])
    assert calls == [(["SELECT wrong"], False), (["SELECT candidate"], True)]
    assert result["query_sql"] == "SELECT candidate"
    assert result["memory_ids"] == ["a"] and result["scores"] == [1.0]
    assert "advice" in result["context"]


@pytest.mark.parametrize("failed_first", [False, True])
@pytest.mark.parametrize("requires_success", [False, True])
@pytest.mark.parametrize("retrieval_backend", ["stub", "tfidf", "bm25"])
def test_memory_is_injected_after_first_successful_execute_only(
    task, config, sample_db, monkeypatch, failed_first, requires_success, retrieval_backend,
):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)
    ))
    wrong = "SELECT count(*) FROM departments"
    actions = ([("execute_sql", "SELECT * FROM nonexistent")] if failed_first else [])
    actions += [("execute_sql", wrong), ("execute_sql", task.reference_sql),
                ("submit_sql", task.reference_sql)]
    inputs = []
    calls = []
    marker = "MEMORY_AFTER_FIRST_TRIGGER"
    sql_retriever = SqlMemoryRetriever(
        [{"memory_id": "a", "experience": marker, "sql_before": wrong,
          "sql_after": "SELECT historical_example"}],
        retriever=retrieval_backend, top_k=5, max_tokens=10000, count_tokens=len,
    ) if retrieval_backend != "stub" else None

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            inputs.append(json.loads(json.dumps(messages)))
            return json.dumps(messages)

    class LLM:
        def generate(self, prompts, params, **kwargs):
            name, sql = actions[len(inputs) - 1]
            text = "<reasoning>Inspect result.</reasoning><tool>" + json.dumps(
                {"name": name, "arguments": {"sql": sql}}
            ) + "</tool>"
            return [SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason="stop")])]

    def callback(sqls):
        calls.append(sqls)
        if sql_retriever is not None:
            return sql_retriever(sqls)
        return [{"query_sql": sql, "context": marker, "memory_ids": ["a"],
                 "scores": [0.5]} for sql in sqls]

    (result,) = evaluate_batch([task], LLM(), Tokenizer(), "system", config, 512,
                               execute_sql_memory=callback,
                               execute_sql_memory_requires_success=requires_success)
    trigger = int(failed_first and requires_success)
    trigger_sql = actions[trigger][1]
    assert calls == [[trigger_sql]]
    assert all(marker not in json.dumps(messages) for messages in inputs[:trigger + 1])
    assert marker in json.dumps(inputs[trigger + 1])
    assert marker not in json.dumps(result["initial_messages"])
    assert result["turns"][trigger]["observation"]["status"] == (
        "error" if failed_first and not requires_success else "success")
    assert result["memory_retrieval"]["query_sql"] == trigger_sql
    assert sum("memory_retrieval" in turn for turn in result["turns"]) == 1
    assert result["correct"] is True


def test_no_execute_sql_means_no_retrieval(task, config, sample_db, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)
    ))

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

    class LLM:
        def generate(self, prompts, params, **kwargs):
            text = "<reasoning>Submit.</reasoning><tool>" + json.dumps(
                {"name": "submit_sql", "arguments": {"sql": task.reference_sql}}
            ) + "</tool>"
            return [SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason="stop")])]

    (record,) = evaluate_batch([task], LLM(), Tokenizer(), "system", config, 512,
                               execute_sql_memory=lambda sqls: pytest.fail("No successful SQL"))
    assert record["correct"] and record["memory_retrieval"] is None


@pytest.mark.parametrize("failed_first", [False, True])
@pytest.mark.parametrize("empty_second_retrieval", [False, True])
def test_retrieve_on_every_execute_from_second_including_errors(
    task, config, sample_db, monkeypatch, failed_first, empty_second_retrieval,
):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)
    ))
    first = "SELECT * FROM nonexistent" if failed_first else "SELECT count(*) FROM departments"
    actions = [
        ("execute_sql", {"sql": first}), ("list_tables", {}),
        ("execute_sql", {"sql": "SELECT * FROM nonexistent"}),
        ("inspect_tables", {"table_names": ["employees"]}),
        ("execute_sql", {"sql": task.reference_sql}),
        ("submit_sql", {"sql": task.reference_sql}),
    ]
    inputs, calls = [], []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            inputs.append(json.loads(json.dumps(messages)))
            return json.dumps(messages)

    class LLM:
        def generate(self, prompts, params, **kwargs):
            name, arguments = actions[len(inputs) - 1]
            text = "<reasoning>Inspect.</reasoning><tool>" + json.dumps(
                {"name": name, "arguments": arguments}
            ) + "</tool>"
            return [SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason="stop")])]

    def callback(sqls):
        calls.append(sqls)
        context = "" if empty_second_retrieval and len(calls) == 1 else f"MEMORY_PASS_{len(calls)}"
        return [{"query_sql": sql, "context": context, "memory_ids": ["a"] if context else [],
                 "scores": [0.5] if context else []} for sql in sqls]

    (record,) = evaluate_batch([task], LLM(), Tokenizer(), "system", config, 512,
                               execute_sql_memory=callback,
                               execute_sql_memory_requires_success=False,
                               execute_sql_memory_start_index=2,
                               execute_sql_memory_repeat=True)
    assert calls == [["SELECT * FROM nonexistent"], [task.reference_sql]]
    assert record["turns"][2]["observation"]["status"] == "error"
    assert [r["execute_index"] for r in record["memory_retrievals"]] == [2, 3]
    assert [r["turn_index"] for r in record["memory_retrievals"]] == [2, 4]
    assert all("MEMORY_PASS" not in json.dumps(m) for m in inputs[:3])
    if not empty_second_retrieval:
        assert "MEMORY_PASS_1" in json.dumps(inputs[3])
    assert "MEMORY_PASS_1" not in json.dumps(inputs[5])
    assert "MEMORY_PASS_2" in json.dumps(inputs[5])
    assert record["correct"] is True


@pytest.mark.parametrize("top_k", [5, 10])
def test_sql_bm25_scores_limits_and_indexed_field(top_k, monkeypatch):
    import math

    monkeypatch.setattr("experience_memory.retrieve.EmbeddingEncoder",
                        lambda *args, **kwargs: pytest.fail("BM25 must not load embeddings"))
    memories = [{"memory_id": str(i), "question": "unindexed_question",
                 "experience": "unindexed_advice", "sql_before": "SELECT salary FROM employees",
                 "sql_after": "SELECT COUNT(*) FROM departments"} for i in range(12)]
    retriever = SqlMemoryRetriever(memories, retriever="bm25", top_k=top_k,
                                  max_tokens=10000, count_tokens=len)
    retriever.prepare()
    result, empty = retriever(["SELECT salary FROM employees", "departments unindexed_advice"])
    assert result["memory_ids"] == [str(i) for i in range(top_k)]
    assert result["scores"] == pytest.approx([4 * math.log(1 + 0.5 / 12.5)] * top_k)
    assert result["context"].count("修改前 SQL：") == top_k
    assert empty["memory_ids"] == [] and empty["context"] == ""


def test_bm25_term_saturation_length_normalization_and_query_terms():
    import math

    memories = [{"sql_before": "rare rare common"}, {"sql_before": "common filler filler filler filler"}]
    retriever = BM25Retriever(memories)
    matches, repeated = retriever.search(["rare", "RARE rare"], return_scores=True)
    expected = math.log(2) * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / 4))
    assert len(matches) == 1 and matches[0]["score"] == pytest.approx(expected)
    assert repeated == matches
    common, = retriever.search(["common"], return_scores=True)
    assert common[0]["memory"] == memories[0]
    assert common[0]["score"] > common[1]["score"]
    assert retriever.search(["unknown"], 0) == [[]]
    assert BM25Retriever([]).search(["rare"]) == [[]]
    assert BM25Retriever([{"sql_before": "!!!"}]).search(["rare"]) == [[]]
    with pytest.raises(ValueError, match="top_k"):
        retriever.search(["rare"], -1)
    with pytest.raises(ValueError, match="sql_before"):
        BM25Retriever([{"experience": "advice"}])


@pytest.mark.parametrize("kwargs", [{"k1": 0}, {"b": -1}, {"b": 2}, {"k1": float("nan")}])
def test_bm25_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match="BM25"):
        BM25Retriever([], **kwargs)


def test_first_executed_sql_includes_errors_and_ignores_unexecuted_calls():
    trajectory = {"final_sql": "SELECT final", "turns": [
        {"tool": "execute_sql", "arguments": {"sql": "SELECT unexecuted"}},
        {"tool": "execute_sql", "arguments": {"sql": "SELECT * FROM missing"},
         "observation": {"status": "error", "error_type": "sql_error"}},
        {"tool": "execute_sql", "arguments": {"sql": "SELECT 1"},
         "observation": {"status": "success"}},
    ]}
    assert first_executed_sql(trajectory) == "SELECT * FROM missing"
    assert first_executed_sql({"final_sql": "SELECT final", "turns": []}) is None


def test_backfill_and_new_admission_preserve_first_sql(tmp_path, sample_db, task, config):
    import sqlite3

    task = replace(task, split="train")
    database = tmp_path / "mem.sqlite"
    store = MemoryStore(database)
    initial_path, retry_path = setup_evidence(tmp_path, task)
    initial = json.loads(initial_path.read_text())
    initial["turns"] = [{"tool": "execute_sql", "arguments": {"sql": "SELECT first"},
                         "observation": {"status": "error"}}]
    write_json(initial_path, initial)
    arguments = dict(task=task, experience=EXPERIENCE, initial_path=initial_path,
                     retry_path=retry_path, verifier=ExecutionVerifier(sample_db, task.reference_sql, config))
    assert store.append_verified(**arguments)
    record, = store.all()
    assert record["sql_first_execute"] == "SELECT first"
    legacy = {k: v for k, v in record.items() if k != "sql_first_execute"}
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE memories SET record = ?", (json.dumps(legacy),))
    assert store.all() == [legacy]
    assert not store.append_verified(**arguments)
    assert store.all() == [legacy]
    output = tmp_path / "memories.jsonl"
    assert backfill_first_sql(database, output) == {
        "total": 1, "updated": 1, "with_first_execute_sql": 1, "without_execute_sql": 0}
    assert store.all() == [record]
    assert json.loads(output.read_text()) == record
    assert backfill_first_sql(database, output)["updated"] == 0
    initial["turns"][0]["arguments"]["sql"] = "SELECT changed"
    write_json(initial_path, initial)
    with pytest.raises(ValueError, match="disagrees"):
        backfill_first_sql(database, output)
    assert store.all() == [record]



@pytest.mark.parametrize("backend", ["tfidf", "bm25"])
def test_first_execute_sql_index_excludes_missing_and_never_uses_final(backend):
    memories = [
        {"memory_id": "a", "sql_first_execute": "SELECT salary FROM employees",
         "sql_before": "SELECT albums FROM records", "sql_after": "SELECT fixed", "experience": "advice"},
        {"memory_id": "b", "sql_first_execute": "SELECT albums FROM records",
         "sql_before": "SELECT salary FROM employees", "sql_after": "SELECT fixed", "experience": "advice"},
        {"memory_id": "c", "sql_first_execute": None, "sql_before": "SELECT unique_missing",
         "sql_after": "SELECT fixed", "experience": "advice"},
        {"memory_id": "d", "sql_before": "SELECT legacy_missing",
         "sql_after": "SELECT fixed", "experience": "advice"},
    ]
    retriever = SqlMemoryRetriever(memories, retriever=backend, text_field="sql_first_execute",
                                   top_k=1, max_tokens=10000, count_tokens=len)
    assert retriever.indexed_memory_count == 2
    assert retriever.excluded_missing_sql_count == 2
    match, absent = retriever(["SELECT salary FROM employees", "unique_missing legacy_missing"])
    assert match["memory_ids"] == ["a"]
    assert absent["memory_ids"] == []
    empty = SqlMemoryRetriever(memories[2:], retriever=backend, text_field="sql_first_execute",
                               top_k=10, max_tokens=10000, count_tokens=len)
    assert empty(["SELECT 1"])[0]["context"] == ""
