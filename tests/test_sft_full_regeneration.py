from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import replace

import pytest

from sft.collect_reasoning import collect_trajectory
from sft.curate_reasoning import curate, validate_record
from sft.regenerate_reasoning import regenerate
from sql_agent.deepseek import Completion
from sql_agent.protocol import build_prompt


def response(tool, arguments):
    return (
        "<reasoning>Use the question and observed evidence.</reasoning><tool>"
        + json.dumps(
            {
                "name": tool,
                "arguments": arguments,
            }
        )
        + "</tool>"
    )


class Client:
    model = "fake-teacher"

    def __init__(self, responses):
        self.responses = deque(responses)
        self.requests = []

    def complete_thinking_text(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        return Completion(
            {"content": self.responses.popleft(), "reasoning_content": "PRIVATE_TEACHER_REASONING"},
            "stop",
            None,
            None,
            {},
        )

    def complete_text(self, messages, **kwargs):
        return self.complete_thinking_text(messages, **kwargs)


def source(task):
    return {
        "task_id": task.task_id,
        "question": task.question,
        "db_id": task.db_id,
        "correct": False,
        "messages": [
            {"role": "system", "content": build_prompt(10)},
            {"role": "user", "content": task.question},
            {"role": "assistant", "content": "OLD_FAILED_PREFIX"},
        ],
    }


def test_full_regeneration_private_context_and_public_audit(sample_db, config, task):
    task = replace(task, split="train")
    client = Client(
        [
            response("list_tables", {}),
            response("execute_sql", {"sql": task.reference_sql}),
            response("submit_sql", {"sql": task.reference_sql}),
            "accept",
        ]
    )
    record = regenerate(task, source(task), config, client)
    assert record["sft_eligible"] and record["leak_audit"] == "accept"
    assert "OLD_FAILED_PREFIX" not in json.dumps(client.requests)
    assert "<teacher_only_sql>" in client.requests[0][1]["content"]
    assert any("reasoning_content" in message for message in client.requests[1])
    public = json.dumps(record["student_messages"])
    assert "teacher_only_sql" not in public and "PRIVATE_TEACHER_REASONING" not in public
    audit_input = json.loads(client.requests[-1][1]["content"])
    assert set(audit_input) == {"question", "messages"}
    assert audit_input["messages"] == record["student_messages"]
    normalized = validate_record(record, task, config, regenerated=True)
    assert normalized["merged_origin"] == "gold_regeneration"
    assert normalized["trainable_turn_numbers"] == [1, 2, 3]


@pytest.mark.parametrize("case", ["audit_reject", "extra_exploration", "wrong_sql", "no_test"])
def test_regeneration_rejects_unqualified_attempts(case, sample_db, config, task):
    task = replace(task, split="train")
    turns = [
        response("list_tables", {}),
        response("execute_sql", {"sql": task.reference_sql}),
        response("submit_sql", {"sql": task.reference_sql}),
        "accept",
    ]
    if case == "audit_reject":
        turns[-1] = "reject"
    elif case == "extra_exploration":
        turns[2] = response("list_tables", {})
    elif case == "wrong_sql":
        turns[2] = response("submit_sql", {"sql": task.reference_sql + ";"})
    elif case == "no_test":
        turns[1] = turns[2]
    client = Client(turns)
    if case == "audit_reject":
        assert not regenerate(task, source(task), config, client)["sft_eligible"]
    else:
        with pytest.raises(ValueError):
            regenerate(task, source(task), config, client)


def test_curator_replays_observations_and_deduplicates(tmp_path, sample_db, config, task):
    task = replace(task, split="train")
    client = Client(
        [
            response("list_tables", {}),
            response("execute_sql", {"sql": task.reference_sql}),
            response("submit_sql", {"sql": task.reference_sql}),
        ]
    )
    record = collect_trajectory(task, config, client)
    assert validate_record(record, task, config, regenerated=False)["correct"]
    altered = copy.deepcopy(record)
    altered["turns"][1]["observation"]["rows"] = [[999]]
    with pytest.raises(ValueError, match="Stored observation"):
        validate_record(altered, task, config, regenerated=False)
    original = tmp_path / "original" / "trajectories"
    original.mkdir(parents=True)
    for name in ["first", "duplicate"]:
        (original / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    counts = curate(
        original=original.parent,
        regenerated=None,
        tasks=[task],
        config=config,
        output=tmp_path / "curated",
    )
    assert counts["original"] == 1 and counts["skipped"] == 1


def test_curator_rejects_holdout_and_teacher_material(sample_db, config, task):
    with pytest.raises(ValueError, match="training"):
        validate_record({"correct": True}, task, config, regenerated=False)
    task = replace(task, split="train")
    record = collect_trajectory(
        task,
        config,
        Client(
            [
                response("list_tables", {}),
                response("execute_sql", {"sql": task.reference_sql}),
                response("submit_sql", {"sql": task.reference_sql}),
            ]
        ),
    )
    record["messages"][0]["content"] += "<teacher_only_sql>private</teacher_only_sql>"
    with pytest.raises(ValueError, match="Teacher supplement"):
        validate_record(record, task, config, regenerated=False)
