from __future__ import annotations

import json

import pytest
import torch

from sft.reasoning_trajectories import prepare


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        text = "".join(f"{message['role']}:{message['content']}|" for message in messages)
        return text + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return type("Tokens", (), {"input_ids": [ord(char) for char in text]})()


def _prepare_record(tmp_path, monkeypatch, *, origin, cutoff=None):
    source = tmp_path / "source"
    trajectory_dir = source / "trajectories"
    trajectory_dir.mkdir(parents=True)
    messages = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": "question"},
    ]
    for turn in range(1, 4):
        messages.append(
            {
                "role": "assistant",
                "content": f"<reasoning>step {turn}</reasoning><tool>action {turn}</tool>",
            }
        )
        if turn < 3:
            messages.append({"role": "user", "content": "<observation>result</observation>"})
    record = {
        "task_id": "spider_train_00001",
        "split": "train",
        "correct": True,
        "merged_origin": origin,
        "messages": messages,
        "trainable_turn_numbers": [1, 2, 3],
    }
    if cutoff is not None:
        record["cutoff_turn"] = cutoff
    (trajectory_dir / "example.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        "sft.reasoning_trajectories.AutoTokenizer.from_pretrained", lambda _: _Tokenizer()
    )
    monkeypatch.setattr("sft.reasoning_trajectories.tokenizer_fingerprint", lambda _: "fake")
    output = tmp_path / "dataset"
    statistics = prepare(source, tmp_path / "model", output, 2048)
    shard = torch.load(output / "shards" / "00000.pt", map_location="cpu", weights_only=True)
    return statistics, shard


def test_full_gold_regeneration_trains_all_assistant_turns(tmp_path, monkeypatch):
    statistics, shard = _prepare_record(tmp_path, monkeypatch, origin="gold_regeneration")
    assert [turn["turn"] for turn in shard["turns"]] == [1, 2, 3]
    assert statistics["turns"] == 3
    assert statistics["action_tokens"] == sum(
        int(turn["action_mask"].sum()) for turn in shard["turns"]
    )
    for turn in shard["turns"]:
        mask = turn["action_mask"]
        first = int(mask.nonzero()[0])
        assert first > 0
        assert not mask[:first].any()
        assert mask[first:].all()


def test_original_trajectory_still_trains_every_selected_turn(tmp_path, monkeypatch):
    statistics, shard = _prepare_record(tmp_path, monkeypatch, origin="original")
    assert [turn["turn"] for turn in shard["turns"]] == [1, 2, 3]
    assert statistics["turns"] == 3


def test_old_forked_gold_suffixes_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Forked Gold suffixes"):
        _prepare_record(tmp_path, monkeypatch, origin="gold_repair")
