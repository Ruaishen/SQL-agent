import copy
import json
import shutil

import pytest

from dpo.audit import audit_input
from dpo.dataset import load_audited_pairs
from dpo.prepare import prepare_audited
from dpo.train import load_dataset
from tests.test_dpo_audit import action, pair
from tests.test_dpo_prepare import _CharacterTokenizer


@pytest.fixture
def accepted_dataset(tmp_path):
    value = pair()
    value.update(task_id="spider_train_00001", split="train", loss_from_turn=2,
                 rejected_verification={"correct": False}, sanitized_reasoning_turns=[])
    value["rejected_messages"] = copy.deepcopy(value["chosen_messages"])
    value["rejected_messages"][-1]["content"] = action("submit_sql", "SELECT 2")
    root = tmp_path / "audit"
    (root / "accepted/pairs").mkdir(parents=True)
    (root / "reviews").mkdir()
    (root / "report.json").write_text(json.dumps({"status": "completed", "accepted_files": 1}))
    (root / "accepted/pairs/one.json").write_text(json.dumps(value))
    review = {"task_id": value["task_id"], "decision": "accept", "issue_codes": [],
              "review": {"verdict": "accept", "issues": [], "summary": "ok"},
              "mechanical_checks": audit_input(value)["mechanical_checks"]}
    (root / "reviews/one.json").write_text(json.dumps(review))
    source = tmp_path / "pairs.jsonl"
    source.write_text(json.dumps(value) + "\n")
    return source, root, value


@pytest.mark.parametrize("mutation", ["changed", "duplicate", "missing", "incomplete", "reject"])
def test_strict_input_rejects_invalid_evidence(accepted_dataset, mutation):
    source, root, value = accepted_dataset
    if mutation == "changed":
        value["chosen_messages"][-1]["content"] = action("submit_sql", "SELECT 99")
        source.write_text(json.dumps(value))
    elif mutation == "duplicate":
        source.write_text(source.read_text() * 2)
    elif mutation == "missing":
        source.write_text("")
    elif mutation == "incomplete":
        (root / "report.json").write_text(json.dumps({"status": "running_or_incomplete"}))
    else:
        review_path = root / "reviews/one.json"
        review = json.loads(review_path.read_text())
        review["decision"] = "reject"
        review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        load_audited_pairs(source, [root])


def test_audited_preparation_masks_and_portable_shards(accepted_dataset, tmp_path, monkeypatch):
    source, root, value = accepted_dataset
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}")
    monkeypatch.setattr("dpo.prepare.AutoTokenizer.from_pretrained", lambda _: _CharacterTokenizer())
    output = tmp_path / "prepared"
    manifest = prepare_audited(source, [root], model, output, 2000)
    assert manifest["pair_count"] == 1
    assert manifest["shards"] == ["shards/00000.pt"]
    import torch
    data = torch.load(output / manifest["shards"][0], weights_only=True)
    for branch_name in ("chosen", "rejected"):
        branch = data[branch_name]
        scored = "".join(chr(int(branch["input_ids"][i + 1]))
                         for i, flag in enumerate(branch["loss_mask"]) if flag)
        assert "<observation>" not in scored
        assert "execute_sql" not in scored
        assert "submit_sql" in scored
    moved = tmp_path / "moved"
    shutil.copytree(output, moved)
    config = {"dataset_dir": str(moved), "sft_checkpoint": str(model)}
    _, paths = load_dataset(config)
    assert paths[0] == moved / "shards/00000.pt"
    with paths[0].open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        load_dataset(config)
