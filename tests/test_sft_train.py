from __future__ import annotations

import torch

from sft.train import collate_samples, epoch_order, learning_rate_at_step, masked_ce_sum
from sft.train_config import SftTrainConfig


def _sample(values: list[int], mask: list[bool], task_id: str):
    return {
        "input_ids": torch.tensor(values),
        "action_mask": torch.tensor(mask),
        "task_id": task_id,
        "turn": 1,
    }


def test_collate_preserves_shifted_action_masks() -> None:
    samples = [
        _sample([1, 2, 3], [False, True], "a"),
        _sample([4, 5], [True], "b"),
    ]
    input_ids, attention_mask, action_mask = collate_samples(samples, 0, "cpu")
    assert input_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert action_mask.tolist() == [[False, True], [True, False]]


def test_masked_ce_excludes_prompt_tokens() -> None:
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = -20
    logits[0, 1, 3] = 20
    input_ids = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[False, True]])
    assert masked_ce_sum(logits, input_ids, mask).item() < 1e-5


def test_epoch_order_is_deterministic_and_complete() -> None:
    samples = [_sample([1, 2], [True], str(index)) for index in range(20)]
    first = epoch_order(samples, 42, bucket_size=5)
    second = epoch_order(samples, 42, bucket_size=5)
    assert [x["task_id"] for x in first] == [x["task_id"] for x in second]
    assert {x["task_id"] for x in first} == {str(index) for index in range(20)}


def test_learning_rate_warms_up_then_cosine_decays(tmp_path) -> None:
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    config = SftTrainConfig(model, data, tmp_path / "output", warmup_steps=2)
    assert learning_rate_at_step(config, 1, 10) == config.learning_rate / 2
    assert learning_rate_at_step(config, 2, 10) == config.learning_rate
    assert learning_rate_at_step(config, 10, 10) == 0.0
