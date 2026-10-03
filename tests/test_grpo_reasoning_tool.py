from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from grpo.config import GrpoConfig
from grpo.loss import grpo_token_loss
from grpo.rollout import RolloutEpisode, RolloutTurn, collate_training_turns
from grpo.train import _episode_weights, _policy_clip_counts, collect_groups
from grpo.tagged_rollout import _OneToolStop, rollout_tagged_task
from sql_agent.config import EnvConfig
from sql_agent.models import TaskRecord


def _config(tmp_path: Path) -> GrpoConfig:
    return GrpoConfig(
        student_model=tmp_path, reference_model=tmp_path,
        train_data=tmp_path, output_dir=tmp_path,
        rollouts_per_prompt=6, medium_tasks=1,
        interaction_protocol="reasoning_tool",
    )


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="example", source="test", db_id="db", db_path="db.sqlite",
        question="How many?", reference_sql="SELECT 1",
        difficulty="medium", split="train",
    )


@pytest.mark.parametrize(
    ("successes", "expected_kind"),
    [
        (set(), "all_zero"),
        ({0, 2}, "mixed"),
        (set(range(6)), "all_one"),
    ],
)
def test_binary_grpo_keeps_six_on_policy_rollouts(
    monkeypatch, tmp_path, successes, expected_kind
):
    calls = []

    def fake_rollout(model, tokenizer, task, env, config, *, rollout_index):
        calls.append(rollout_index)
        episode = RolloutEpisode(task.task_id, task.db_id, task.difficulty)
        episode.success = rollout_index in successes
        episode.submitted = True
        return episode

    monkeypatch.setattr("grpo.train.rollout_tagged_task", fake_rollout)
    episodes, groups, details, sizes = collect_groups(
        None, None, [_task()], EnvConfig(), _config(tmp_path)
    )
    assert sizes == [6]
    assert groups[0].kind == expected_kind
    assert len(episodes) == len(details) == 6
    assert calls == list(range(6))
    assert all(episode.origin == "on_policy" for episode in episodes)
    assert [detail["raw_execution_reward"] for detail in details] == [
        float(index in successes) for index in range(6)
    ]
    if not successes or len(successes) == 6:
        assert groups[0].advantages == (0.0,) * 6
    else:
        assert all(
            (advantage > 0) == (index in successes)
            for index, advantage in enumerate(groups[0].advantages)
        )


def test_policy_clip_counts_only_advantage_constrained_side():
    current = torch.tensor([[0.3, -0.3], [0.3, -0.3]], requires_grad=True)
    output = grpo_token_loss(
        current, torch.zeros_like(current), torch.zeros_like(current),
        torch.tensor([1.0, -1.0]), torch.ones_like(current, dtype=torch.bool),
        clip_ratio=0.2, kl_beta=0.0,
    )
    assert output.clipped.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert output.policy_clipped.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    counts = _policy_clip_counts(
        output, torch.ones_like(current, dtype=torch.bool),
        torch.tensor([1.0, -1.0]), ["on_policy", "on_policy"]
    )
    assert counts["on_policy"] == {"clipped": 2, "eligible": 4, "outside": 4, "tokens": 4}
    output.policy_loss.sum().backward()
    assert current.grad[0, 0] == current.grad[1, 1] == 0
    assert current.grad[0, 1] != 0 and current.grad[1, 0] != 0


def test_clip_higher_has_asymmetric_gradient_boundary_and_no_kl():
    current = torch.tensor([
        [math.log(1.25), math.log(1.30)],
        [math.log(0.75), math.log(0.85)],
    ], requires_grad=True)
    output = grpo_token_loss(
        current, torch.zeros_like(current), None,
        torch.tensor([1.0, -1.0]), torch.ones_like(current, dtype=torch.bool),
        clip_ratio=0.2, clip_ratio_high=0.28, kl_beta=0.0,
    )
    assert output.policy_clipped.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert torch.count_nonzero(output.kl) == 0
    assert torch.equal(output.token_loss, output.policy_loss)
    output.token_loss.sum().backward()
    assert current.grad[0, 0] < 0 and current.grad[1, 1] > 0
    assert current.grad[0, 1] == current.grad[1, 0] == 0


def test_zero_kl_collates_turns_without_reference_scores():
    turn = RolloutTurn(
        task_id="example", turn=1,
        input_ids=torch.tensor([1, 2, 3]), prompt_length=2,
        action_mask=torch.tensor([False, True]),
        old_log_probs=torch.tensor([0.0, -0.2]),
        response_text="answer", observation={},
    )
    _, _, mask, old, reference = collate_training_turns(
        [turn], pad_token_id=0, device="cpu", require_teacher=False
    )
    assert mask.tolist() == [[False, True]]
    assert old[0, 1].item() == pytest.approx(-0.2)
    assert reference is None


def test_each_six_rollout_group_has_equal_total_weight():
    weights = _episode_weights([6, 6], group_equal=True)
    assert sum(weights[:6]) == pytest.approx(0.5)
    assert sum(weights[6:]) == pytest.approx(0.5)


def test_generation_stops_after_first_tool_block():
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(map(chr, ids)))
    stopping = _OneToolStop(tokenizer)
    assert not stopping(torch.tensor([[ord(char) for char in "<tool>x"]]), None)
    assert stopping(torch.tensor([[ord(char) for char in "<tool>x</tool>"]]), None)


def test_gold_injection_is_rejected_even_for_old_configs(tmp_path):
    with pytest.raises(ValueError, match="no longer supported"):
        GrpoConfig(
            student_model=tmp_path, reference_model=tmp_path,
            train_data=tmp_path, output_dir=tmp_path,
            medium_tasks=1, gold_injection=True,
        ).validate()


def test_rollout_generation_and_scoring_never_receive_gold_hint(monkeypatch, tmp_path):
    class Tokenizer:
        all_special_ids = [0]
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            transcript = "\n".join(item["content"] for item in messages)
            return torch.tensor([[ord(char) + 1 for char in transcript]], dtype=torch.long)

        def decode(self, ids, **kwargs):
            return "".join(chr(int(token) - 1) for token in ids if int(token) > 0)

    class Model:
        def __init__(self):
            self.generation_prompts = []
            self.sampling_options = []
            self.scoring_inputs = []

        def eval(self):
            return self

        def generate(self, *, input_ids, **kwargs):
            self.generation_prompts.append(tokenizer.decode(input_ids[0]))
            self.sampling_options.append(kwargs)
            actions = [
                ('list_tables', '{}'),
                ('inspect_tables', '{"table_names":["numbers"]}'),
                ('execute_sql', '{"sql":"SELECT 123"}'),
                ('submit_sql', '{"sql":"SELECT 123"}'),
            ]
            name, arguments = actions[len(self.generation_prompts) - 1]
            response = (
                '<reasoning>I will check the database evidence.</reasoning>'
                f'<tool>{{"name":"{name}","arguments":{arguments}}}</tool>'
            )
            tokens = torch.tensor([[ord(char) + 1 for char in response]], dtype=torch.long)
            return torch.cat((input_ids, tokens), dim=1)

        def __call__(self, *, input_ids, **kwargs):
            self.scoring_inputs.append(input_ids.clone())
            return SimpleNamespace(logits=torch.zeros(
                input_ids.shape[0], input_ids.shape[1], 258
            ))

    class Environment:
        def __init__(self, *args, **kwargs):
            self.done = False
            self.turn = 0

        def reset(self, task):
            return {}

        def submit_sql(self, arguments):
            assert arguments == {"sql": "SELECT 123"}
            self.done = True
            return {"verification": {"correct": True}}, True

        def step(self, action):
            self.turn += 1
            return {"status": "success"}, False

        def close(self):
            pass

    monkeypatch.setattr("grpo.tagged_rollout.SQLAgentEnv", Environment)
    tokenizer = Tokenizer()
    model = Model()
    task = TaskRecord(
        task_id="ordinary", source="test", db_id="db", db_path="db.sqlite",
        question="How many?", reference_sql="SELECT 987654",
        difficulty="medium", split="train",
    )
    config = replace(
        _config(tmp_path),
        top_k=None, temperature=1.0, top_p=0.99,
        clip_ratio_high=0.28, kl_beta=0.0, learning_rate=1e-6,
    )
    episode = rollout_tagged_task(
        model, tokenizer, task, EnvConfig(max_turns=3), config,
        device="cpu",
    )
    assert episode.success and episode.origin == "on_policy"
    assert all("SELECT 987654" not in prompt for prompt in model.generation_prompts)
    assert all("Training-only target SQL" not in prompt for prompt in model.generation_prompts)
    assert all(
        torch.equal(turn.input_ids, ids[0])
        for turn, ids in zip(episode.turns, model.scoring_inputs, strict=True)
    )
    assert model.sampling_options[0]["top_k"] == 0
    assert model.sampling_options[0]["temperature"] == 1.0
    assert model.sampling_options[0]["top_p"] == 0.99
    clean_prompt = tokenizer.decode(episode.turns[0].input_ids[:episode.turns[0].prompt_length])
    assert "SELECT 123" not in clean_prompt
    assert len(episode.turns) == 4
    assert episode.turns[0].action_mask.sum() > 0
