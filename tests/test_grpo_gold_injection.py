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


def _config(tmp_path: Path, *, inject: bool) -> GrpoConfig:
    return GrpoConfig(
        student_model=tmp_path, reference_model=tmp_path,
        train_data=tmp_path, output_dir=tmp_path,
        rollouts_per_prompt=4, medium_tasks=1,
        interaction_protocol="reasoning_tool", gold_injection=inject,
    )


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="example", source="test", db_id="db", db_path="db.sqlite",
        question="How many?", reference_sql="SELECT 1",
        difficulty="medium", split="train",
    )


@pytest.mark.parametrize(
    ("successes", "inject", "expected_size", "expected_guided", "expected_kind"),
    [
        (set(), True, 5, 1, "mixed"),
        ({0, 2}, True, 4, 0, "mixed"),
        ({0, 1, 2, 3}, True, 4, 0, "all_one"),
        (set(), False, 4, 0, "all_zero"),
    ],
)
def test_conditional_gold_injection_only_on_all_wrong(
    monkeypatch, tmp_path, successes, inject, expected_size, expected_guided, expected_kind
):
    calls = []

    def fake_rollout(model, tokenizer, task, env, config, *, rollout_index, gold_guided=False):
        calls.append(gold_guided)
        episode = RolloutEpisode(task.task_id, task.db_id, task.difficulty)
        episode.origin = "gold_guided" if gold_guided else "on_policy"
        episode.success = gold_guided or rollout_index in successes
        episode.submitted = True
        if gold_guided:
            episode.turns = [object()]
        return episode

    monkeypatch.setattr("grpo.train.rollout_tagged_task", fake_rollout)
    episodes, groups, details, sizes = collect_groups(
        None, None, [_task()], EnvConfig(), _config(tmp_path, inject=inject)
    )
    assert sizes == [expected_size]
    assert groups[0].kind == expected_kind
    assert len(episodes) == len(details) == expected_size
    assert sum(calls) == expected_guided
    if expected_guided:
        assert groups[0].advantages[-1] > 0
        assert all(value < 0 for value in groups[0].advantages[:-1])
        assert details[-1]["trajectory_origin"] == "gold_guided"


def test_three_failed_gold_attempts_leave_all_zero_group(monkeypatch, tmp_path):
    calls = []

    def fake_rollout(model, tokenizer, task, env, config, *, rollout_index, gold_guided=False):
        calls.append(gold_guided)
        episode = RolloutEpisode(task.task_id, task.db_id, task.difficulty)
        episode.origin = "gold_guided" if gold_guided else "on_policy"
        episode.success = False
        episode.submitted = gold_guided
        return episode

    monkeypatch.setattr("grpo.train.rollout_tagged_task", fake_rollout)
    config = _config(tmp_path, inject=True)
    assert config.gold_injection_max_attempts == 3
    episodes, groups, details, sizes = collect_groups(
        None, None, [_task()], EnvConfig(), config
    )

    assert calls == [False] * 4 + [True] * 3
    assert sizes == [4]
    assert len(episodes) == len(details) == 4
    assert groups[0].kind == "all_zero"
    assert groups[0].advantages == (0.0,) * 4
    assert all(detail["gold_injection_attempts"] == 3 for detail in details)
    assert all(not detail["gold_injection_succeeded"] for detail in details)


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
        torch.tensor([1.0, -1.0]), ["on_policy", "gold_guided"]
    )
    assert counts["on_policy"] == {"clipped": 1, "eligible": 2, "outside": 2, "tokens": 2}
    assert counts["gold_guided"] == {"clipped": 1, "eligible": 2, "outside": 2, "tokens": 2}
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


def test_injected_group_keeps_same_total_weight_as_standard_group():
    weights = _episode_weights([4, 5], group_equal=True)
    assert sum(weights[:4]) == pytest.approx(0.5)
    assert sum(weights[4:]) == pytest.approx(0.5)


def test_generation_stops_after_first_tool_block():
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(map(chr, ids)))
    stopping = _OneToolStop(tokenizer)
    assert not stopping(torch.tensor([[ord(char) for char in "<tool>x"]]), None)
    assert stopping(torch.tensor([[ord(char) for char in "<tool>x</tool>"]]), None)


def test_gold_injection_requires_tagged_binary_fixed_group(tmp_path):
    with pytest.raises(ValueError, match="requires reasoning_tool"):
        GrpoConfig(
            student_model=tmp_path, reference_model=tmp_path,
            train_data=tmp_path, output_dir=tmp_path,
            medium_tasks=1, gold_injection=True,
        ).validate()


def test_guided_rollout_is_rescored_without_gold_hint(monkeypatch, tmp_path):
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
            return SimpleNamespace(logits=torch.zeros(
                input_ids.shape[0], input_ids.shape[1], 258
            ))

    class Environment:
        def __init__(self, *args, **kwargs):
            self.done = False
            self.turn = 0

        def reset(self, task):
            return {}

        def verify(self, sql):
            assert sql == "SELECT 123"
            return SimpleNamespace(agent_sql_valid=True, agent_sql_executable=True, correct=True)

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
        task_id="guided", source="test", db_id="db", db_path="db.sqlite",
        question="How many?", reference_sql="SELECT 123",
        difficulty="medium", split="train",
    )
    config = replace(
        _config(tmp_path, inject=True),
        top_k=None, temperature=1.0, top_p=0.99,
        clip_ratio_high=0.28, kl_beta=0.0, learning_rate=1e-6,
    )
    episode = rollout_tagged_task(
        model, tokenizer, task, EnvConfig(max_turns=3), config,
        device="cpu", gold_guided=True,
    )
    assert episode.success and episode.origin == "gold_guided"
    assert "SELECT 123" in model.generation_prompts[0]
    assert model.sampling_options[0]["top_k"] == 0
    assert model.sampling_options[0]["temperature"] == 1.0
    assert model.sampling_options[0]["top_p"] == 0.99
    clean_prompt = tokenizer.decode(episode.turns[0].input_ids[:episode.turns[0].prompt_length])
    assert "SELECT 123" not in clean_prompt
    assert len(episode.turns) == 4
    assert episode.turns[0].action_mask.sum() > 0
