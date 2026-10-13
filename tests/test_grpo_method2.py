import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from grpo.config import GrpoConfig
from grpo.guided_injection import inject_guided_trajectory
from grpo.rollout import RolloutEpisode, RolloutTurn
from grpo.train import _episode_weights, collect_groups
from sft.regenerate_reasoning import PROMPTS
from sql_agent.deepseek import Completion


def tagged(name, arguments):
    return (
        "<reasoning>Check the database.</reasoning><tool>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool>"
    )


def setup(tmp_path, task):
    cfg = GrpoConfig(
        student_model=tmp_path,
        reference_model=tmp_path,
        train_data=tmp_path,
        output_dir=tmp_path,
        medium_tasks=1,
        rollouts_per_prompt=4,
        interaction_protocol="reasoning_tool",
        method2_guided_injection=True,
        temperature=1,
        top_p=1,
        top_k=None,
        kl_beta=0,
        max_action_tokens=2048,
    )
    originals = [RolloutEpisode(task.task_id, task.db_id, task.difficulty) for _ in range(4)]
    for episode in originals:
        episode.turns = [
            RolloutTurn(
                task.task_id,
                1,
                torch.tensor([1, 2]),
                1,
                torch.tensor([True]),
                torch.tensor([-1.0]),
                "invalid",
                {},
            )
        ]
    return cfg, originals


class Tokenizer:
    all_special_ids = [0]
    eos_token_id = pad_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(char) + 1 for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(int(token) - 1) for token in ids if int(token) > 0)

    def apply_chat_template(self, messages, **kwargs):
        content = "".join(f"<{m['role']}>{m['content']}\n" for m in messages) + "<assistant>"
        return torch.tensor([self.encode(content)], dtype=torch.long)


class Model:
    def __init__(self, tokenizer, sql):
        self.tokenizer = tokenizer
        self.prompts, self.inputs = [], []
        self.actions = [
            tagged("list_tables", {}),
            tagged("execute_sql", {"sql": sql}),
            tagged("submit_sql", {"sql": sql}),
        ]

    def eval(self):
        return self

    def generate(self, *, input_ids, **kwargs):
        self.prompts.append(self.tokenizer.decode(input_ids[0]))
        text = "\n" + self.actions[(len(self.prompts) - 1) % 3] + " \n"
        return torch.cat([input_ids, torch.tensor([self.tokenizer.encode(text)])], dim=1)

    def __call__(self, *, input_ids, **kwargs):
        self.inputs.append(input_ids.clone())
        logits = torch.zeros((*input_ids.shape, 256))
        boost = 2.0 if "<teacher_only_sql>" in self.tokenizer.decode(input_ids[0]) else 0.0
        logits[:, :-1].scatter_(-1, input_ids[:, 1:, None], boost)
        return SimpleNamespace(logits=logits)


class Auditor:
    def __init__(self, verdicts):
        self.verdicts, self.calls = list(verdicts), []

    def complete_thinking_text(self, messages, **kwargs):
        self.calls.append(messages)
        return Completion({"content": self.verdicts.pop(0)}, "stop", None, "test", {})


@pytest.mark.parametrize("accepted_at", [1, 2, 3, None])
def test_real_sql_source_probability_prompt_audit_and_three_attempt_limit(
    sample_db,
    config,
    task,
    tmp_path,
    accepted_at,
):
    task = replace(task, split="train")
    cfg, originals = setup(tmp_path, task)
    tokenizer = Tokenizer()
    model = Model(tokenizer, task.reference_sql)
    verdicts = ["reject"] * (3 if accepted_at is None else accepted_at - 1)
    if accepted_at is not None:
        verdicts += ["accept"]
    auditor = Auditor(verdicts)
    candidate = inject_guided_trajectory(
        model, tokenizer, task, config, cfg, originals, client=auditor, device="cpu"
    )
    assert len(auditor.calls) == (accepted_at or 3)
    assert len(model.prompts) == 3 * (accepted_at or 3)
    assert originals[0].method2_info["attempts"] == (accepted_at or 3)
    assert all("<teacher_only_sql>" in prompt for prompt in model.prompts)
    for call in auditor.calls:
        assert call[0]["content"] == (PROMPTS / "external_answer_leak_audit_zh.md").read_text(
            encoding="utf-8"
        )
        payload = json.loads(call[1]["content"])
        assert set(payload) == {"question", "messages", "trajectory_comparison"}
        assert "<teacher_only_sql>" not in call[1]["content"]
    if candidate is None:
        assert accepted_at is None
        return
    assert candidate.origin == "guided_old_student" and candidate.success
    for turn in candidate.turns:
        assert turn.response_text.startswith("\n") and turn.response_text.endswith(" \n")
        assert "<teacher_only_sql>" not in tokenizer.decode(turn.input_ids)
        assert "<teacher_only_sql>" in tokenizer.decode(turn.source_prompt_ids)
        expected = 2 - math.log(math.exp(2) + 255)
        torch.testing.assert_close(
            turn.old_log_probs[turn.action_mask],
            torch.full_like(turn.old_log_probs[turn.action_mask], expected),
        )
        assert not turn.old_log_probs.requires_grad
        assert not turn.action_mask[: turn.prompt_length - 1].any()


@pytest.mark.parametrize("has_success", [False, True])
def test_group_integration_keeps_originals_and_recomputes_advantages(
    monkeypatch, tmp_path, task, config, has_success
):
    task = replace(task, split="train")
    cfg, originals = setup(tmp_path, task)
    originals[0].success = has_success
    candidate = RolloutEpisode(
        task.task_id, task.db_id, task.difficulty, success=True, origin="guided_old_student"
    )
    calls = []
    monkeypatch.setattr(
        "grpo.train.rollout_tagged_task",
        lambda *args, rollout_index, **kwargs: originals[rollout_index],
    )

    def inject(*args, **kwargs):
        calls.append(True)
        return candidate

    monkeypatch.setattr("grpo.guided_injection.inject_guided_trajectory", inject)
    episodes, groups, details, sizes = collect_groups(None, None, [task], config, cfg)
    if has_success:
        assert not calls and episodes == originals and sizes == [4]
    else:
        assert calls == [True] and episodes == originals + [candidate] and sizes == [5]
        assert groups[0].advantages[-1] == pytest.approx(2, abs=1e-5)
        assert groups[0].advantages[0] == pytest.approx(-0.5, abs=1e-5)
        assert len(details) == 5
    weights = _episode_weights([4, 5], group_equal=True)
    assert sum(weights[:4]) == sum(weights[4:]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "override",
    [{"method2_max_attempts": 4}, {"top_p": 0.99}, {"temperature": 0.7}, {"kl_beta": 0.01}],
)
def test_method2_invalid_contract_rejected(tmp_path, task, override):
    cfg, _ = setup(tmp_path, task)
    with pytest.raises(ValueError):
        replace(cfg, **override).validate()
