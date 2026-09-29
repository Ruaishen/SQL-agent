from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from grpo.config import GrpoConfig
from grpo.groups import (
    normalize_group_rewards,
    normalize_shaped_group_rewards,
    summarize_groups,
)
from grpo.loss import grpo_token_loss
from grpo.prepare import (
    build_curriculum_pool,
    task_fingerprint,
)
from grpo.turn_reward import (
    DEFAULT_TURN_BASELINES,
    DEFAULT_TURN_LIMITS,
    difficulty_turn_reward,
)


def _curriculum_task(task_id: str, difficulty: str, question: str | None = None) -> dict:
    return {
        "task_id": task_id,
        "source": "test",
        "db_id": f"db-{task_id}",
        "db_path": f"database/db-{task_id}/db.sqlite",
        "question": question or f"question {task_id}",
        "reference_sql": "SELECT 1",
        "difficulty": difficulty,
        "split": "train",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_curriculum_pool_excludes_sft_ids_and_fingerprints(tmp_path: Path) -> None:
    selected = [
        _curriculum_task("zero", "easy"),
        _curriculum_task("boundary", "medium"),
        _curriculum_task("explore", "hard"),
        _curriculum_task("mastered", "extra"),
    ]
    teacher = _curriculum_task("teacher", "easy")
    semantic_copy = {**teacher, "task_id": "teacher-copy"}
    balanced = _curriculum_task("balanced", "medium")
    all_tasks = selected + [teacher, semantic_copy, balanced]
    train_path = tmp_path / "train.jsonl"
    candidate_path = tmp_path / "candidates.jsonl"
    ability_path = tmp_path / "ability.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    balanced_path = tmp_path / "balanced.jsonl"
    _write_jsonl(train_path, all_tasks)
    _write_jsonl(candidate_path, all_tasks)
    _write_jsonl(
        ability_path,
        [
            {"task_id": "zero", "correct_count": 0},
            {"task_id": "boundary", "correct_count": 3},
            {"task_id": "mastered", "correct_count": 6},
        ],
    )
    _write_jsonl(teacher_path, [{"task_id": "teacher"}])
    _write_jsonl(balanced_path, [{"task_id": "balanced"}])
    config = GrpoConfig(
        student_model=tmp_path,
        reference_model=tmp_path,
        train_data=train_path,
        output_dir=tmp_path / "out",
        candidate_tasks=candidate_path,
        ability_map=ability_path,
        exclusion_files=(teacher_path, balanced_path),
        rollouts_per_prompt=6,
        easy_tasks=1,
        medium_tasks=1,
        hard_tasks=1,
        extra_tasks=1,
        source_difficulty_quotas={
            "zero": {"easy": 1, "medium": 0, "hard": 0, "extra": 0},
            "boundary": {"easy": 0, "medium": 1, "hard": 0, "extra": 0},
            "exploration": {"easy": 0, "medium": 0, "hard": 1, "extra": 0},
            "mastered": {"easy": 0, "medium": 0, "hard": 0, "extra": 1},
        },
    )

    tasks, metadata, diagnostics = build_curriculum_pool(config)

    assert {task["task_id"] for task in tasks} == {"zero", "boundary", "explore", "mastered"}
    assert Counter(record["selection_source"] for record in metadata) == {
        "zero": 1,
        "boundary": 1,
        "exploration": 1,
        "mastered": 1,
    }
    assert {record["task_id"] for record in metadata} == {task["task_id"] for task in tasks}
    assert diagnostics["selected_exclusion_id_intersection"] == 0
    assert diagnostics["selected_exclusion_fingerprint_intersection"] == 0
    assert task_fingerprint(teacher) == task_fingerprint(semantic_copy)


def test_group_advantages_and_variance() -> None:
    group = normalize_group_rewards([0.0, 0.0, 1.0, 1.0])
    assert group.kind == "mixed"
    assert group.mean_reward == 0.5
    assert group.reward_std == 0.5
    assert sum(group.advantages) == pytest.approx(0.0)
    assert group.advantages[0] == pytest.approx(-1.0, rel=1e-5)


def test_constant_reward_group_has_zero_advantages() -> None:
    zero = normalize_group_rewards([0.0] * 4)
    one = normalize_group_rewards([1.0] * 4)
    assert zero.kind == "all_zero"
    assert one.kind == "all_one"
    assert zero.advantages == (0.0,) * 4
    assert one.advantages == (0.0,) * 4
    summary = summarize_groups([zero, one])
    assert summary["mixed_group_ratio"] == 0.0
    assert summary["all_zero_group_ratio"] == 0.5


def test_grpo_clipping_and_gradient_direction() -> None:
    current = torch.tensor([[0.3, -0.3]], requires_grad=True)
    old = torch.zeros_like(current)
    reference = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)
    output = grpo_token_loss(
        current,
        old,
        reference,
        torch.tensor([1.0]),
        mask,
        clip_ratio=0.2,
        kl_beta=0.0,
    )
    loss = output.token_loss.mean()
    loss.backward()
    assert output.clipped.tolist() == [[1.0, 1.0]]
    assert current.grad is not None
    assert current.grad[0, 0].item() == pytest.approx(0.0)
    assert current.grad[0, 1].item() < 0.0


def test_grpo_reference_kl_is_nonnegative() -> None:
    current = torch.tensor([[0.2, -0.4]], requires_grad=True)
    output = grpo_token_loss(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        torch.tensor([0.0]),
        torch.ones_like(current, dtype=torch.bool),
        clip_ratio=0.2,
        kl_beta=0.1,
    )
    assert torch.all(output.kl >= 0)
    output.token_loss.mean().backward()
    assert current.grad is not None


def test_turn_reward_is_zero_for_failure_and_relative_to_difficulty_baseline() -> None:
    assert (
        difficulty_turn_reward(
            correct=False,
            difficulty="extra",
            turns_used=1,
            baselines=DEFAULT_TURN_BASELINES,
            step_reward=0.1,
        )
        == 0.0
    )
    assert difficulty_turn_reward(
        correct=True,
        difficulty="easy",
        turns_used=3,
        baselines=DEFAULT_TURN_BASELINES,
        step_reward=0.1,
    ) == pytest.approx(1.1)
    assert difficulty_turn_reward(
        correct=True,
        difficulty="hard",
        turns_used=9,
        baselines=DEFAULT_TURN_BASELINES,
        step_reward=0.1,
    ) == pytest.approx(0.8)


def test_shaped_reward_gives_all_correct_groups_efficiency_signal() -> None:
    group = normalize_shaped_group_rewards([1.1, 1.0, 0.9], [1.0, 1.0, 1.0])
    assert group.kind == "all_one"
    assert group.advantages[0] > group.advantages[1] > group.advantages[2]
    assert sum(group.advantages) == pytest.approx(0.0)


def test_difficulty_adaptive_turn_config() -> None:
    config = GrpoConfig(
        student_model=Path("."),
        reference_model=Path("."),
        train_data=Path("."),
        output_dir=Path("."),
        easy_tasks=1,
        medium_tasks=0,
        max_assistant_turns=10,
        reward_mode="difficulty_adaptive_turn",
        max_assistant_turns_by_difficulty=dict(DEFAULT_TURN_LIMITS),
    )
    config.validate()
    assert config.assistant_turn_limit("easy") == 6
    assert config.assistant_turn_limit("medium") == 7
    assert config.assistant_turn_limit("hard") == 9
    assert config.assistant_turn_limit("extra") == 10
