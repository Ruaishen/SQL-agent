from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True, slots=True)
class GroupResult:
    advantages: tuple[float, ...]
    mean_reward: float
    reward_std: float
    kind: str


@dataclass(frozen=True, slots=True)
class HierarchicalGroupResult:
    group: GroupResult
    execution_advantages: tuple[float, ...]
    schema_rank_advantages: tuple[float, ...]
    submission_rank_advantages: tuple[float, ...]
    diagnostic_composite_rewards: tuple[float, ...]
    schema_rescued: bool


def normalize_group_rewards(rewards: list[float], *, epsilon: float = 1e-6) -> GroupResult:
    if len(rewards) < 2:
        raise ValueError("a GRPO group requires at least two rewards")
    if any(reward not in (0.0, 1.0) for reward in rewards):
        raise ValueError("execution rewards must be binary")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = sqrt(variance)
    if all(reward == 0.0 for reward in rewards):
        kind = "all_zero"
    elif all(reward == 1.0 for reward in rewards):
        kind = "all_one"
    else:
        kind = "mixed"
    advantages = tuple((reward - mean) / (std + epsilon) for reward in rewards)
    return GroupResult(advantages, mean, std, kind)


def normalize_shaped_group_rewards(
    rewards: list[float],
    execution_rewards: list[float],
    *,
    epsilon: float = 1e-6,
) -> GroupResult:
    """Normalize non-binary rewards while retaining outcome-based group labels."""
    if len(rewards) < 2 or len(rewards) != len(execution_rewards):
        raise ValueError("shaped reward groups require aligned rewards and outcomes")
    if any(reward < 0.0 for reward in rewards):
        raise ValueError("shaped rewards must be non-negative")
    if any(reward not in (0.0, 1.0) for reward in execution_rewards):
        raise ValueError("execution rewards must be binary")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = sqrt(variance)
    if all(reward == 0.0 for reward in execution_rewards):
        kind = "all_zero"
    elif all(reward == 1.0 for reward in execution_rewards):
        kind = "all_one"
    else:
        kind = "mixed"
    advantages = tuple((reward - mean) / (std + epsilon) for reward in rewards)
    return GroupResult(advantages, mean, std, kind)


def _tied_centered_ranks(values: list[float]) -> tuple[float, ...]:
    if len(values) < 2:
        return (0.0,) * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        for index in ordered[start:end]:
            ranks[index] = average_rank
        start = end
    midpoint = (len(values) - 1) / 2
    if midpoint == 0:
        return (0.0,) * len(values)
    return tuple((rank - midpoint) / midpoint for rank in ranks)


def hierarchical_schema_advantages(
    execution_rewards: list[float],
    schema_scores: list[float],
    *,
    coefficient: float = 0.2,
    submission_scores: list[float] | None = None,
    submission_coefficient: float = 0.0,
    diagnostic_eta: float = 0.05,
) -> HierarchicalGroupResult:
    if len(execution_rewards) != len(schema_scores):
        raise ValueError("execution reward/schema score count mismatch")
    if any(not 0.0 <= score <= 1.0 for score in schema_scores):
        raise ValueError("schema scores must be in [0, 1]")
    if submission_scores is None:
        submission_scores = [0.0] * len(execution_rewards)
    if len(submission_scores) != len(execution_rewards):
        raise ValueError("execution reward/submission score count mismatch")
    execution = normalize_group_rewards(execution_rewards)
    failure_indices = [index for index, reward in enumerate(execution_rewards) if reward == 0.0]
    failure_ranks = _tied_centered_ranks([schema_scores[index] for index in failure_indices])
    schema_ranks = [0.0] * len(schema_scores)
    for index, rank in zip(failure_indices, failure_ranks, strict=True):
        schema_ranks[index] = rank
    submission_failure_ranks = _tied_centered_ranks(
        [submission_scores[index] for index in failure_indices]
    )
    submission_ranks = [0.0] * len(schema_scores)
    for index, rank in zip(failure_indices, submission_failure_ranks, strict=True):
        submission_ranks[index] = rank
    advantages = tuple(
        value + coefficient * schema_rank + submission_coefficient * submission_rank
        for value, schema_rank, submission_rank in zip(
            execution.advantages, schema_ranks, submission_ranks, strict=True
        )
    )
    diagnostic = tuple(
        reward + (1.0 - reward) * diagnostic_eta * score
        for reward, score in zip(execution_rewards, schema_scores, strict=True)
    )
    schema_rescued = execution.kind == "all_zero" and len(set(advantages)) > 1
    return HierarchicalGroupResult(
        group=GroupResult(
            advantages=advantages,
            mean_reward=execution.mean_reward,
            reward_std=execution.reward_std,
            kind=execution.kind,
        ),
        execution_advantages=execution.advantages,
        schema_rank_advantages=tuple(schema_ranks),
        submission_rank_advantages=tuple(submission_ranks),
        diagnostic_composite_rewards=diagnostic,
        schema_rescued=schema_rescued,
    )


def summarize_groups(groups: list[GroupResult]) -> dict[str, float | int]:
    if not groups:
        raise ValueError("cannot summarize zero GRPO groups")
    count = len(groups)
    kinds = {
        kind: sum(group.kind == kind for group in groups)
        for kind in ("all_zero", "all_one", "mixed")
    }
    return {
        "groups": count,
        "all_zero_groups": kinds["all_zero"],
        "all_one_groups": kinds["all_one"],
        "mixed_groups": kinds["mixed"],
        "all_zero_group_ratio": kinds["all_zero"] / count,
        "all_one_group_ratio": kinds["all_one"] / count,
        "mixed_group_ratio": kinds["mixed"] / count,
        "mean_group_reward_std": sum(group.reward_std for group in groups) / count,
        "mean_reward": sum(group.mean_reward for group in groups) / count,
    }
