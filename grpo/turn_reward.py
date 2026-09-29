from __future__ import annotations

from collections.abc import Mapping

from sql_agent.models import Difficulty

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard", "extra")
DEFAULT_TURN_BASELINES: dict[Difficulty, int] = {
    "easy": 4,
    "medium": 5,
    "hard": 7,
    "extra": 8,
}
DEFAULT_TURN_LIMITS: dict[Difficulty, int] = {
    "easy": 6,
    "medium": 7,
    "hard": 9,
    "extra": 10,
}


def validate_difficulty_map(values: Mapping[str, int], *, name: str) -> None:
    if set(values) != set(DIFFICULTIES):
        raise ValueError(f"{name} must define easy/medium/hard/extra")
    if any(not isinstance(value, int) or value < 1 for value in values.values()):
        raise ValueError(f"{name} values must be positive integers")


def difficulty_turn_reward(
    *,
    correct: bool,
    difficulty: Difficulty,
    turns_used: int,
    baselines: Mapping[str, int],
    step_reward: float,
) -> float:
    """Return zero for failure, otherwise reward correctness and relative efficiency."""
    if turns_used < 1:
        raise ValueError("turns_used must be positive")
    if difficulty not in baselines:
        raise ValueError(f"missing turn baseline for {difficulty}")
    if step_reward <= 0:
        raise ValueError("step_reward must be positive")
    if not correct:
        return 0.0
    reward = 1.0 + step_reward * (baselines[difficulty] - turns_used)
    if reward <= 0:
        raise ValueError("a correct trajectory must retain positive reward")
    return reward
