from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from grpo.turn_reward import (
    DEFAULT_TURN_BASELINES,
    DEFAULT_TURN_LIMITS,
    validate_difficulty_map,
)


@dataclass(frozen=True, slots=True)
class GrpoConfig:
    student_model: Path
    reference_model: Path
    train_data: Path
    output_dir: Path
    seed: int = 42
    max_steps: int = 32
    prompts_per_step: int = 8
    rollouts_per_prompt: int = 4
    interaction_protocol: str = "legacy"
    gold_injection: bool = False
    gold_injection_max_attempts: int = 3
    homogeneous_resampling_max_rollouts: int = 0
    easy_tasks: int = 0
    medium_tasks: int = 64
    hard_tasks: int = 0
    extra_tasks: int = 0
    candidate_tasks: Path | None = None
    ability_map: Path | None = None
    reuse_pool_dir: Path | None = None
    exclusion_files: tuple[Path, ...] = ()
    source_difficulty_quotas: dict[str, dict[str, int]] = field(default_factory=dict)
    max_assistant_turns: int = 10
    max_sequence_tokens: int = 16384
    max_action_tokens: int = 384
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = 20
    clip_ratio: float = 0.2
    clip_ratio_high: float | None = None
    kl_beta: float = 0.001
    learning_rate: float = 5e-7
    gradient_clip_norm: float = 1.0
    update_micro_batch_size: int = 8
    reference_score_micro_batch_size: int = 16
    gradient_checkpointing: bool = True
    target_gpu_memory_utilization: float = 0.90
    checkpoint_interval: int = 0
    reward_mode: str = "binary_execution"
    turn_reward_step: float = 0.1
    turn_reward_baselines: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_TURN_BASELINES)
    )
    max_assistant_turns_by_difficulty: dict[str, int] = field(default_factory=dict)
    schema_advantage_coefficient: float = 0.2
    submission_advantage_coefficient: float = 0.0
    schema_diagnostic_eta: float = 0.05
    schema_component_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    schema_single_table_weights: tuple[float, float, float] = (0.45, 0.45, 0.10)
    schema_multi_table_weights: tuple[float, float, float] = (0.30, 0.30, 0.40)
    execute_sql_schema_factor: float = 1.0

    @classmethod
    def load(cls, path: Path) -> GrpoConfig:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("GRPO config must be a YAML mapping")
        for key in ("student_model", "reference_model", "train_data", "output_dir"):
            raw[key] = Path(raw[key])
        for key in ("candidate_tasks", "ability_map", "reuse_pool_dir"):
            if raw.get(key) is not None:
                raw[key] = Path(raw[key])
        raw["exclusion_files"] = tuple(Path(value) for value in raw.get("exclusion_files", ()))
        for key in ("turn_reward_baselines", "max_assistant_turns_by_difficulty"):
            if key in raw:
                raw[key] = {str(name): int(value) for name, value in raw[key].items()}
        for key in (
            "schema_component_weights",
            "schema_single_table_weights",
            "schema_multi_table_weights",
        ):
            if key in raw:
                raw[key] = tuple(float(value) for value in raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        optional_paths = tuple(
            path
            for path in (self.candidate_tasks, self.ability_map, self.reuse_pool_dir)
            if path is not None
        )
        for path in (
            self.student_model,
            self.reference_model,
            self.train_data,
            *optional_paths,
            *self.exclusion_files,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        for value in (
            self.max_steps,
            self.prompts_per_step,
            self.rollouts_per_prompt,
            self.max_assistant_turns,
        ):
            if value < 1:
                raise ValueError("GRPO counts must be positive")
        if self.rollouts_per_prompt < 2:
            raise ValueError("GRPO requires at least two rollouts per prompt")
        if self.interaction_protocol not in {"legacy", "reasoning_tool"}:
            raise ValueError("unsupported GRPO interaction protocol")
        if self.gold_injection and (
            self.interaction_protocol != "reasoning_tool" or self.reward_mode != "binary_execution"
        ):
            raise ValueError("gold injection requires reasoning_tool and binary_execution")
        if self.gold_injection and self.homogeneous_resampling_max_rollouts:
            raise ValueError("gold injection requires fixed on-policy group size")
        if self.gold_injection_max_attempts < 1:
            raise ValueError("gold_injection_max_attempts must be positive")
        if self.homogeneous_resampling_max_rollouts:
            if self.reward_mode != "binary_execution":
                raise ValueError("homogeneous resampling requires binary execution reward")
            if self.homogeneous_resampling_max_rollouts <= self.rollouts_per_prompt:
                raise ValueError("homogeneous resampling max must exceed rollouts_per_prompt")
        difficulty_counts = {
            "easy": self.easy_tasks,
            "medium": self.medium_tasks,
            "hard": self.hard_tasks,
            "extra": self.extra_tasks,
        }
        if any(value < 0 for value in difficulty_counts.values()) or self.task_count < 1:
            raise ValueError("GRPO task-pool counts must be non-negative and non-empty")
        if self.source_difficulty_quotas:
            if self.candidate_tasks is None or self.ability_map is None:
                raise ValueError("curriculum GRPO requires candidate_tasks and ability_map")
            allowed_sources = {"zero", "boundary", "exploration", "mastered"}
            allowed_difficulties = set(difficulty_counts)
            if set(self.source_difficulty_quotas) != allowed_sources:
                raise ValueError("curriculum quotas must define zero/boundary/exploration/mastered")
            quota_total = 0
            quota_difficulties = {name: 0 for name in allowed_difficulties}
            for source, quotas in self.source_difficulty_quotas.items():
                if set(quotas) != allowed_difficulties or any(
                    value < 0 for value in quotas.values()
                ):
                    raise ValueError(f"invalid curriculum quotas for {source}")
                quota_total += sum(quotas.values())
                for difficulty, value in quotas.items():
                    quota_difficulties[difficulty] += value
            if quota_total != self.task_count or quota_difficulties != difficulty_counts:
                raise ValueError("curriculum source/difficulty quotas do not match task counts")
        if self.max_action_tokens >= self.max_sequence_tokens:
            raise ValueError("max_action_tokens must be below max_sequence_tokens")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or (
            self.top_k is not None and self.top_k < 0
        ):
            raise ValueError("invalid rollout sampling parameters")
        if (
            not 0 <= self.clip_ratio < 1
            or (
                self.clip_ratio_high is not None
                and not self.clip_ratio <= self.clip_ratio_high < 1
            )
            or self.kl_beta < 0
        ):
            raise ValueError("invalid GRPO clip/KL parameters")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer parameters must be positive")
        if self.update_micro_batch_size < 1 or self.reference_score_micro_batch_size < 1:
            raise ValueError("micro-batch sizes must be positive")
        if not 0.5 <= self.target_gpu_memory_utilization <= 0.95:
            raise ValueError("target_gpu_memory_utilization must be in [0.5, 0.95]")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.reward_mode not in {
            "binary_execution",
            "difficulty_adaptive_turn",
            "schema_hierarchical",
            "schema_hierarchical_v2",
        }:
            raise ValueError("unsupported GRPO reward mode")
        if self.reward_mode == "difficulty_adaptive_turn":
            validate_difficulty_map(self.turn_reward_baselines, name="turn_reward_baselines")
            validate_difficulty_map(
                self.max_assistant_turns_by_difficulty,
                name="max_assistant_turns_by_difficulty",
            )
            if self.turn_reward_step <= 0:
                raise ValueError("turn_reward_step must be positive")
            if self.max_assistant_turns < max(self.max_assistant_turns_by_difficulty.values()):
                raise ValueError("max_assistant_turns must cover every difficulty turn limit")
            for difficulty, baseline in self.turn_reward_baselines.items():
                limit = self.max_assistant_turns_by_difficulty[difficulty]
                if baseline > limit or 1.0 + self.turn_reward_step * (baseline - limit) <= 0:
                    raise ValueError("turn reward baselines/limits produce an invalid reward")
        if not 0 <= self.schema_advantage_coefficient <= 0.4:
            raise ValueError("schema_advantage_coefficient must be in [0, 0.4]")
        if not 0 <= self.submission_advantage_coefficient <= 0.4:
            raise ValueError("submission_advantage_coefficient must be in [0, 0.4]")
        if self.schema_advantage_coefficient + self.submission_advantage_coefficient > 0.4:
            raise ValueError("combined auxiliary advantage coefficient must be <= 0.4")
        if not 0 <= self.schema_diagnostic_eta < 1:
            raise ValueError("schema_diagnostic_eta must be in [0, 1)")
        for weights in (
            self.schema_component_weights,
            self.schema_single_table_weights,
            self.schema_multi_table_weights,
        ):
            if (
                len(weights) != 3
                or any(value < 0 for value in weights)
                or abs(sum(weights) - 1.0) > 1e-9
            ):
                raise ValueError(
                    "schema component weights must be three nonnegative values summing to 1"
                )
        if not 0 <= self.execute_sql_schema_factor <= 1:
            raise ValueError("execute_sql_schema_factor must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {
            key: (
                str(item)
                if isinstance(item, Path)
                else [str(path) for path in item]
                if key == "exclusion_files"
                else item
            )
            for key, item in value.items()
        }

    @property
    def task_count(self) -> int:
        return self.easy_tasks + self.medium_tasks + self.hard_tasks + self.extra_tasks

    def assistant_turn_limit(self, difficulty: str) -> int:
        if self.reward_mode != "difficulty_adaptive_turn":
            return self.max_assistant_turns
        return self.max_assistant_turns_by_difficulty.get(
            difficulty, DEFAULT_TURN_LIMITS[difficulty]
        )
