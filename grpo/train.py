from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.config import GrpoConfig
from grpo.groups import (
    GroupResult,
    hierarchical_schema_advantages,
    normalize_group_rewards,
    normalize_shaped_group_rewards,
    summarize_groups,
)
from grpo.loss import grpo_token_loss
from grpo.schema_reward import SchemaRewarder
from grpo.turn_reward import difficulty_turn_reward
from grpo.rollout import (
    RolloutEpisode,
    RolloutTurn,
    collate_training_turns,
    rollout_task,
    score_turns_with_teacher,
)
from grpo.scoring import causal_selected_log_probs
from grpo.tagged_rollout import rollout_tagged_task
from sql_agent.config import EnvConfig
from sql_agent.models import TaskRecord


def _load_pool(path: Path) -> list[TaskRecord]:
    with path.open(encoding="utf-8") as handle:
        return [TaskRecord.from_dict(json.loads(line)) for line in handle if line.strip()]


def _load_metadata(path: Path, pool: list[TaskRecord]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {
            task.task_id: {
                "task_id": task.task_id,
                "selection_source": "legacy",
                "prior_correct_count": None,
            }
            for task in pool
        }
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    metadata = {record["task_id"]: record for record in records}
    if set(metadata) != {task.task_id for task in pool}:
        raise ValueError("GRPO task metadata does not match task pool")
    return metadata


def _step_tasks(pool: list[TaskRecord], step: int, count: int) -> list[TaskRecord]:
    start = ((step - 1) * count) % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(count)]


def collect_groups(
    student,
    tokenizer,
    tasks: list[TaskRecord],
    env_config: EnvConfig,
    config: GrpoConfig,
    schema_rewarder: SchemaRewarder | None = None,
) -> tuple[
    list[RolloutEpisode],
    list[GroupResult],
    list[dict[str, Any]],
    list[int],
]:
    episodes: list[RolloutEpisode] = []
    groups: list[GroupResult] = []
    reward_details: list[dict[str, Any]] = []
    group_sizes: list[int] = []
    for task in tasks:
        if config.interaction_protocol == "reasoning_tool" and task.split != "train":
            raise ValueError("reasoning_tool GRPO requires training split tasks")
        turn_limit = config.assistant_turn_limit(task.difficulty)
        task_env_config = replace(env_config, max_turns=turn_limit)
        task_rollout_config = replace(config, max_assistant_turns=turn_limit)
        rollout_fn = rollout_tagged_task if config.interaction_protocol == "reasoning_tool" else rollout_task
        task_episodes = [
            rollout_fn(
                student,
                tokenizer,
                task,
                task_env_config,
                task_rollout_config,
                rollout_index=rollout_index,
            )
            for rollout_index in range(config.rollouts_per_prompt)
        ]
        if config.homogeneous_resampling_max_rollouts:
            initial_success = task_episodes[0].success
            while len(task_episodes) < config.homogeneous_resampling_max_rollouts and all(
                episode.success == initial_success for episode in task_episodes
            ):
                task_episodes.append(
                    rollout_fn(
                        student,
                        tokenizer,
                        task,
                        task_env_config,
                        task_rollout_config,
                        rollout_index=len(task_episodes),
                    )
                )
        original_successes = sum(episode.success for episode in task_episodes)
        injection_attempts = 0
        if config.gold_injection and original_successes == 0:
            for attempt in range(config.gold_injection_max_attempts):
                injection_attempts += 1
                guided = rollout_tagged_task(
                    student, tokenizer, task, task_env_config, task_rollout_config,
                    rollout_index=len(task_episodes) + attempt,
                    gold_guided=True,
                )
                if guided.success and guided.submitted and guided.turns:
                    task_episodes.append(guided)
                    break
        group_sizes.append(len(task_episodes))
        episodes.extend(task_episodes)
        execution_rewards = [float(episode.success) for episode in task_episodes]
        if config.reward_mode == "difficulty_adaptive_turn":
            turn_rewards = [
                difficulty_turn_reward(
                    correct=episode.success,
                    difficulty=task.difficulty,
                    turns_used=len(episode.turns),
                    baselines=config.turn_reward_baselines,
                    step_reward=config.turn_reward_step,
                )
                for episode in task_episodes
            ]
            execution_group = normalize_group_rewards(execution_rewards)
            group = normalize_shaped_group_rewards(turn_rewards, execution_rewards)
            groups.append(group)
            efficiency_signal = group.kind == "all_one" and any(
                abs(value) > 1e-12 for value in group.advantages
            )
            for index, (episode, reward) in enumerate(
                zip(task_episodes, turn_rewards, strict=True)
            ):
                reward_details.append(
                    {
                        "raw_execution_reward": execution_rewards[index],
                        "turn_reward": reward,
                        "turn_reward_adjustment": reward - execution_rewards[index],
                        "turn_reward_baseline": config.turn_reward_baselines[task.difficulty],
                        "turn_limit": turn_limit,
                        "turns_used": len(episode.turns),
                        "execution_advantage": execution_group.advantages[index],
                        "turn_reward_advantage": group.advantages[index],
                        "schema_rank_advantage": 0.0,
                        "submission_rank_advantage": 0.0,
                        "diagnostic_composite_reward": reward,
                        "schema_rescued_group": False,
                        "turn_efficiency_signal_group": efficiency_signal,
                    }
                )
        elif config.reward_mode in {"schema_hierarchical", "schema_hierarchical_v2"}:
            if schema_rewarder is None:
                raise ValueError("schema_hierarchical mode requires a SchemaRewarder")
            scores = [schema_rewarder.score_episode(task, episode) for episode in task_episodes]
            shaped = hierarchical_schema_advantages(
                execution_rewards,
                [score.schema_score for score in scores],
                coefficient=config.schema_advantage_coefficient,
                submission_scores=[
                    1.0
                    if score.candidate_source == "execute_sql"
                    else -1.0
                    for score in scores
                ],
                submission_coefficient=config.submission_advantage_coefficient,
                diagnostic_eta=config.schema_diagnostic_eta,
            )
            groups.append(shaped.group)
            for index, score in enumerate(scores):
                reward_details.append(
                    {
                        **score.to_dict(),
                        "raw_execution_reward": execution_rewards[index],
                        "execution_advantage": shaped.execution_advantages[index],
                        "schema_rank_advantage": shaped.schema_rank_advantages[index],
                        "submission_rank_advantage": shaped.submission_rank_advantages[index],
                        "diagnostic_composite_reward": shaped.diagnostic_composite_rewards[index],
                        "schema_rescued_group": shaped.schema_rescued,
                    }
                )
        else:
            group = normalize_group_rewards(execution_rewards)
            groups.append(group)
            for index, reward in enumerate(execution_rewards):
                reward_details.append(
                    {
                        "trajectory_origin": task_episodes[index].origin,
                        "original_student_correct_count": original_successes,
                        "gold_injection_attempts": injection_attempts,
                        "gold_injection_succeeded": len(task_episodes) > config.rollouts_per_prompt and task_episodes[-1].origin == "gold_guided",
                        "raw_execution_reward": reward,
                        "execution_advantage": group.advantages[index],
                        "schema_rank_advantage": 0.0,
                        "submission_rank_advantage": 0.0,
                        "diagnostic_composite_reward": reward,
                        "schema_rescued_group": False,
                    }
                )
    return episodes, groups, reward_details, group_sizes


def _training_records(
    episodes: list[RolloutEpisode],
    groups: list[GroupResult],
    group_sizes: list[int],
) -> list[tuple[RolloutTurn, float, int]]:
    if len(group_sizes) != len(groups) or len(episodes) != sum(group_sizes):
        raise ValueError("episode/group count mismatch")
    records: list[tuple[RolloutTurn, float, int]] = []
    start = 0
    for group, group_size in zip(groups, group_sizes, strict=True):
        task_episodes = episodes[start : start + group_size]
        start += group_size
        for episode, advantage in zip(task_episodes, group.advantages, strict=True):
            episode_tokens = sum(int(turn.action_mask.sum()) for turn in episode.turns)
            if episode_tokens < 1:
                raise ValueError("GRPO episode contains no action tokens")
            records.extend((turn, advantage, episode_tokens) for turn in episode.turns)
    return records


def _episode_weights(group_sizes: list[int], *, group_equal: bool) -> list[float]:
    if not group_sizes or any(size < 1 for size in group_sizes):
        raise ValueError("GRPO group sizes must be positive")
    if group_equal:
        return [1.0 / (len(group_sizes) * size) for size in group_sizes for _ in range(size)]
    episode_count = sum(group_sizes)
    return [1.0 / episode_count] * episode_count


def _policy_clip_counts(output, action_mask, advantages, origins: list[str]) -> dict[str, dict[str, int]]:
    if len(origins) != action_mask.shape[0]:
        raise ValueError("clip origins must align with training turns")
    eligible = action_mask & advantages[:, None].ne(0)
    clipped = output.policy_clipped.bool() & eligible
    outside = output.clipped.bool() & action_mask
    result: dict[str, dict[str, int]] = {}
    for origin in set(origins):
        selected = torch.tensor(
            [value == origin for value in origins], device=action_mask.device
        )[:, None]
        values = torch.stack((
            (clipped & selected).sum(), (eligible & selected).sum(),
            (outside & selected).sum(), (action_mask & selected).sum(),
        )).tolist()
        result[origin] = dict(zip(("clipped", "eligible", "outside", "tokens"), values, strict=True))
    return result


def _update_once(
    student,
    tokenizer,
    optimizer,
    episodes: list[RolloutEpisode],
    groups: list[GroupResult],
    config: GrpoConfig,
    micro_batch_size: int,
    group_sizes: list[int],
) -> dict[str, Any]:
    records = _training_records(
        episodes,
        groups,
        group_sizes,
    )
    group_equal = config.interaction_protocol == "reasoning_tool"
    episode_weights = _episode_weights(group_sizes, group_equal=group_equal)
    record_weights = [weight for episode, weight in zip(episodes, episode_weights, strict=True)
                      for _ in episode.turns]
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "ratio": 0.0, "clip": 0.0}
    clip_counts = {origin: {"clipped": 0, "eligible": 0, "outside": 0, "tokens": 0}
                   for origin in ("on_policy", "gold_guided")}
    record_origins = [episode.origin for episode in episodes for _ in episode.turns]
    student.train()
    student.config.use_cache = False
    if config.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    for start in range(0, len(records), micro_batch_size):
        chunk = records[start : start + micro_batch_size]
        turns = [record[0] for record in chunk]
        input_ids, attention_mask, action_mask, old_log_probs, reference_log_probs = (
            collate_training_turns(
                turns, tokenizer.pad_token_id, "cuda", require_teacher=config.kl_beta > 0
            )
        )
        advantages = torch.tensor(
            [record[1] for record in chunk], dtype=torch.float32, device="cuda"
        )
        token_weights = torch.zeros_like(old_log_probs)
        for index, (_, _, episode_tokens) in enumerate(chunk):
            token_weights[index] = action_mask[index].to(torch.float32) / (
                episode_tokens
            ) * record_weights[start + index]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = student(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            current_log_probs = causal_selected_log_probs(logits, input_ids)
            output = grpo_token_loss(
                current_log_probs,
                old_log_probs,
                reference_log_probs,
                advantages,
                action_mask,
                clip_ratio=config.clip_ratio,
                clip_ratio_high=config.clip_ratio_high,
                kl_beta=config.kl_beta,
            )
            loss = (output.token_loss * token_weights).sum()
        loss.backward()
        totals["loss"] += float(loss.detach().cpu())
        totals["policy_loss"] += float((output.policy_loss * token_weights).sum().detach().cpu())
        totals["kl"] += float((output.kl * token_weights).sum().detach().cpu())
        totals["ratio"] += float((output.ratio * token_weights).sum().detach().cpu())
        totals["clip"] += float((output.clipped * token_weights).sum().detach().cpu())
        with torch.no_grad():
            for origin, values in _policy_clip_counts(
                output, action_mask, advantages, record_origins[start : start + len(chunk)]
            ).items():
                counts = clip_counts[origin]
                for key, value in values.items():
                    counts[key] += value
        del (
            logits,
            current_log_probs,
            output,
            loss,
            input_ids,
            attention_mask,
            action_mask,
            old_log_probs,
            reference_log_probs,
            advantages,
            token_weights,
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("non-finite full-parameter GRPO gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    grad_value = float(grad_norm.detach().cpu())
    return {
        **totals,
        "policy_clip_fraction": sum(value["clipped"] for value in clip_counts.values())
        / max(1, sum(value["eligible"] for value in clip_counts.values())),
        "policy_clip_eligible_tokens": sum(value["eligible"] for value in clip_counts.values()),
        **{
            f"{origin}_{key}": value
            for origin, counts in clip_counts.items()
            for key, value in {
                "policy_clip_fraction": counts["clipped"] / max(1, counts["eligible"]),
                "policy_clip_tokens": counts["clipped"],
                "policy_clip_eligible_tokens": counts["eligible"],
                "ratio_outside_fraction": counts["outside"] / max(1, counts["tokens"]),
            }.items()
        },
        "turns": len(records),
        "action_tokens": sum(
            int(turn.action_mask.sum()) for episode in episodes for turn in episode.turns
        ),
        "grad_norm_before_clip": grad_value,
        "gradient_clip_norm": config.gradient_clip_norm,
        "gradient_clip_scale": min(1.0, config.gradient_clip_norm / (grad_value + 1e-12)),
        "update_micro_batch_size": micro_batch_size,
        "loss_normalization": "group_equal_episode_equal" if group_equal else "episode_equal",
    }


def update_with_oom_backoff(
    student,
    tokenizer,
    optimizer,
    episodes: list[RolloutEpisode],
    groups: list[GroupResult],
    config: GrpoConfig,
    micro_batch_size: int,
    group_sizes: list[int],
) -> dict[str, Any]:
    attempted = min(micro_batch_size, sum(len(episode.turns) for episode in episodes))
    while attempted >= 1:
        try:
            return _update_once(
                student,
                tokenizer,
                optimizer,
                episodes,
                groups,
                config,
                attempted,
                group_sizes,
            )
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if attempted == 1:
                raise
            attempted = max(1, attempted // 2)
    raise AssertionError("unreachable")


def _save_episodes(
    episodes: list[RolloutEpisode],
    groups: list[GroupResult],
    output_dir: Path,
    group_sizes: list[int],
    task_metadata: list[dict[str, Any]],
    reward_details: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_groups = [
        (group_index, rollout_index)
        for group_index, group_size in enumerate(group_sizes)
        for rollout_index in range(group_size)
    ]
    if len(episode_groups) != len(episodes):
        raise ValueError("episode/group size count mismatch")
    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for index, (episode, (group_index, rollout_index)) in enumerate(
            zip(episodes, episode_groups, strict=True)
        ):
            details = reward_details[index]
            record = {
                "task_id": episode.task_id,
                "db_id": episode.db_id,
                "difficulty": episode.difficulty,
                "group_index": group_index,
                "rollout_index": rollout_index,
                "reward": float(episode.success),
                "advantage": groups[group_index].advantages[rollout_index],
                "group_kind": groups[group_index].kind,
                "group_size": group_sizes[group_index],
                "trajectory_origin": episode.origin,
                "selection_source": task_metadata[group_index]["selection_source"],
                "prior_correct_count": task_metadata[group_index]["prior_correct_count"],
                **details,
                "submitted": episode.submitted,
                "valid_actions": episode.valid_actions,
                "total_actions": episode.total_actions,
                "steps": [
                    {
                        "turn": turn.turn,
                        "response": turn.response_text,
                        "observation": turn.observation,
                        "action_tokens": int(turn.action_mask.sum()),
                    }
                    for turn in episode.turns
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _schema_metrics(
    reward_details: list[dict[str, Any]],
    groups: list[GroupResult],
    group_size: int,
) -> dict[str, Any]:
    scores = [record for record in reward_details if "schema_score" in record]
    if not scores:
        return {}
    rescued_groups = sum(
        bool(scores[index]["schema_rescued_group"]) for index in range(0, len(scores), group_size)
    )
    zero_advantage_groups = sum(
        all(abs(value) <= 1e-12 for value in group.advantages) for group in groups
    )
    return {
        "mean_schema_score": sum(record["schema_score"] for record in scores) / len(scores),
        "mean_schema_phi": sum(record["phi"] for record in scores) / len(scores),
        "mean_table_f1": sum(record["table_f1"] for record in scores) / len(scores),
        "mean_column_f1": sum(record["column_f1"] for record in scores) / len(scores),
        "mean_edge_f1": sum(record["edge_f1"] for record in scores) / len(scores),
        "mean_schema_source_factor": sum(record["source_factor"] for record in scores)
        / len(scores),
        "candidate_sql_coverage": sum(record["candidate_sql_available"] for record in scores)
        / len(scores),
        "candidate_parse_rate": sum(record["candidate_parsed"] for record in scores) / len(scores),
        "identifier_valid_rate": sum(record["identifiers_valid"] for record in scores)
        / len(scores),
        "schema_rescued_all_zero_groups": rescued_groups,
        "schema_rescued_all_zero_group_ratio": rescued_groups
        / max(1, sum(group.kind == "all_zero" for group in groups)),
        "zero_final_advantage_groups": zero_advantage_groups,
        "zero_final_advantage_group_ratio": zero_advantage_groups / len(groups),
        "mean_diagnostic_composite_reward": sum(
            record["diagnostic_composite_reward"] for record in scores
        )
        / len(scores),
    }


def _turn_reward_metrics(
    reward_details: list[dict[str, Any]],
    groups: list[GroupResult],
    group_size: int,
) -> dict[str, Any]:
    records = [record for record in reward_details if "turn_reward" in record]
    if not records:
        return {}
    signal_groups = sum(
        bool(records[index]["turn_efficiency_signal_group"])
        for index in range(0, len(records), group_size)
    )
    correct_records = [record for record in records if record["raw_execution_reward"] == 1.0]
    return {
        "mean_execution_reward": sum(record["raw_execution_reward"] for record in records)
        / len(records),
        "mean_turn_reward_adjustment": sum(record["turn_reward_adjustment"] for record in records)
        / len(records),
        "mean_turn_reward": sum(record["turn_reward"] for record in records) / len(records),
        "mean_correct_turns": (
            sum(record["turns_used"] for record in correct_records) / len(correct_records)
            if correct_records
            else 0.0
        ),
        "turn_efficiency_signal_groups": signal_groups,
        "turn_efficiency_signal_group_ratio": signal_groups / len(groups),
        "zero_final_advantage_groups": sum(
            all(abs(value) <= 1e-12 for value in group.advantages) for group in groups
        ),
    }


def _curriculum_metrics(
    episodes: list[RolloutEpisode],
    groups: list[GroupResult],
    task_metadata: list[dict[str, Any]],
    group_sizes: list[int],
    prior_rollouts_per_prompt: int,
) -> dict[str, Any]:
    current_counts: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    start = 0
    for group_index, group_size in enumerate(group_sizes):
        current = sum(episode.success for episode in episodes[start : start + group_size])
        original = sum(
            episode.success for episode in episodes[start : start + group_size]
            if episode.origin == "on_policy"
        )
        original_counts[f"{original}/{prior_rollouts_per_prompt}"] += 1
        start += group_size
        current_label = f"{current}/{group_size}"
        prior_count = task_metadata[group_index].get("prior_correct_count")
        prior_label = (
            "unmapped" if prior_count is None else f"{prior_count}/{prior_rollouts_per_prompt}"
        )
        current_counts[current_label] += 1
        transitions[f"{prior_label}->{current_label}"] += 1
    return {
        "current_correct_count_histogram": dict(sorted(current_counts.items())),
        "original_student_correct_count_histogram": dict(sorted(original_counts.items())),
        "prior_to_current_transitions": dict(sorted(transitions.items())),
    }


def _resampling_metrics(
    groups: list[GroupResult], group_sizes: list[int], initial_group_size: int
) -> dict[str, Any]:
    if len(groups) != len(group_sizes):
        raise ValueError("group/group size count mismatch")
    supplemented_groups = sum(size > initial_group_size for size in group_sizes)
    rescued_groups = sum(
        size > initial_group_size and group.kind == "mixed"
        for group, size in zip(groups, group_sizes, strict=True)
    )
    histogram = Counter(group_sizes)
    supplemental_episodes = sum(max(0, size - initial_group_size) for size in group_sizes)
    initial_mixed_groups = sum(
        size == initial_group_size and group.kind == "mixed"
        for group, size in zip(groups, group_sizes, strict=True)
    )
    return {
        "initial_mixed_groups": initial_mixed_groups,
        "initial_mixed_group_ratio": initial_mixed_groups / len(groups),
        "supplemented_groups": supplemented_groups,
        "supplemental_episodes": supplemental_episodes,
        "supplemented_to_mixed_groups": rescued_groups,
        "supplement_rescue_ratio": rescued_groups / max(1, supplemented_groups),
        "group_size_histogram": {str(size): count for size, count in sorted(histogram.items())},
        "mean_group_size": sum(group_sizes) / len(group_sizes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded full-parameter SQL-agent GRPO")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    config = GrpoConfig.load(args.config)
    prepared_output_dir = config.output_dir
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if not 1 <= args.max_steps <= config.max_steps:
        raise ValueError(f"--max-steps must be in [1, {config.max_steps}]")
    env_config = EnvConfig.from_yaml(args.env_config)
    completed_before_resume = 0
    student_initialization = config.student_model
    if args.resume_from is not None:
        state_path = args.resume_from / "training_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"resume checkpoint lacks training_state.json: {state_path}")
        resume_state = json.loads(state_path.read_text(encoding="utf-8"))
        completed_before_resume = int(resume_state["completed_step"])
        if not 0 < completed_before_resume < args.max_steps:
            raise ValueError("resume completed_step must be below --max-steps")
        student_initialization = args.resume_from
    pool = _load_pool(prepared_output_dir / "task_pool.jsonl")
    if len(pool) != config.task_count:
        raise ValueError("prepared GRPO pool size differs from config")
    metadata = _load_metadata(prepared_output_dir / "task_metadata.jsonl", pool)
    metrics_path = config.output_dir / "metrics.jsonl"
    if metrics_path.exists():
        if args.resume_from is None:
            raise FileExistsError(f"refusing to append to existing GRPO run: {metrics_path}")
        metric_rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not metric_rows or int(metric_rows[-1]["step"]) != completed_before_resume:
            raise ValueError("metrics do not end at the requested resume checkpoint")
    elif args.resume_from is not None:
        raise FileNotFoundError("resume requires the original metrics.jsonl")
    schema_v2 = config.reward_mode == "schema_hierarchical_v2"
    schema_rewarder = (
        SchemaRewarder(
            env_config.spider_root,
            config.schema_component_weights,
            single_table_weights=config.schema_single_table_weights if schema_v2 else None,
            multi_table_weights=config.schema_multi_table_weights if schema_v2 else None,
            execute_sql_factor=config.execute_sql_schema_factor if schema_v2 else 1.0,
        )
        if config.reward_mode in {"schema_hierarchical", "schema_hierarchical_v2"}
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(config.student_model)
    student = AutoModelForCausalLM.from_pretrained(
        student_initialization, dtype=torch.float32, device_map={"": "cuda:0"}
    )
    student.requires_grad_(True)
    if config.kl_beta > 0:
        reference = AutoModelForCausalLM.from_pretrained(
            config.reference_model, dtype=torch.bfloat16, device_map={"": "cuda:0"}
        )
        reference.requires_grad_(False)
        reference.eval()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    current_micro_batch = config.update_micro_batch_size
    started = time.monotonic()
    final_checkpoint: Path | None = None
    for step in range(completed_before_resume + 1, args.max_steps + 1):
        step_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        tasks = _step_tasks(pool, step, config.prompts_per_step)
        step_metadata = [metadata[task.task_id] for task in tasks]
        episodes, groups, reward_details, group_sizes = collect_groups(
            student,
            tokenizer,
            tasks,
            env_config,
            config,
            schema_rewarder,
        )
        if config.kl_beta > 0:
            score_turns_with_teacher(
                reference,
                tokenizer,
                episodes,
                micro_batch_size=config.reference_score_micro_batch_size,
            )
        step_dir = config.output_dir / f"step_{step:04d}"
        _save_episodes(
            episodes,
            groups,
            step_dir,
            group_sizes,
            step_metadata,
            reward_details,
        )
        update = update_with_oom_backoff(
            student,
            tokenizer,
            optimizer,
            episodes,
            groups,
            config,
            current_micro_batch,
            group_sizes,
        )
        group_metrics = summarize_groups(groups)
        curriculum_metrics = _curriculum_metrics(
            episodes,
            groups,
            step_metadata,
            group_sizes,
            config.rollouts_per_prompt,
        )
        elapsed = time.monotonic() - started
        completed_this_process = step - completed_before_resume
        eta = elapsed / completed_this_process * (args.max_steps - step)
        metric = {
            "step": step,
            "total_steps": args.max_steps,
            "progress_percent": 100.0 * step / args.max_steps,
            "episodes": len(episodes),
            "submitted": sum(episode.submitted for episode in episodes),
            "valid_action_rate": sum(episode.valid_actions for episode in episodes)
            / max(1, sum(episode.total_actions for episode in episodes)),
            **group_metrics,
            **curriculum_metrics,
            **_resampling_metrics(groups, group_sizes, config.rollouts_per_prompt),
            "original_student_success_rate": sum(
                episode.success for episode in episodes if episode.origin == "on_policy"
            ) / max(1, sum(episode.origin == "on_policy" for episode in episodes)),
            "gold_injected_groups": sum(
                episode.origin == "gold_guided" for episode in episodes
            ),
            "gold_injection_attempts": sum(
                detail.get("gold_injection_attempts", 0)
                for detail in reward_details if detail.get("trajectory_origin") == "on_policy"
            ) / config.rollouts_per_prompt if config.gold_injection else 0,
            "original_all_zero_groups": sum(
                detail.get("original_student_correct_count") == 0
                for detail in reward_details if detail.get("trajectory_origin") == "on_policy"
            ) / config.rollouts_per_prompt if config.interaction_protocol == "reasoning_tool" else None,
            **_schema_metrics(reward_details, groups, config.rollouts_per_prompt),
            **update,
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            **_turn_reward_metrics(reward_details, groups, config.rollouts_per_prompt),
            "step_seconds": time.monotonic() - step_started,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "estimated_finish_utc": (datetime.now(UTC) + timedelta(seconds=eta)).isoformat(
                timespec="seconds"
            ),
        }
        should_save = not args.no_save and (
            step == args.max_steps
            or (config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0)
        )
        if should_save:
            final_checkpoint = config.output_dir / f"checkpoint_step_{step:04d}"
            student.save_pretrained(final_checkpoint, safe_serialization=True, max_shard_size="2GB")
            tokenizer.save_pretrained(final_checkpoint)
            (final_checkpoint / "training_state.json").write_text(
                json.dumps(
                    {
                        "completed_step": step,
                        "total_steps": args.max_steps,
                        "student_checkpoint": str(final_checkpoint),
                        "reference_model": str(config.reference_model),
                        "optimizer_resume": "fresh_adamw",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")
        (step_dir / "metrics.json").write_text(
            json.dumps(metric, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metric, sort_keys=True), flush=True)
        requested = current_micro_batch
        used = int(update["update_micro_batch_size"])
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        fraction = metric["peak_gpu_reserved_gib"] / total_gib
        if used < requested:
            current_micro_batch = used
        elif fraction < config.target_gpu_memory_utilization - 0.1:
            current_micro_batch = min(used * 2, update["turns"])
        else:
            current_micro_batch = used
    summary = {
        "status": "completed",
        "completed_steps": args.max_steps,
        "checkpoint": str(final_checkpoint) if final_checkpoint else None,
        "student_initialization": str(config.student_model),
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "reference_model": str(config.reference_model),
        "reward": config.reward_mode,
        "schema_advantage_coefficient": config.schema_advantage_coefficient,
        "submission_advantage_coefficient": config.submission_advantage_coefficient,
        "schema_component_weights": config.schema_component_weights,
        "schema_single_table_weights": config.schema_single_table_weights,
        "schema_multi_table_weights": config.schema_multi_table_weights,
        "execute_sql_schema_factor": config.execute_sql_schema_factor,
        "homogeneous_resampling_max_rollouts": config.homogeneous_resampling_max_rollouts,
        "interaction_protocol": config.interaction_protocol,
        "gold_injection": config.gold_injection,
        "student_update": "full",
        "formal_training": True,
        "turn_reward_step": config.turn_reward_step,
        "turn_reward_baselines": config.turn_reward_baselines,
        "max_assistant_turns_by_difficulty": config.max_assistant_turns_by_difficulty,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
