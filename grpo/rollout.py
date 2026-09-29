from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from evaluation.runner import extract_sql
from grpo.config import GrpoConfig
from grpo.scoring import causal_selected_log_probs, causal_selected_log_probs_chunked
from sql_agent.action_parser import ActionParseError, parse_action
from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.models import TaskRecord
from sql_agent.prompts import render_agent_messages


@dataclass(slots=True)
class RolloutTurn:
    task_id: str
    turn: int
    input_ids: Tensor
    prompt_length: int
    action_mask: Tensor
    old_log_probs: Tensor
    response_text: str
    observation: dict[str, Any]
    teacher_log_probs: Tensor | None = None

    def validate(self, *, require_teacher: bool = False) -> None:
        if self.input_ids.ndim != 1 or self.input_ids.numel() < 2:
            raise ValueError("turn input_ids must be a nontrivial 1D sequence")
        target_length = self.input_ids.numel() - 1
        if self.action_mask.shape != (target_length,):
            raise ValueError("turn action_mask is not aligned to causal targets")
        if self.old_log_probs.shape != (target_length,):
            raise ValueError("turn old_log_probs is not aligned to causal targets")
        if not 1 <= self.prompt_length < self.input_ids.numel():
            raise ValueError("turn prompt_length is invalid")
        if self.action_mask[: self.prompt_length - 1].any():
            raise ValueError("turn action_mask includes prompt tokens")
        if not self.action_mask.any():
            raise ValueError("turn contains no trainable action token")
        if not torch.isfinite(self.old_log_probs[self.action_mask]).all():
            raise ValueError("turn old_log_probs contains non-finite action values")
        if require_teacher and self.teacher_log_probs is None:
            raise ValueError("turn is missing teacher_log_probs")
        if self.teacher_log_probs is not None:
            if self.teacher_log_probs.shape != (target_length,):
                raise ValueError("turn teacher_log_probs is not aligned to causal targets")
            if not torch.isfinite(self.teacher_log_probs[self.action_mask]).all():
                raise ValueError("turn teacher_log_probs contains non-finite action values")


@dataclass(slots=True)
class RolloutEpisode:
    task_id: str
    db_id: str
    difficulty: str
    turns: list[RolloutTurn] = field(default_factory=list)
    submitted: bool = False
    success: bool = False
    valid_actions: int = 0
    total_actions: int = 0

    def validate(self, *, require_teacher: bool = False) -> None:
        if not self.turns:
            raise ValueError("rollout episode has no turns")
        for expected_turn, turn in enumerate(self.turns, start=1):
            if turn.task_id != self.task_id or turn.turn != expected_turn:
                raise ValueError("rollout turn identity/order mismatch")
            turn.validate(require_teacher=require_teacher)


def _tokenize_prompt(tokenizer, messages: list[dict[str, str]], device: str) -> Tensor:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)


def rollout_task(
    model,
    tokenizer,
    task: TaskRecord,
    env_config: EnvConfig,
    grpo_config: GrpoConfig,
    *,
    device: str = "cuda",
    rollout_index: int = 0,
) -> RolloutEpisode:
    env = SQLAgentEnv(env_config)
    episode = RolloutEpisode(task.task_id, task.db_id, task.difficulty)
    special_ids = set(tokenizer.all_special_ids)
    try:
        env.reset(task)
        while not env.done and len(episode.turns) < grpo_config.max_assistant_turns:
            messages = render_agent_messages(env.history)
            prompt_ids = _tokenize_prompt(tokenizer, messages, device)
            if prompt_ids.shape[1] + grpo_config.max_action_tokens > grpo_config.max_sequence_tokens:
                raise ValueError(f"task {task.task_id} prompt exceeds GRPO sequence budget")
            task_seed = int.from_bytes(
                hashlib.sha256(task.task_id.encode()).digest()[:4], "big"
            )
            generation_seed = (
                grpo_config.seed
                + task_seed
                + len(episode.turns)
                + rollout_index * 1_000_003
            )
            prompt_mask = torch.ones_like(prompt_ids)
            model.eval()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(generation_seed)
                torch.cuda.manual_seed_all(generation_seed)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sequence = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        do_sample=True,
                        temperature=grpo_config.temperature,
                        top_p=grpo_config.top_p,
                        top_k=grpo_config.top_k,
                        max_new_tokens=grpo_config.max_action_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                attention_mask = torch.ones_like(sequence)
                logits = model(
                    input_ids=sequence, attention_mask=attention_mask, use_cache=False
                ).logits
                old_log_probs = causal_selected_log_probs(logits, sequence)[0]
            prompt_length = prompt_ids.shape[1]
            generated_ids = sequence[0, prompt_length:]
            action_mask = torch.zeros_like(old_log_probs, dtype=torch.bool)
            action_mask[prompt_length - 1 :] = True
            targets = sequence[0, 1:]
            for token_id in special_ids:
                action_mask &= targets.ne(token_id)
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not response_text:
                response_text = extract_sql(tokenizer.decode(generated_ids).strip())
            action = None
            try:
                action = parse_action(response_text)
                episode.valid_actions += 1
            except ActionParseError:
                pass
            episode.total_actions += 1
            observation, done = env.step(response_text)
            if done and "verification" in observation:
                episode.submitted = True
                episode.success = bool(observation.get("verification", {}).get("correct", False))
            turn = RolloutTurn(
                task_id=task.task_id,
                turn=len(episode.turns) + 1,
                input_ids=sequence[0].detach().cpu(),
                prompt_length=prompt_length,
                action_mask=action_mask.detach().cpu(),
                old_log_probs=old_log_probs.detach().cpu(),
                response_text=response_text,
                observation=observation,
            )
            turn.validate()
            episode.turns.append(turn)
            del logits, old_log_probs, sequence
            if done:
                break
    finally:
        env.close()
    episode.validate()
    return episode


def _collate_turns(turns: list[RolloutTurn], pad_token_id: int, device: str):
    max_length = max(turn.input_ids.numel() for turn in turns)
    input_ids = torch.full((len(turns), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    action_mask = torch.zeros((len(turns), max_length - 1), dtype=torch.bool)
    for index, turn in enumerate(turns):
        length = turn.input_ids.numel()
        input_ids[index, :length] = turn.input_ids
        attention_mask[index, :length] = 1
        action_mask[index, : length - 1] = turn.action_mask
    return input_ids.to(device), attention_mask.to(device), action_mask.to(device)


def collate_training_turns(turns: list[RolloutTurn], pad_token_id: int, device: str):
    input_ids, attention_mask, action_mask = _collate_turns(turns, pad_token_id, device)
    old_log_probs = torch.zeros_like(action_mask, dtype=torch.float32)
    teacher_log_probs = torch.zeros_like(action_mask, dtype=torch.float32)
    for index, turn in enumerate(turns):
        turn.validate(require_teacher=True)
        length = turn.input_ids.numel() - 1
        old_log_probs[index, :length] = turn.old_log_probs.to(device)
        teacher_log_probs[index, :length] = turn.teacher_log_probs.to(device)
    return input_ids, attention_mask, action_mask, old_log_probs, teacher_log_probs


def score_turns_with_teacher(
    teacher,
    tokenizer,
    episodes: list[RolloutEpisode],
    *,
    micro_batch_size: int,
    device: str = "cuda",
) -> None:
    turns = [turn for episode in episodes for turn in episode.turns]
    teacher.eval()
    start = 0
    active_micro_batch = min(micro_batch_size, len(turns))
    while start < len(turns):
        chunk = turns[start : start + active_micro_batch]
        input_ids = attention_mask = action_mask = logits = log_probs = None
        try:
            input_ids, attention_mask, action_mask = _collate_turns(
                chunk, tokenizer.pad_token_id, device
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = teacher(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                ).logits
            log_probs = causal_selected_log_probs_chunked(logits, input_ids)
        except torch.OutOfMemoryError:
            del input_ids, attention_mask, action_mask, logits, log_probs
            torch.cuda.empty_cache()
            if active_micro_batch == 1:
                raise
            active_micro_batch = max(1, active_micro_batch // 2)
            continue
        for index, turn in enumerate(chunk):
            length = turn.input_ids.numel() - 1
            if not torch.equal(action_mask[index, :length].cpu(), turn.action_mask):
                raise ValueError("teacher scoring changed the action mask")
            turn.teacher_log_probs = log_probs[index, :length].detach().cpu()
            turn.validate(require_teacher=True)
        del logits, log_probs, input_ids, attention_mask, action_mask
        start += len(chunk)
    for episode in episodes:
        episode.validate(require_teacher=True)
