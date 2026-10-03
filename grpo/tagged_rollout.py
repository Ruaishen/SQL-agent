"""On-policy rollouts for binary execution GRPO with the Spider CoT protocol."""

from __future__ import annotations

import hashlib

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from grpo.config import GrpoConfig
from grpo.rollout import RolloutEpisode, RolloutTurn
from grpo.scoring import causal_selected_log_probs
from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.models import TaskRecord
from sql_agent.truncation import canonical_json
from sql_planner.collect_tagged import build_prompt, parse_response


class _OneToolStop(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        return "</tool>" in self.tokenizer.decode(input_ids[0, -24:], skip_special_tokens=True)


def _prompt_ids(tokenizer, messages: list[dict[str, str]], device: str) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
    ).to(device)


def rollout_tagged_task(
    model,
    tokenizer,
    task: TaskRecord,
    env_config: EnvConfig,
    config: GrpoConfig,
    *,
    device: str = "cuda",
    rollout_index: int = 0,
) -> RolloutEpisode:
    """Sample and score assistant actions under the same ordinary policy context."""
    if task.split != "train":
        raise ValueError("GRPO rollouts require a training task")
    clean_system = build_prompt(env_config.max_turns)
    env = SQLAgentEnv(
        env_config, reserve_final_submission=True, system_prompt=clean_system
    )
    episode = RolloutEpisode(task.task_id, task.db_id, task.difficulty)
    clean_messages: list[dict[str, str]] = [
        {"role": "system", "content": clean_system},
        {"role": "user", "content": task.question},
    ]
    special_ids = set(tokenizer.all_special_ids)
    task_seed = int.from_bytes(hashlib.sha256(task.task_id.encode()).digest()[:4], "big")
    try:
        env.reset(task)
        while not env.done and len(episode.turns) <= config.max_assistant_turns:
            clean_ids = _prompt_ids(tokenizer, clean_messages, device)
            if clean_ids.shape[1] + config.max_action_tokens > config.max_sequence_tokens:
                if not episode.turns:
                    raise ValueError(f"task {task.task_id} prompt exceeds GRPO sequence budget")
                break
            generation_seed = (
                config.seed + task_seed + len(episode.turns) + rollout_index * 1_000_003
            )
            model.eval()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
                torch.manual_seed(generation_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(generation_seed)
                with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                    generated = model.generate(
                        input_ids=clean_ids,
                        attention_mask=torch.ones_like(clean_ids),
                        do_sample=True,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=0 if config.top_k is None else config.top_k,
                        max_new_tokens=config.max_action_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([_OneToolStop(tokenizer)]),
                    )
            continuation = generated[0, clean_ids.shape[1]:]
            sequence = generated
            prompt_length = clean_ids.shape[1]
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                logits = model(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False).logits
                old_log_probs = causal_selected_log_probs(logits, sequence)[0]
            action_mask = torch.zeros_like(old_log_probs, dtype=torch.bool)
            action_mask[prompt_length - 1:] = True
            for token_id in special_ids:
                action_mask &= sequence[0, 1:].ne(token_id)
            if not action_mask.any():
                break
            response = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            episode.total_actions += 1
            try:
                _, action = parse_response(response)
                if env.turn >= env_config.max_turns and action["tool"] != "submit_sql":
                    raise ValueError("exploration budget exhausted")
                episode.valid_actions += 1
                if action["tool"] == "submit_sql":
                    observation, _ = env.submit_sql(action["arguments"])
                    episode.submitted = "verification" in observation
                    episode.success = bool(observation.get("verification", {}).get("correct", False))
                else:
                    observation, _ = env.step(action)
            except ValueError as exc:
                observation = {"status": "invalid_format", "message": str(exc)}
                episode.turns.append(RolloutTurn(
                    task_id=task.task_id, turn=len(episode.turns) + 1,
                    input_ids=sequence[0].detach().cpu(), prompt_length=prompt_length,
                    action_mask=action_mask.detach().cpu(),
                    old_log_probs=old_log_probs.detach().cpu(),
                    response_text=response, observation=observation,
                ))
                break
            episode.turns.append(RolloutTurn(
                task_id=task.task_id, turn=len(episode.turns) + 1,
                input_ids=sequence[0].detach().cpu(), prompt_length=prompt_length,
                action_mask=action_mask.detach().cpu(),
                old_log_probs=old_log_probs.detach().cpu(),
                response_text=response, observation=observation,
            ))
            if env.done:
                break
            observation_message = {
                "role": "user", "content": f"<observation>{canonical_json(observation)}</observation>"
            }
            clean_messages.extend(({"role": "assistant", "content": response}, observation_message))
        if episode.turns:
            episode.validate()
        return episode
    finally:
        env.close()
