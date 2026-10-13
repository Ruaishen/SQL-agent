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
from sql_agent.protocol import build_prompt
from sql_agent.protocol import parse_tagged_response as parse_response
from sql_agent.truncation import canonical_json


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
    guidance: str | None = None,
) -> RolloutEpisode:
    """Sample and score assistant actions under the same ordinary policy context."""
    if task.split != "train":
        raise ValueError("GRPO rollouts require a training task")
    clean_system = build_prompt(env_config.max_turns)
    env = SQLAgentEnv(env_config, reserve_final_submission=True, system_prompt=clean_system)
    episode = RolloutEpisode(task.task_id, task.db_id, task.difficulty)
    if guidance is not None:
        if not config.method2_guided_injection:
            raise ValueError("guided rollout requires method2_guided_injection")
        episode.origin = "guided_old_student"
    clean_messages: list[dict[str, str]] = [
        {"role": "system", "content": clean_system},
        {"role": "user", "content": task.question},
    ]
    special_ids = set(tokenizer.all_special_ids)
    source_messages = (
        [dict(clean_messages[0]), {"role": "user", "content": guidance}, dict(clean_messages[1])]
        if guidance is not None
        else clean_messages
    )
    tested_gold = False
    task_seed = int.from_bytes(hashlib.sha256(task.task_id.encode()).digest()[:4], "big")
    try:
        env.reset(task)
        if guidance is not None and not env.verify(task.reference_sql).correct:
            raise ValueError("Gold SQL failed preflight")
        while not env.done and len(episode.turns) <= config.max_assistant_turns:
            clean_ids = _prompt_ids(tokenizer, clean_messages, device)
            source_ids = (
                _prompt_ids(tokenizer, source_messages, device)
                if guidance is not None
                else clean_ids
            )
            if (
                max(clean_ids.shape[1], source_ids.shape[1]) + config.max_action_tokens
                > config.max_sequence_tokens
            ):
                if not episode.turns:
                    raise ValueError(f"task {task.task_id} prompt exceeds GRPO sequence budget")
                break
            generation_seed = (
                config.seed + task_seed + len(episode.turns) + rollout_index * 1_000_003
            )
            model.eval()
            with torch.random.fork_rng(
                devices=[torch.cuda.current_device()] if device == "cuda" else []
            ):
                torch.manual_seed(generation_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(generation_seed)
                with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                    generated = model.generate(
                        input_ids=source_ids,
                        attention_mask=torch.ones_like(source_ids),
                        do_sample=True,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=0 if config.top_k is None else config.top_k,
                        max_new_tokens=config.max_action_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([_OneToolStop(tokenizer)]),
                        **(
                            {
                                "repetition_penalty": 1.0,
                                "no_repeat_ngram_size": 0,
                                "num_beams": 1,
                                "typical_p": 1.0,
                                "min_p": None,
                                "min_length": 0,
                                "min_new_tokens": 0,
                                "forced_bos_token_id": None,
                                "forced_eos_token_id": None,
                                "bad_words_ids": None,
                                "suppress_tokens": None,
                                "begin_suppress_tokens": None,
                                "sequence_bias": None,
                            }
                            if config.method2_guided_injection
                            else {}
                        ),
                    )
            continuation = generated[0, source_ids.shape[1] :]
            sequence = torch.cat((clean_ids, continuation.unsqueeze(0)), dim=1)
            prompt_length = clean_ids.shape[1]
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                logits = model(
                    input_ids=generated, attention_mask=torch.ones_like(generated), use_cache=False
                ).logits
                source_log_probs = causal_selected_log_probs(logits, generated)[0]
                if guidance is None:
                    old_log_probs = source_log_probs
                else:
                    old_log_probs = torch.zeros(sequence.shape[1] - 1, device=sequence.device)
                    old_log_probs[prompt_length - 1 :] = source_log_probs[source_ids.shape[1] - 1 :]
            action_mask = torch.zeros_like(old_log_probs, dtype=torch.bool)
            action_mask[prompt_length - 1 :] = True
            for token_id in special_ids:
                action_mask &= sequence[0, 1:].ne(token_id)
            if not action_mask.any():
                break
            response = tokenizer.decode(continuation, skip_special_tokens=True)
            if guidance is None:
                response = response.strip()
            episode.total_actions += 1
            try:
                _, action = parse_response(response)
                if env.turn >= env_config.max_turns and action["tool"] != "submit_sql":
                    raise ValueError("exploration budget exhausted")
                if guidance is not None:
                    if not episode.turns and action["tool"] != "list_tables":
                        raise ValueError("Full regeneration must start with list_tables")
                    if tested_gold and action["tool"] != "submit_sql":
                        raise ValueError(
                            "Successful exact Gold execution must be followed by submission"
                        )
                    if action["tool"] == "submit_sql" and (
                        not tested_gold or action["arguments"].get("sql") != task.reference_sql
                    ):
                        raise ValueError("Submit the exact previously tested Gold SQL")
                    if (
                        env.turn == env_config.max_turns - 1
                        and action["tool"] != "submit_sql"
                        and (
                            action["tool"] != "execute_sql"
                            or action["arguments"].get("sql") != task.reference_sql
                        )
                    ):
                        raise ValueError("Reserve final exploration for exact Gold execution")
                episode.valid_actions += 1
                if action["tool"] == "submit_sql":
                    observation, _ = env.submit_sql(action["arguments"])
                    episode.submitted = "verification" in observation
                    episode.success = bool(
                        observation.get("verification", {}).get("correct", False)
                    )
                else:
                    observation, _ = env.step(action)
                    if guidance is not None:
                        tested_gold = (
                            action["tool"] == "execute_sql"
                            and action["arguments"].get("sql") == task.reference_sql
                            and observation.get("status") == "success"
                        )
            except ValueError as exc:
                observation = {"status": "invalid_format", "message": str(exc)}
                episode.turns.append(
                    RolloutTurn(
                        task_id=task.task_id,
                        turn=len(episode.turns) + 1,
                        input_ids=sequence[0].detach().cpu(),
                        prompt_length=prompt_length,
                        action_mask=action_mask.detach().cpu(),
                        old_log_probs=old_log_probs.detach().cpu(),
                        response_text=response,
                        observation=observation,
                        source_prompt_ids=source_ids[0].detach().cpu()
                        if guidance is not None
                        else None,
                    )
                )
                break
            episode.turns.append(
                RolloutTurn(
                    task_id=task.task_id,
                    turn=len(episode.turns) + 1,
                    input_ids=sequence[0].detach().cpu(),
                    prompt_length=prompt_length,
                    action_mask=action_mask.detach().cpu(),
                    old_log_probs=old_log_probs.detach().cpu(),
                    response_text=response,
                    observation=observation,
                    source_prompt_ids=source_ids[0].detach().cpu()
                    if guidance is not None
                    else None,
                )
            )
            if env.done:
                break
            observation_message = {
                "role": "user",
                "content": f"<observation>{canonical_json(observation)}</observation>",
            }
            clean_messages.extend(({"role": "assistant", "content": response}, observation_message))
            if guidance is not None:
                source_messages.extend(
                    ({"role": "assistant", "content": response}, dict(observation_message))
                )
        if episode.turns:
            episode.validate()
        return episode
    finally:
        env.close()
