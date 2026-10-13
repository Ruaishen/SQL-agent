"""Method 2 on the existing native GRPO trainer; no veRL dependency."""

import json
import os
from types import SimpleNamespace

from sft.curate_reasoning import EXTERNAL_REFERENCE
from sft.regenerate_reasoning import PROMPTS, trajectory_comparison, trajectory_profile
from sql_agent.deepseek import DeepSeekClient
from sql_agent.protocol import build_prompt, parse_tagged_response
from sql_agent.truncation import canonical_json


def public_messages(task, env_config, episode):
    messages = [
        {"role": "system", "content": build_prompt(env_config.max_turns)},
        {"role": "user", "content": task.question},
    ]
    for index, turn in enumerate(episode.turns):
        messages.append({"role": "assistant", "content": turn.response_text})
        if index < len(episode.turns) - 1:
            messages.append(
                {
                    "role": "user",
                    "content": f"<observation>{canonical_json(turn.observation)}</observation>",
                }
            )
    return messages


def audit_trajectory(task, messages, comparison, client, *, max_tokens):
    for message in messages:
        if message["role"] == "assistant":
            reasoning, _ = parse_tagged_response(message["content"])
            if EXTERNAL_REFERENCE.search(reasoning):
                return "reject", None
    completion = client.complete_thinking_text(
        [
            {
                "role": "system",
                "content": (PROMPTS / "external_answer_leak_audit_zh.md").read_text(
                    encoding="utf-8"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": task.question,
                        "messages": messages,
                        "trajectory_comparison": comparison,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        max_tokens=max_tokens,
    )
    verdict = completion.message.get("content")
    if completion.finish_reason != "stop" or verdict not in {"accept", "reject"}:
        verdict = "reject"
    return verdict, {
        "message": completion.message,
        "finish_reason": completion.finish_reason,
        "usage": completion.usage,
        "model": completion.model,
    }


def inject_guided_trajectory(
    model, tokenizer, task, env_config, config, originals, *, client=None, device="cuda"
):
    from grpo.tagged_rollout import rollout_tagged_task

    if not config.method2_guided_injection or task.split != "train" or not originals:
        raise ValueError("method 2 requires a nonempty failed training group")
    if any(episode.success for episode in originals):
        raise ValueError("method 2 only runs for all-wrong groups")
    if (
        not isinstance(config.method2_max_attempts, int)
        or not 1 <= config.method2_max_attempts <= 3
    ):
        raise ValueError("method 2 allows at most three generation attempts")
    if config.temperature != 1 or config.top_p != 1 or config.top_k not in (None, 0):
        raise ValueError("method 2 source scoring requires full-support temperature=1 sampling")
    counter = SimpleNamespace(
        count_text=lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    )
    original_profile = trajectory_profile(public_messages(task, env_config, originals[0]), counter)
    guidance = (
        (PROMPTS / "gold_repair_supplement_zh.md")
        .read_text(encoding="utf-8")
        .replace("{gold_sql}", task.reference_sql)
        .replace("{original_profile}", json.dumps(original_profile, ensure_ascii=False))
    )
    if client is None:
        client = DeepSeekClient(
            os.environ.get(config.method2_audit_key_env, ""),
            model=config.method2_audit_model,
            base_url=config.method2_audit_base_url,
        )
    info = {
        "triggered": True,
        "attempts": 0,
        "injected": False,
        "attempt_records": [],
        "private_guidance": guidance,
    }
    accepted = None
    for attempt in range(config.method2_max_attempts):
        info["attempts"] += 1
        record = {"attempt": attempt + 1}
        try:
            candidate = rollout_tagged_task(
                model,
                tokenizer,
                task,
                env_config,
                config,
                device=device,
                rollout_index=len(originals) + attempt,
                guidance=guidance,
            )
            record["execution_correct"] = candidate.success
            if candidate.success:
                candidate.validate()
                messages = public_messages(task, env_config, candidate)
                comparison = trajectory_comparison(
                    original_profile, trajectory_profile(messages, counter)
                )
                verdict, audit = audit_trajectory(
                    task, messages, comparison, client, max_tokens=config.method2_audit_max_tokens
                )
                record.update(
                    leak_audit=verdict, trajectory_comparison=comparison, private_audit_call=audit
                )
                if verdict == "accept":
                    accepted = candidate
                    info["injected"] = True
        except (ValueError, RuntimeError) as error:
            record["error"] = str(error)
        info["attempt_records"].append(record)
        if accepted is not None:
            break
    for episode in originals + ([accepted] if accepted is not None else []):
        episode.method2_info = info
    return accepted
