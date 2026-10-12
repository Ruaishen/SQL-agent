import json
from dataclasses import replace

import pytest

from sft.curate_reasoning import validate_record
from sql_agent.env import SQLAgentEnv
from sql_agent.protocol import build_prompt
from sql_agent.truncation import canonical_json


def record_for(task, config, sql):
    messages = [{"role": "system", "content": build_prompt(10)}, {"role": "user", "content": task.question}]
    turns = []
    env = SQLAgentEnv(config, reserve_final_submission=True, system_prompt=messages[0]["content"])
    try:
        env.reset(task)
        for tool, arguments in (("list_tables", {}), ("execute_sql", {"sql": "SELECT 0"}), ("submit_sql", {"sql": sql})):
            response = '<reasoning>Check the database result.</reasoning><tool>' + json.dumps({"name": tool, "arguments": arguments}) + '</tool>'
            observation, _ = env.submit_sql(arguments) if tool == "submit_sql" else env.step({"tool": tool, "arguments": arguments})
            messages.append({"role": "assistant", "content": response})
            turns.append({"tool": tool, "arguments": arguments, "response": response, "observation": observation})
            if tool != "submit_sql":
                messages.append({"role": "user", "content": f"<observation>{canonical_json(observation)}</observation>"})
    finally:
        env.close()
    return {"question": task.question, "db_id": task.db_id, "correct": True, "generation_mode": "full_regeneration_from_question",
            "leak_audit": "accept", "sft_eligible": True, "student_messages": messages, "turns": turns, "final_sql": sql}


def test_recovery_validates_final_result_without_exact_gold_test(sample_db, config, task):
    task = replace(task, split="train")
    record = record_for(task, config, "SELECT COUNT(*) FROM employees;")
    with pytest.raises(ValueError):
        validate_record(record, task, config, regenerated=True)
    assert validate_record(record, task, config, regenerated=True, strict_submission=False)["correct"]


def test_recovery_still_rejects_wrong_results_and_fabricated_observations(sample_db, config, task):
    task = replace(task, split="train")
    wrong = record_for(task, config, "SELECT 0")
    with pytest.raises(ValueError, match="Final SQL failed"):
        validate_record(wrong, task, config, regenerated=True, strict_submission=False)
    fabricated = record_for(task, config, task.reference_sql)
    fabricated["turns"][0]["observation"]["status"] = "fabricated"
    with pytest.raises(ValueError, match="Stored observation differs"):
        validate_record(fabricated, task, config, regenerated=True, strict_submission=False)
