from __future__ import annotations

from typing import Any

from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.models import TaskRecord
from sql_agent.verifier import ExecutionVerifier


def replay_record(
    record: dict[str, Any], task: TaskRecord, config: EnvConfig
) -> dict[str, Any]:
    if record["task_id"] != task.task_id:
        raise ValueError("Log record and task ID do not match")
    if record["mode"] != "multi_turn_agent":
        sql = record.get("final_sql") or ""
        actual = ExecutionVerifier(
            task.resolve_db_path(config.spider_root), task.reference_sql, config
        ).verify(sql)
        return {"matches": actual.correct == record["success"], "success": actual.correct}

    env = SQLAgentEnv(config)
    observation: dict[str, Any] = {}
    try:
        env.reset(task)
        for step in record["steps"]:
            observation, done = env.step(step["response"])
            if observation != step["observation"]:
                return {"matches": False, "success": False, "reason": "observation_mismatch"}
            if done:
                break
    finally:
        env.close()
    success = observation.get("reward") == 1.0
    return {"matches": success == record["success"], "success": success}
