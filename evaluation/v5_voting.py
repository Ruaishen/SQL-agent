from __future__ import annotations

import json
import time
from typing import Any

from evaluation.tool_xml import RESPONSE_PATTERN, ToolCall
from sql_agent.action_parser import ActionParseError, parse_action
from sql_agent.config import EnvConfig
from sql_agent.model_adapter import ModelAdapter, ModelAdapterError
from sql_agent.models import TaskRecord
from sql_agent.sandbox import SQLSandbox, SQLSandboxError, connect_readonly
from sql_agent.tools import SQLTools
from sql_agent.truncation import ObservationLimiter, TokenCounter, canonical_json
from sql_agent.verifier import ExecutionVerifier


def parse_v5_response(response: str) -> ToolCall:
    match = RESPONSE_PATTERN.fullmatch(response)
    if match is None or not match.group(1).strip():
        raise ActionParseError("invalid_format", "Use <reasoning>...</reasoning><tool>...</tool>")
    try:
        value = json.loads(match.group(2))
    except json.JSONDecodeError as exc:
        raise ActionParseError("invalid_json", "Tool block must contain valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        raise ActionParseError("invalid_action", "Tool JSON needs only name and arguments")
    if value["name"] == "submit_sql":
        arguments = value["arguments"]
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"sql"}
            or not isinstance(arguments["sql"], str)
            or not arguments["sql"].strip()
        ):
            raise ActionParseError("invalid_arguments", "submit_sql requires a nonempty sql string")
        return ToolCall("submit_sql", arguments)
    action = parse_action({"tool": value["name"], "arguments": value["arguments"]})
    if action.tool not in {"list_tables", "inspect_tables", "inspect_values", "execute_sql"}:
        raise ActionParseError("invalid_action", "This tool is not available")
    return ToolCall(action.tool, action.arguments.model_dump())


class V5VotingEvaluator:
    def __init__(self, config: EnvConfig, model: ModelAdapter, *, max_tokens: int = 512):
        config.validate()
        self.config = config
        self.model = model
        self.max_tokens = max_tokens
        self.limiter = ObservationLimiter(config, TokenCounter(config.tokenizer_path))

    @staticmethod
    def _legacy_error(exc: SQLSandboxError) -> dict[str, str]:
        message = (
            "Query references an unknown column"
            if exc.error_type == "unknown_column"
            else exc.message
        )
        return {"status": "error", "error_type": exc.error_type, "message": message}

    @staticmethod
    def _execute(tools: SQLTools, action: ToolCall) -> dict[str, Any]:
        if action.name == "execute_sql":
            return tools.execute_sql(action.arguments["sql"])
        if action.name == "list_tables":
            return tools.list_tables()
        if action.name == "inspect_tables":
            return tools.inspect_tables(action.arguments["table_names"])
        if action.name == "inspect_values":
            return tools.inspect_values(
                action.arguments["table_name"], action.arguments["column_name"]
            )
        raise ValueError(f"Unexpected tool: {action.name}")

    def evaluate(
        self,
        task: TaskRecord,
        first_messages: list[dict[str, str]],
        initial_sql: str,
        sample_index: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        db_path = task.resolve_db_path(self.config.spider_root)
        messages = [dict(message) for message in first_messages]
        first_user = messages[1]["content"]
        suffix = f"\nExploratory calls remaining: {self.config.max_turns}"
        if not first_user.endswith(suffix):
            raise ValueError("Source trajectory has an unexpected first user message")
        context = first_user[: -len(suffix)]
        turns: list[dict[str, Any]] = []
        remaining = self.config.max_turns
        final_sql: str | None = None
        last_sql = initial_sql
        last_successful_sql: str | None = None
        status = "missing_submission"
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        connection = connect_readonly(db_path)
        tools = SQLTools(connection, self.config, SQLSandbox(self.config), self.limiter)

        def observation_message(observation: dict[str, Any]) -> str:
            return f"<observation>{canonical_json(observation)}</observation>\n\n{context}"

        try:
            if not initial_sql.strip():
                raise ValueError("Initial SQL must not be empty")
            initial_action = {"name": "execute_sql", "arguments": {"sql": initial_sql}}
            replay_response = (
                "<reasoning>I will execute the supplied candidate query.</reasoning>\n"
                f"<tool>{canonical_json(initial_action)}</tool>"
            )
            try:
                observation = tools.execute_sql(initial_sql)
            except SQLSandboxError as exc:
                observation = self._legacy_error(exc)
            if observation.get("status") == "success":
                last_successful_sql = initial_sql
            remaining -= 1
            observation = self.limiter.limit({**observation, "turns_remaining": remaining})
            turns.append(
                {
                    "source": "baseline_replay",
                    "response": replay_response,
                    "tool": "execute_sql",
                    "arguments": {"sql": initial_sql},
                    "observation": observation,
                }
            )
            messages.extend(
                [
                    {"role": "assistant", "content": replay_response},
                    {"role": "user", "content": observation_message(observation)},
                ]
            )
            while len(turns) <= self.config.max_turns:
                if remaining == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Final turn: no exploratory calls remain. Your next response must "
                                "call submit_sql in <tool> with your best SQL. Do not call "
                                "execute_sql or any inspection tool.\n\n"
                                f"{context}"
                            ),
                        }
                    )
                try:
                    generation = self.model.generate(
                        messages,
                        temperature=0.8,
                        top_p=1.0,
                        top_k=20,
                        min_p=0.0,
                        max_tokens=self.max_tokens,
                        seed=self.config.split_seed + sample_index,
                        enable_thinking=False,
                    )
                except ModelAdapterError as exc:
                    status = "model_error"
                    turns.append({"error": str(exc)})
                    break
                usage["prompt_tokens"] += generation.prompt_tokens
                usage["completion_tokens"] += generation.completion_tokens
                response = generation.text
                if (
                    generation.finish_reason == "stop"
                    and "<tool>" in response
                    and "</tool>" not in response
                ):
                    response += "</tool>"
                messages.append({"role": "assistant", "content": response})
                turn: dict[str, Any] = {
                    "source": "model",
                    "response": response,
                    "finish_reason": generation.finish_reason,
                }
                turns.append(turn)
                forced_reason: str | None = None
                proposed_sql: str | None = None
                try:
                    action = parse_v5_response(response)
                    turn["tool"] = action.name
                    turn["arguments"] = action.arguments
                    if action.name == "submit_sql":
                        final_sql = action.arguments["sql"]
                        status = "submitted_sql"
                        break
                    if remaining == 0:
                        forced_reason = "wrong_tool_on_final_turn"
                        if action.name == "execute_sql":
                            proposed_sql = action.arguments["sql"]
                    else:
                        if action.name == "execute_sql":
                            last_sql = action.arguments["sql"]
                        observation = self._execute(tools, action)
                        if action.name == "execute_sql" and observation.get("status") == "success":
                            last_successful_sql = action.arguments["sql"]
                except (ActionParseError, SQLSandboxError) as exc:
                    if remaining == 0:
                        forced_reason = exc.error_type
                    else:
                        observation = self._legacy_error(exc)
                if forced_reason is not None:
                    final_sql = (
                        proposed_sql or last_successful_sql or last_sql or "SELECT NULL WHERE 0"
                    )
                    status = "forced_submit_sql"
                    turn["forced_submission_reason"] = forced_reason
                    forced_action = {"name": "submit_sql", "arguments": {"sql": final_sql}}
                    forced_response = (
                        "<reasoning>Final submission after the turn budget.</reasoning>\n"
                        f"<tool>{canonical_json(forced_action)}</tool>"
                    )
                    turns.append(
                        {
                            "source": "evaluator_fallback",
                            "response": forced_response,
                            "tool": "submit_sql",
                            "arguments": {"sql": final_sql},
                            "reason": forced_reason,
                        }
                    )
                    messages.append({"role": "assistant", "content": forced_response})
                    break
                if remaining > 0:
                    remaining -= 1
                observation = self.limiter.limit({**observation, "turns_remaining": remaining})
                turn["observation"] = observation
                messages.append({"role": "user", "content": observation_message(observation)})
        finally:
            connection.close()
        verification = (
            ExecutionVerifier(db_path, task.reference_sql, self.config).verify(final_sql)
            if final_sql is not None
            else None
        )
        return {
            "task_id": task.task_id,
            "db_id": task.db_id,
            "difficulty": task.difficulty,
            "sample_index": sample_index,
            "status": status,
            "correct": verification.correct if verification else False,
            "final_sql": final_sql,
            "verification": verification.to_dict() if verification else None,
            "turns": turns,
            "messages": messages,
            "initial_sql": initial_sql,
            "usage": usage,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def vote_sql(
    task: TaskRecord, candidates: list[dict[str, Any]], config: EnvConfig
) -> dict[str, Any]:
    connection = connect_readonly(task.resolve_db_path(config.spider_root))
    sandbox = SQLSandbox(config)
    groups: dict[frozenset[tuple[Any, ...]], list[int]] = {}
    try:
        for index, candidate in enumerate(candidates):
            sql = candidate.get("final_sql")
            if not sql:
                continue
            try:
                result = sandbox.execute(
                    connection,
                    sql,
                    row_limit=config.verifier_max_rows,
                    timeout_seconds=config.verifier_timeout_seconds,
                    byte_limit=config.verifier_max_bytes,
                )
            except SQLSandboxError:
                continue
            if not result.has_more:
                groups.setdefault(frozenset(result.result.rows), []).append(index)
    finally:
        connection.close()
    if not groups:
        return {"selected_sample": None, "final_sql": None, "votes": 0, "valid": 0}
    winning_group = max(groups.values(), key=len)
    selected_sample = winning_group[0]
    return {
        "selected_sample": selected_sample,
        "final_sql": candidates[selected_sample]["final_sql"],
        "votes": len(winning_group),
        "valid": sum(map(len, groups.values())),
    }
