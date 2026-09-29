from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sql_agent.action_parser import (
    TOOL_DEFINITIONS,
    ActionParseError,
    ExecuteSQLAction,
    InspectTableAction,
    InspectTablesAction,
    InspectValuesAction,
    ListTablesAction,
    parse_action,
)
from sql_agent.config import EnvConfig
from sql_agent.models import TaskRecord, VerifierResult
from sql_agent.prompts import (
    LEGACY_PROMPT_VERSION,
    PROMPT_VERSION,
    build_system_prompt,
)
from sql_agent.sandbox import SQLSandbox, SQLSandboxError, connect_readonly
from sql_agent.tools import SQLTools
from sql_agent.truncation import ObservationLimiter, TokenCounter, canonical_json
from sql_agent.verifier import ExecutionVerifier


class SQLAgentEnv:
    def __init__(
        self,
        config: EnvConfig,
        *,
        prompt_version: str = PROMPT_VERSION,
        reserve_final_submission: bool = False,
        system_prompt: str | None = None,
        tool_definitions: tuple[dict[str, Any], ...] | None = None,
    ):
        config.validate()
        self.config = config
        self.token_counter = TokenCounter(config.tokenizer_path)
        self.limiter = ObservationLimiter(config, self.token_counter)
        self.sandbox = SQLSandbox(config)
        self.connection: sqlite3.Connection | None = None
        self.tools: SQLTools | None = None
        self.task: TaskRecord | None = None
        self.db_path: Path | None = None
        self.verifier: ExecutionVerifier | None = None
        self.history: list[dict[str, Any]] = []
        self.turn = 0
        self.last_executed_sql: str | None = None
        self.done = True
        self.prompt_version = prompt_version
        self.reserve_final_submission = reserve_final_submission
        self.system_prompt = system_prompt or build_system_prompt(
            config.max_turns, prompt_version=prompt_version
        )
        self.tool_definitions = tool_definitions or TOOL_DEFINITIONS

    def reset(self, task: TaskRecord | dict[str, Any]) -> dict[str, Any]:
        self.close()
        record = TaskRecord.from_dict(task) if isinstance(task, dict) else task
        db_path = record.resolve_db_path(self.config.spider_root)
        connection = connect_readonly(db_path)
        self.connection = connection
        self.tools = SQLTools(connection, self.config, self.sandbox, self.limiter)
        self.task = record
        self.db_path = db_path
        self.verifier = ExecutionVerifier(db_path, record.reference_sql, self.config)
        self.turn = 0
        self.last_executed_sql = None
        self.done = False
        self.history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": record.question},
        ]
        ready = {
            "status": "ready",
            "question": record.question,
            "tools": [definition["name"] for definition in TOOL_DEFINITIONS],
            "action_format": {"tool": "<tool_name>", "arguments": {}},
            "max_turns": self.config.max_turns,
        }
        if self.prompt_version != LEGACY_PROMPT_VERSION:
            ready["turns_remaining"] = self.config.max_turns
        observation = self.limiter.limit(ready)
        self.history.append({"role": "observation", "content": canonical_json(observation)})
        return observation

    def step(self, raw_action: str | Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        if self.done or self.connection is None or self.tools is None:
            return self.limiter.limit(
                self._with_turn_budget(
                    {
                        "status": "error",
                        "error_type": "episode_done",
                        "message": "Reset the environment before taking another action",
                        "termination_reason": "episode_done",
                    }
                )
            ), True
        if self.reserve_final_submission and self.turn >= self.config.max_turns:
            return self.limiter.limit(
                self._with_turn_budget(
                    {
                        "status": "error",
                        "error_type": "exploration_budget_exhausted",
                        "message": "Submit the final SQL; no exploration calls remain",
                    }
                )
            ), False
        if isinstance(raw_action, str):
            action_text = raw_action
        else:
            try:
                action_text = canonical_json(dict(raw_action))
            except (TypeError, ValueError):
                action_text = "<non-json action>"
        if self._context_tokens(action_text) > self._context_action_limit:
            return self._terminate(
                self.limiter.limit(
                    {
                        "status": "error",
                        "error_type": "context_limit",
                        "message": "Action would exceed the context token budget",
                        "reward": 0.0,
                    }
                ),
                "context_limit",
            )
        self.turn += 1
        try:
            action = parse_action(raw_action)
            if isinstance(action, ListTablesAction):
                observation = self.tools.list_tables()
            elif isinstance(action, InspectTableAction):
                observation = self.tools.inspect_table(action.arguments.table_name)
            elif isinstance(action, InspectTablesAction):
                observation = self.tools.inspect_tables(action.arguments.table_names)
            elif isinstance(action, InspectValuesAction):
                observation = self.tools.inspect_values(
                    action.arguments.table_name, action.arguments.column_name
                )
            elif isinstance(action, ExecuteSQLAction):
                self.last_executed_sql = action.arguments.sql
                observation = self.tools.execute_sql(action.arguments.sql)
            else:  # pragma: no cover - exhaustive discriminated union
                raise AssertionError("Unhandled action type")
        except ActionParseError as exc:
            observation = self.limiter.limit(
                {"status": "error", "error_type": exc.error_type, "message": exc.message}
            )
        except SQLSandboxError as exc:
            observation = self.tools.error(exc)
        self._append_history(action_text, observation)
        observation = self.limiter.limit(self._with_turn_budget(observation))
        if self.turn >= self.config.max_turns and not self.reserve_final_submission:
            observation = dict(observation)
            if self.last_executed_sql is not None:
                result = self.verify(self.last_executed_sql)
                observation.update(
                    {
                        "status": "finalized",
                        "reward": result.reward,
                        "verification": result.to_dict(),
                        "termination_reason": "final_execute_sql",
                    }
                )
            else:
                observation.update({"reward": 0.0, "termination_reason": "max_turns"})
            observation = self.limiter.limit(observation)
            return self._terminate(
                observation, observation["termination_reason"], append_history=False
            )
        return observation, False

    def submit_sql(self, arguments: Any) -> tuple[dict[str, Any], bool]:
        """Finalize a collected trajectory without spending an exploration call."""
        if not self.reserve_final_submission:
            raise RuntimeError("Final submission is not reserved in this environment")
        if self.done or self.tools is None:
            return self.limiter.limit(
                self._with_turn_budget(
                    {"status": "error", "error_type": "episode_done"}
                )
            ), True
        try:
            action_text = canonical_json({"tool": "submit_sql", "arguments": arguments})
        except (TypeError, ValueError):
            action_text = "<invalid submission>"
        if self._context_tokens(action_text) > self._context_action_limit:
            return self._terminate(
                {"status": "error", "error_type": "context_limit", "reward": 0.0},
                "context_limit",
            )
        try:
            action = parse_action({"tool": "execute_sql", "arguments": arguments})
            assert isinstance(action, ExecuteSQLAction)
        except ActionParseError as exc:
            return self._terminate(
                {"status": "error", "error_type": exc.error_type, "message": exc.message},
                "invalid_submission",
            )
        try:
            observation = self.tools.execute_sql(action.arguments.sql)
        except SQLSandboxError as exc:
            observation = self.tools.error(exc)
        verification = self.verify(action.arguments.sql).to_dict()
        observation = {
            **observation,
            "verification": verification,
            "reward": verification["reward"],
        }
        self._append_history(action_text, observation)
        return self._terminate(observation, "submit_sql", append_history=False)

    def verify(self, submitted_sql: str) -> VerifierResult:
        if self.verifier is None:
            raise RuntimeError("Environment has not been reset")
        return self.verifier.verify(submitted_sql)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self.connection = None
        self.tools = None
        self.done = True

    @property
    def _context_action_limit(self) -> int:
        return self.config.max_context_tokens - self.config.reserved_action_tokens

    def _context_tokens(self, pending_action: str = "") -> int:
        transcript = "\n".join(f"{item['role']}:{item['content']}" for item in self.history)
        if pending_action:
            transcript += f"\nassistant:{pending_action}"
        return self.token_counter.count_text(transcript)

    def _append_history(self, action_text: str, observation: dict[str, Any]) -> None:
        self.history.extend(
            [
                {"role": "assistant", "content": action_text},
                {"role": "observation", "content": canonical_json(observation)},
            ]
        )

    def _with_turn_budget(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.prompt_version == LEGACY_PROMPT_VERSION:
            return observation
        return {
            **observation,
            "max_turns": self.config.max_turns,
            "turns_remaining": max(0, self.config.max_turns - self.turn),
        }

    def _terminate(
        self,
        observation: dict[str, Any],
        reason: str,
        *,
        append_history: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        if "termination_reason" not in observation:
            observation = {**observation, "termination_reason": reason}
        observation = self.limiter.limit(self._with_turn_budget(observation))
        if append_history:
            self.history.append({"role": "observation", "content": canonical_json(observation)})
        self.close()
        return observation, True
