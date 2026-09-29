from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, TokenError

from evaluation.logging import append_jsonl, summarize
from evaluation.schema import oracle_table_names, render_schema
from sql_agent.action_parser import (
    ActionParseError,
    ExecuteSQLAction,
    parse_action,
)
from sql_agent.config import EnvConfig
from sql_agent.env import SQLAgentEnv
from sql_agent.model_adapter import ModelAdapter, ModelAdapterError
from sql_agent.models import TaskRecord, VerifierResult
from sql_agent.prompts import (
    PROMPT_VERSION,
    SUPPORTED_PROMPT_VERSIONS,
    build_direct_sql_prompt,
    render_agent_messages,
)
from sql_agent.verifier import ExecutionVerifier


class BaselineMode(StrEnum):
    DIRECT_FULL_SCHEMA = "direct_full_schema"
    DIRECT_NO_SCHEMA = "direct_no_schema"
    MULTI_TURN_AGENT = "multi_turn_agent"
    ORACLE_TABLE = "oracle_table"


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 512
    seed: int = 42
    enable_thinking: bool = False
    workers: int = 1
    prompt_version: str = PROMPT_VERSION


def extract_sql(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, dict):
        if isinstance(parsed.get("sql"), str):
            return parsed["sql"].strip()
        arguments = parsed.get("arguments")
    return value


class EvaluationRunner:
    def __init__(
        self,
        env_config: EnvConfig,
        model: ModelAdapter,
        runner_config: RunnerConfig | None = None,
    ):
        self.env_config = env_config
        self.model = model
        self.config = runner_config or RunnerConfig(seed=env_config.split_seed)
        if self.config.prompt_version not in SUPPORTED_PROMPT_VERSIONS:
            raise ValueError(f"unsupported prompt version: {self.config.prompt_version}")

    def run(
        self,
        tasks: Iterable[TaskRecord],
        modes: Iterable[BaselineMode],
        *,
        log_path: Path,
        summary_path: Path,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        identifier = run_id or uuid.uuid4().hex
        records: list[dict[str, Any]] = []
        jobs = [(task, mode) for task in tasks for mode in modes]

        def evaluate(job: tuple[TaskRecord, BaselineMode]) -> dict[str, Any]:
            return self.run_task(job[0], job[1], identifier)

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            for record in executor.map(evaluate, jobs):
                append_jsonl(log_path, record)
                records.append(record)
        summary = {
            "schema_version": 1,
            "prompt_version": self.config.prompt_version,
            "run_id": identifier,
            "model": self.model.model_name,
            "parameters": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "min_p": self.config.min_p,
                "max_tokens": self.config.max_tokens,
                "seed": self.config.seed,
                "enable_thinking": self.config.enable_thinking,
                "workers": self.config.workers,
                "agent_max_turns": self.env_config.max_turns,
            },
            **summarize(records),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def run_task(self, task: TaskRecord, mode: BaselineMode, run_id: str) -> dict[str, Any]:
        started = time.monotonic()
        try:
            if mode is BaselineMode.MULTI_TURN_AGENT:
                details = self._run_agent(task)
            else:
                details = self._run_direct(task, mode)
        except ModelAdapterError as exc:
            details = self._model_error(str(exc))
        return {
            "schema_version": 1,
            "prompt_version": self.config.prompt_version,
            "run_id": run_id,
            "model": self.model.model_name,
            "mode": mode.value,
            "seed": self.config.seed,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "max_tokens": self.config.max_tokens,
            "enable_thinking": self.config.enable_thinking,
            "agent_max_turns": self.env_config.max_turns,
            "task_id": task.task_id,
            "db_id": task.db_id,
            "split": task.split,
            "difficulty": task.difficulty,
            **details,
            "latency_seconds": time.monotonic() - started,
        }

    def _run_direct(self, task: TaskRecord, mode: BaselineMode) -> dict[str, Any]:
        db_path = task.resolve_db_path(self.env_config.spider_root)
        if mode is BaselineMode.DIRECT_NO_SCHEMA:
            schema = None
        elif mode is BaselineMode.ORACLE_TABLE:
            schema = render_schema(db_path, oracle_table_names(task.reference_sql))
        else:
            schema = render_schema(db_path)
        messages = build_direct_sql_prompt(task.question, schema)
        generation = self._generate(messages)
        sql = extract_sql(generation.text)
        result = ExecutionVerifier(db_path, task.reference_sql, self.env_config).verify(sql)
        first_correct = result.correct if result.agent_sql_valid else None
        return self._details(
            result=result,
            valid_action=bool(sql),
            turns=1,
            tool_calls=[],
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            first_sql_correct=first_correct,
            recovered=False,
            final_sql=sql,
            reference_sql=task.reference_sql,
            steps=[{"messages": messages, "response": generation.text}],
        )

    def _run_agent(self, task: TaskRecord) -> dict[str, Any]:
        env = SQLAgentEnv(self.env_config, prompt_version=self.config.prompt_version)
        steps: list[dict[str, Any]] = []
        tool_calls: list[str] = []
        sql_results: list[VerifierResult] = []
        all_actions_valid = True
        prompt_tokens = 0
        completion_tokens = 0
        final_sql: str | None = None
        final_observation: dict[str, Any] = {}
        try:
            env.reset(task)
            while not env.done:
                messages = render_agent_messages(env.history)
                generation = self._generate(messages)
                prompt_tokens += generation.prompt_tokens
                completion_tokens += generation.completion_tokens
                action_text = generation.text.strip()
                action = None
                try:
                    action = parse_action(action_text)
                    tool_calls.append(action.tool)
                except ActionParseError:
                    all_actions_valid = False
                if isinstance(action, ExecuteSQLAction):
                    candidate = action.arguments.sql
                    sql_results.append(env.verify(candidate))
                observation, done = env.step(action_text)
                if done and env.last_executed_sql is not None:
                    final_sql = env.last_executed_sql
                    verification = observation.get("verification")
                    if isinstance(verification, dict):
                        sql_results.append(VerifierResult(**verification))
                final_observation = observation
                steps.append(
                    {
                        "messages": messages,
                        "response": generation.text,
                        "observation": observation,
                        "generation": self._generation_metadata(generation),
                    }
                )
                if done:
                    break
        finally:
            env.close()
        final_result = sql_results[-1] if final_sql is not None and sql_results else None
        first_correct = sql_results[0].correct if sql_results else None
        recovered = bool(final_result and final_result.correct and first_correct is False)
        if final_result is None:
            final_result = self._failure_result(final_observation.get("error_type", "no_final_sql"))
        return self._details(
            result=final_result,
            valid_action=all_actions_valid,
            turns=len(steps),
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            first_sql_correct=first_correct,
            recovered=recovered,
            final_sql=final_sql,
            reference_sql=task.reference_sql,
            steps=steps,
        )

    def _generate(self, messages: list[dict[str, str]]):
        generation = self.model.generate(
            messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed,
            enable_thinking=self.config.enable_thinking,
        )
        if generation.completion_tokens:
            return generation
        fallback = self.env_config.tokenizer_path / "tokenizer.json"
        if not fallback.is_file():
            return generation
        from tokenizers import Tokenizer

        count = len(Tokenizer.from_file(str(fallback)).encode(generation.text).ids)
        return replace(generation, completion_tokens=count)

    @staticmethod
    def _generation_metadata(generation: Any) -> dict[str, Any]:
        metadata = asdict(generation)
        metadata.pop("text", None)
        return metadata

    def _details(
        self,
        *,
        result: VerifierResult,
        valid_action: bool,
        turns: int,
        tool_calls: list[str],
        prompt_tokens: int,
        completion_tokens: int,
        first_sql_correct: bool | None,
        recovered: bool,
        final_sql: str | None,
        reference_sql: str,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "success": result.correct,
            "reward": result.reward,
            "valid_action": valid_action,
            "valid_sql": result.agent_sql_valid,
            "executable_sql": result.agent_sql_executable,
            "turns": turns,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "token_count": prompt_tokens + completion_tokens,
            "first_sql_correct": first_sql_correct,
            "recovered_after_error": recovered,
            "failure_category": (
                None if result.correct else self._failure_category(result, final_sql, reference_sql)
            ),
            "final_sql": final_sql,
            "steps": steps,
        }

    @staticmethod
    def _failure_category(
        result: VerifierResult, submitted_sql: str | None, reference_sql: str
    ) -> str:
        mapping = {
            "parse_error": "sql_syntax_error",
            "syntax_error": "sql_syntax_error",
            "unknown_table": "wrong_table",
            "unknown_column": "schema_understanding",
            "ambiguous_column": "schema_understanding",
            "invalid_json": "invalid_json",
            "invalid_action": "invalid_json",
            "invalid_arguments": "wrong_arguments",
            "timeout": "timeout",
            "no_final_sql": "no_final_sql",
        }
        direct = mapping.get(result.error or "")
        if direct:
            return direct
        if result.error != "wrong_result" or not submitted_sql:
            return result.error or "wrong_result"
        try:
            submitted = parse_one(submitted_sql, read="sqlite")
            reference = parse_one(reference_sql, read="sqlite")
        except (ParseError, TokenError):
            return "sql_syntax_error"
        submitted_tables = {node.name.casefold() for node in submitted.find_all(exp.Table)}
        reference_tables = {node.name.casefold() for node in reference.find_all(exp.Table)}
        if submitted_tables != reference_tables:
            return "wrong_table"
        submitted_joins = [node.sql(dialect="sqlite") for node in submitted.find_all(exp.Join)]
        reference_joins = [node.sql(dialect="sqlite") for node in reference.find_all(exp.Join)]
        if submitted_joins != reference_joins:
            return "join_error"
        submitted_aggregates = [
            node.sql(dialect="sqlite") for node in submitted.find_all(exp.AggFunc)
        ]
        reference_aggregates = [
            node.sql(dialect="sqlite") for node in reference.find_all(exp.AggFunc)
        ]
        if submitted_aggregates != reference_aggregates:
            return "aggregation_error"
        submitted_where = submitted.find(exp.Where)
        reference_where = reference.find(exp.Where)
        if (submitted_where and submitted_where.sql(dialect="sqlite")) != (
            reference_where and reference_where.sql(dialect="sqlite")
        ):
            return "condition_error"
        return "wrong_result"

    @staticmethod
    def _failure_result(error: str) -> VerifierResult:
        return VerifierResult(0.0, False, False, False, False, False, False, False, error)

    @staticmethod
    def _model_error(message: str) -> dict[str, Any]:
        return {
            "success": False,
            "reward": 0.0,
            "valid_action": False,
            "valid_sql": False,
            "executable_sql": False,
            "turns": 0,
            "tool_calls": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "token_count": 0,
            "first_sql_correct": None,
            "recovered_after_error": False,
            "failure_category": "model_error",
            "final_sql": None,
            "steps": [],
            "model_error": message,
        }
