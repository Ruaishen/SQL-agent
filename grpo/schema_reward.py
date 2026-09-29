from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, SqlglotError, TokenError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from grpo.rollout import RolloutEpisode
from sql_agent.action_parser import (
    ActionParseError,
    ExecuteSQLAction,
    parse_action,
)
from sql_agent.models import TaskRecord
from sql_agent.sandbox import connect_readonly


def _name(value: str) -> str:
    return value.casefold()


@dataclass(frozen=True, slots=True)
class SQLiteCatalog:
    tables: dict[str, str]
    columns: dict[str, dict[str, str]]

    @classmethod
    def load(cls, path: Path) -> SQLiteCatalog:
        connection = connect_readonly(path)
        try:
            table_names = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            tables = {_name(table): _name(table) for table in table_names}
            columns: dict[str, dict[str, str]] = {}
            for table in table_names:
                escaped = table.replace('"', '""')
                values = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
                columns[_name(table)] = {
                    _name(str(row[1])): _name(str(row[1])) for row in values
                }
            return cls(tables=tables, columns=columns)
        finally:
            connection.close()

    @property
    def sqlglot_schema(self) -> dict[str, dict[str, str]]:
        return {
            table: {column: "UNKNOWN" for column in columns}
            for table, columns in self.columns.items()
        }

    @property
    def all_columns(self) -> frozenset[str]:
        return frozenset(column for values in self.columns.values() for column in values)


@dataclass(frozen=True, slots=True)
class SchemaGraph:
    tables: frozenset[str]
    columns: frozenset[str]
    edges: frozenset[tuple[str, str]]
    parsed: bool
    identifiers_valid: bool


@dataclass(frozen=True, slots=True)
class SchemaScore:
    candidate_source: str
    candidate_sql_available: bool
    candidate_parsed: bool
    identifiers_valid: bool
    table_precision: float
    table_recall: float
    table_f1: float
    column_precision: float
    column_recall: float
    column_f1: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    phi: float
    validity: float
    source_factor: float
    component_weights: tuple[float, float, float]
    multi_table_gold: bool
    schema_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_sql_from_responses(responses: Iterable[str]) -> tuple[str | None, str]:
    latest_execute: str | None = None
    for response in responses:
        try:
            action = parse_action(response)
        except ActionParseError:
            continue
        if isinstance(action, ExecuteSQLAction):
            latest_execute = action.arguments.sql
    if latest_execute is not None:
        return latest_execute, "execute_sql"
    return None, "none"


def candidate_sql_from_episode(episode: RolloutEpisode) -> tuple[str | None, str]:
    return candidate_sql_from_responses(turn.response_text for turn in episode.turns)


def _repair_sqlite_double_quoted_strings(tree: exp.Expression, catalog: SQLiteCatalog) -> None:
    """Repair SQLite DQS literals without changing quoted known identifiers."""
    known_columns = catalog.all_columns
    for column in list(tree.find_all(exp.Column)):
        identifier = column.this
        if (
            isinstance(identifier, exp.Identifier)
            and identifier.args.get("quoted")
            and not column.table
            and _name(identifier.name) not in known_columns
        ):
            column.replace(exp.Literal.string(identifier.name))


def _unknown_table(table: str) -> str:
    return f"!unknown_table:{_name(table)}"


def _unknown_column(column: str) -> str:
    return f"!unknown_column:{_name(column)}"


def _ambiguous_column(column: str) -> str:
    return f"!ambiguous_column:{_name(column)}"


class _GraphExtractor:
    def __init__(self, tree: exp.Expression, catalog: SQLiteCatalog):
        self.tree = tree
        self.catalog = catalog
        self.scopes = list(traverse_scope(tree))
        self.column_scopes: dict[int, Scope] = {
            id(column): scope for scope in self.scopes for column in scope.columns
        }

    def _find_source(self, scope: Scope | None, alias: str) -> tuple[Scope, Any] | None:
        wanted = _name(alias)
        current = scope
        while current is not None:
            for source_alias, source in current.sources.items():
                if _name(source_alias) == wanted:
                    return current, source
            current = current.parent
        return None

    def _derived_projection(self, scope: Scope, column: str) -> exp.Expression | None:
        wanted = _name(column)
        for projection in scope.expression.selects:
            if _name(projection.alias_or_name) == wanted:
                return projection
        return None

    def _source_has_column(self, source: Any, column: str) -> bool:
        wanted = _name(column)
        if isinstance(source, exp.Table):
            return wanted in self.catalog.columns.get(_name(source.name), {})
        if isinstance(source, Scope):
            return self._derived_projection(source, wanted) is not None
        return False

    def _resolve_from_source(
        self,
        source: Any,
        column: str,
        seen: set[tuple[int, str]],
    ) -> set[str]:
        wanted = _name(column)
        if isinstance(source, exp.Table):
            table = _name(source.name)
            if table not in self.catalog.tables:
                return {_unknown_table(table)}
            if wanted not in self.catalog.columns[table]:
                return {f"{table}.{_unknown_column(wanted)}"}
            return {f"{table}.{wanted}"}
        if isinstance(source, Scope):
            marker = (id(source), wanted)
            if marker in seen:
                return {_unknown_column(wanted)}
            projection = self._derived_projection(source, wanted)
            if projection is None:
                return {_unknown_column(wanted)}
            nested_seen = {*seen, marker}
            resolved: set[str] = set()
            for nested in projection.find_all(exp.Column):
                nested_scope = self.column_scopes.get(id(nested), source)
                resolved.update(self.resolve_column(nested_scope, nested, nested_seen))
            return resolved or {_unknown_column(wanted)}
        return {_unknown_column(wanted)}

    def resolve_column(
        self,
        scope: Scope,
        column: exp.Column,
        seen: set[tuple[int, str]] | None = None,
    ) -> set[str]:
        seen = set() if seen is None else seen
        wanted = _name(column.name)
        if column.table:
            found = self._find_source(scope, column.table)
            if found is None:
                physical = _name(column.table)
                if physical in self.catalog.tables:
                    return self._resolve_from_source(
                        exp.to_table(physical), wanted, seen
                    )
                return {_unknown_table(column.table), _unknown_column(wanted)}
            return self._resolve_from_source(found[1], wanted, seen)

        owners = [
            source for source in scope.sources.values() if self._source_has_column(source, wanted)
        ]
        if len(owners) == 1:
            return self._resolve_from_source(owners[0], wanted, seen)
        if len(owners) > 1:
            return {_ambiguous_column(wanted)}
        return {_unknown_column(wanted)}

    def extract(self) -> SchemaGraph:
        tables: set[str] = set()
        for scope in self.scopes:
            for source in scope.sources.values():
                if isinstance(source, exp.Table):
                    table = _name(source.name)
                    tables.add(table if table in self.catalog.tables else _unknown_table(table))

        columns: set[str] = set()
        resolved_by_id: dict[int, set[str]] = {}
        for scope in self.scopes:
            for column in scope.columns:
                resolved = self.resolve_column(scope, column)
                resolved_by_id[id(column)] = resolved
                columns.update(resolved)

        edges: set[tuple[str, str]] = set()
        for equality in self.tree.find_all(exp.EQ):
            left, right = equality.this, equality.expression
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            left_values = resolved_by_id.get(id(left))
            right_values = resolved_by_id.get(id(right))
            if left_values is None:
                left_scope = self.column_scopes.get(id(left))
                left_values = self.resolve_column(left_scope, left) if left_scope else set()
            if right_values is None:
                right_scope = self.column_scopes.get(id(right))
                right_values = self.resolve_column(right_scope, right) if right_scope else set()
            if len(left_values) != 1 or len(right_values) != 1:
                continue
            a, b = next(iter(left_values)), next(iter(right_values))
            if a == b or a.split(".", 1)[0] == b.split(".", 1)[0]:
                continue
            edges.add(tuple(sorted((a, b))))

        invalid = any(value.startswith("!") or ".!" in value for value in tables | columns)
        return SchemaGraph(
            tables=frozenset(tables),
            columns=frozenset(columns),
            edges=frozenset(edges),
            parsed=True,
            identifiers_valid=not invalid,
        )


def extract_schema_graph(
    sql: str,
    catalog: SQLiteCatalog,
    *,
    strict: bool,
) -> SchemaGraph:
    try:
        tree = parse_one(sql, read="sqlite")
        _repair_sqlite_double_quoted_strings(tree, catalog)
        tree = qualify(
            tree,
            dialect="sqlite",
            schema=catalog.sqlglot_schema,
            validate_qualify_columns=strict,
            quote_identifiers=False,
            identify=False,
        )
        return _GraphExtractor(tree, catalog).extract()
    except (SqlglotError, ParseError, TokenError, ValueError, TypeError, AttributeError):
        if strict:
            raise
        return SchemaGraph(frozenset(), frozenset(), frozenset(), False, False)


def _f1(candidate: frozenset[Any], gold: frozenset[Any]) -> tuple[float, float, float]:
    if not candidate and not gold:
        return 1.0, 1.0, 1.0
    if not candidate or not gold:
        return 0.0, 0.0, 0.0
    overlap = len(candidate & gold)
    precision = overlap / len(candidate)
    recall = overlap / len(gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


class SchemaRewarder:
    def __init__(
        self,
        spider_root: Path,
        component_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        *,
        single_table_weights: tuple[float, float, float] | None = None,
        multi_table_weights: tuple[float, float, float] | None = None,
        execute_sql_factor: float = 1.0,
    ):
        self.spider_root = spider_root
        self.component_weights = component_weights
        self.single_table_weights = single_table_weights
        self.multi_table_weights = multi_table_weights
        self.execute_sql_factor = execute_sql_factor
        self._catalogs: dict[Path, SQLiteCatalog] = {}
        self._gold_graphs: dict[str, SchemaGraph] = {}

    def _catalog(self, task: TaskRecord) -> SQLiteCatalog:
        path = task.resolve_db_path(self.spider_root)
        if path not in self._catalogs:
            self._catalogs[path] = SQLiteCatalog.load(path)
        return self._catalogs[path]

    def gold_graph(self, task: TaskRecord) -> SchemaGraph:
        if task.task_id not in self._gold_graphs:
            self._gold_graphs[task.task_id] = extract_schema_graph(
                task.reference_sql, self._catalog(task), strict=True
            )
        return self._gold_graphs[task.task_id]

    def score_sql(
        self,
        task: TaskRecord,
        sql: str | None,
        source: str,
    ) -> SchemaScore:
        candidate = (
            extract_schema_graph(sql, self._catalog(task), strict=False)
            if sql is not None
            else SchemaGraph(frozenset(), frozenset(), frozenset(), False, False)
        )
        gold = self.gold_graph(task)
        tp, tr, tf = _f1(candidate.tables, gold.tables)
        cp, cr, cf = _f1(candidate.columns, gold.columns)
        ep, er, ef = _f1(candidate.edges, gold.edges)
        multi_table_gold = len(gold.tables) > 1
        weights = self.component_weights
        if multi_table_gold and self.multi_table_weights is not None:
            weights = self.multi_table_weights
        elif not multi_table_gold and self.single_table_weights is not None:
            weights = self.single_table_weights
        phi = sum(
            weight * value
            for weight, value in zip(weights, (tf, cf, ef), strict=True)
        )
        validity = 1.0 if candidate.identifiers_valid else 0.25 if candidate.parsed else 0.0
        source_factor = 1.0
        if source == "execute_sql":
            source_factor = self.execute_sql_factor
        else:
            source_factor = 0.0
        return SchemaScore(
            candidate_source=source,
            candidate_sql_available=sql is not None,
            candidate_parsed=candidate.parsed,
            identifiers_valid=candidate.identifiers_valid,
            table_precision=tp,
            table_recall=tr,
            table_f1=tf,
            column_precision=cp,
            column_recall=cr,
            column_f1=cf,
            edge_precision=ep,
            edge_recall=er,
            edge_f1=ef,
            phi=phi,
            validity=validity,
            source_factor=source_factor,
            component_weights=weights,
            multi_table_gold=multi_table_gold,
            schema_score=validity * source_factor * phi,
        )

    def score_episode(self, task: TaskRecord, episode: RolloutEpisode) -> SchemaScore:
        sql, source = candidate_sql_from_episode(episode)
        return self.score_sql(task, sql, source)
