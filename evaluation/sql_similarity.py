"""Schema-bound, value-sensitive SQL similarity. Never executes input SQL.

APTED computes exact unit-cost ordered TED on the canonical comparison trees.
The canonicalizer covers a conservative subset of SQLite equivalences only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Any

import sqlglot
from apted import APTED, Config
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer import optimize
from sqlglot.optimizer.optimizer import RULES
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

METRIC_VERSION = "c-tsed-sql-v3.1.0"
LIMITATIONS = [
    "Exact ordered TED on the comparison trees; not complete SQL equivalence.",
    "SQLGlot optimize() applies its complete pinned-version logical rewrite pipeline.",
    "Simple inner joins with distinct physical tables are canonicalized before optimize().",
    "The optimized representation does not prove full SQLite semantic equivalence.",
    "AND/OR sorting can amplify edits when a changed condition moves in sort order.",
    "Output column names are ignored; output positions and derived-column bindings remain.",
    "An optimal edit mapping is reported; it need not be the unique optimum.",
]


class Unscorable(ValueError):
    """Input outside the supported, safely bound scoring domain."""


def _fold(value: str) -> str:
    # SQLite identifier case folding is ASCII-only; do not casefold Unicode.
    return value.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))


@dataclass(eq=False)
class Node:
    name: str
    children: list[Node] = field(default_factory=list)
    path: str = ""
    region: str = ""

    def key(self) -> tuple:
        return self.name, tuple(child.key() for child in self.children)

    def as_dict(self) -> dict:
        return {"label": self.name, "children": [child.as_dict() for child in self.children]}


def _label(*parts: Any) -> str:
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def _safe_boolean(node: exp.Expression) -> bool:
    # Restrict commutative rewrites to ordinary total scalar predicates.
    allowed = (
        exp.And, exp.Or, exp.Not, exp.Paren, exp.Column, exp.Identifier,
        exp.Literal, exp.Null, exp.Boolean, exp.EQ, exp.NEQ, exp.GT, exp.GTE,
        exp.LT, exp.LTE, exp.Is, exp.In, exp.Between, exp.Neg,
    )
    return all(isinstance(item, allowed) for item in node.walk())


def _literal(node: exp.Expression) -> bool:
    return isinstance(node, (exp.Literal, exp.Null, exp.Boolean)) or (
        isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal)
    )


def _inner_join_signature(query: exp.Select) -> tuple[str, ...] | None:
    """Return a stable table set only for a simple, reorderable join block."""
    from_ = query.args.get("from_")
    joins = query.args.get("joins") or []
    if not from_ or not isinstance(from_.this, exp.Table) or not joins:
        return None
    if query.args.get("limit") or query.args.get("offset"):
        return None
    if any(not isinstance(join.this, exp.Table) or
           join.args.get("side") or join.args.get("method") or join.args.get("using") or
           join.args.get("pivots") or join.args.get("kind") not in (None, "INNER", "CROSS") or
           not set(join.args).issubset({"this", "on", "kind", "pivots"}) or
           (join.args.get("on") is not None and not _safe_boolean(join.args["on"]))
           for join in joins):
        return None
    tables = [from_.this] + [join.this for join in joins]
    if any(not set(table.args).issubset({"this", "alias", "db", "catalog"})
           for table in tables):
        return None
    names = [_fold(table.name) for table in tables]
    if len(names) != len(set(names)):
        # A self-join needs a separate role-isomorphism analysis.
        return None
    return tuple(sorted(names))


def _canonicalize_inner_join(query: exp.Select) -> None:
    """Rewrite one matched inner-join block to stable table and predicate order."""
    tables = [query.args["from_"].this] + [join.this for join in query.args["joins"]]
    ordered = sorted(tables, key=lambda table: _fold(table.name))
    predicates = [join.args["on"].copy() for join in query.args["joins"]
                  if join.args.get("on")]
    if query.args.get("where"):
        predicates.append(query.args["where"].this.copy())
    query.set("from_", exp.From(this=ordered[0].copy()))
    query.set("joins", [exp.Join(this=table.copy(), kind="CROSS") for table in ordered[1:]])
    if predicates:
        condition = predicates[0]
        for predicate in predicates[1:]:
            condition = exp.and_(condition, predicate, copy=False)
        query.set("where", exp.Where(this=condition))


def _canonicalize_matched_inner_joins(pred: exp.Expression, gold: exp.Expression) -> int:
    """Normalize both sides only when corresponding blocks have the same tables.

    Qualification already expanded SELECT *, preserving its output positions.
    Pairwise gating keeps positional source labels comparable for unlike joins.
    """
    pred_selects = list(pred.find_all(exp.Select))
    gold_selects = list(gold.find_all(exp.Select))
    if len(pred_selects) != len(gold_selects):
        return 0
    changed = 0
    for left, right in zip(pred_selects, gold_selects):
        signature = _inner_join_signature(left)
        if signature is None or signature != _inner_join_signature(right):
            continue
        _canonicalize_inner_join(left)
        _canonicalize_inner_join(right)
        changed += 1
    return changed


def _sqlite_double_quotes(tree: exp.Expression, schema: dict, sql: str) -> int:
    """SQLite DQS fallback only when an unqualified quoted name cannot bind.

    Preserve the original spelling of string values before folding identifiers.
    Unknown derived output schemas are left unresolved, not guessed.
    """
    scopes = {id(s.expression): s for s in traverse_scope(tree)}
    changed = 0
    for column in list(tree.find_all(exp.Column)):
        identifier = column.this
        if column.table or not isinstance(identifier, exp.Identifier) or not identifier.quoted:
            continue
        start = identifier.meta.get("start")
        if start is None or sql[start:start + 1] != '"':
            # SQLite does not apply DQS fallback to [bracket] or `backtick` names.
            continue
        ancestor = column.parent
        while ancestor is not None and id(ancestor) not in scopes:
            ancestor = ancestor.parent
        current = scopes.get(id(ancestor))
        visible, known = set(), True
        while current is not None:
            # SELECT aliases are visible in several SQLite clauses, but not in
            # another expression of the same SELECT projection list.
            branch = column
            while branch.parent is not None and branch.parent is not current.expression:
                branch = branch.parent
            in_projection = branch.parent is current.expression and branch.arg_key == "expressions"
            if not in_projection:
                visible.update(_fold(n) for n in current.expression.named_selects)
            for _, source in current.selected_sources.values():
                if isinstance(source, exp.Table):
                    columns = schema.get(_fold(source.name))
                    if columns is None:
                        known = False
                    else:
                        visible.update(columns)
                elif isinstance(source, Scope):
                    names = source.outer_columns or source.expression.named_selects
                    if "*" in names:
                        known = False
                    visible.update(_fold(n) for n in names)
                else:
                    known = False
            current = current.parent
        if known and _fold(column.name) not in visible:
            column.replace(exp.Literal.string(column.name))
            changed += 1
    return changed


def _prepare(sql: str, schema: dict, max_nodes: int,
             sqlite_dqs: bool) -> tuple[exp.Expression, str, list[str]]:
    if not isinstance(sql, str) or not sql.strip():
        raise Unscorable("empty_sql")
    if len(sql) > 100_000:
        raise Unscorable("sql_length_limit")
    parsed = [item for item in sqlglot.parse(sql, read="sqlite")
              if item is not None and not isinstance(item, exp.Semicolon)]
    if len(parsed) != 1 or not isinstance(parsed[0], exp.Query):
        raise Unscorable("expected_one_read_only_query")
    tree = parsed[0]
    if sum(1 for _ in tree.walk()) > max_nodes * 4:
        raise Unscorable("input_node_limit")
    forbidden = (exp.Into, exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command)
    if any(isinstance(item, forbidden) for item in tree.walk()):
        raise Unscorable("non_read_only_query")
    if any(isinstance(item, exp.With) and item.args.get("recursive") for item in tree.walk()):
        raise Unscorable("recursive_cte_not_supported")
    dqs_changes = _sqlite_double_quotes(tree, schema, sql) if sqlite_dqs else 0
    for item in tree.find_all(exp.Identifier):
        item.set("this", _fold(item.this))
    # Verify physical tables independently: COUNT(*) can otherwise evade column validation.
    for scope in traverse_scope(tree):
        for _, source in scope.selected_sources.values():
            if isinstance(source, exp.Table):
                if source.db or source.catalog or source.name not in schema:
                    raise Unscorable(f"unknown_or_attached_table:{source.sql()}")
    tree = qualify(
        tree, dialect="sqlite", schema=schema, infer_schema=False,
        # Our binding-based builder handles alpha renaming. SQLGlot's early alias
        # renaming in 30.17.0 can break expansion of an unaliased table.*.
        canonicalize_table_aliases=False, expand_stars=True,
        validate_qualify_columns=True, allow_partial_qualification=False,
    )
    qualified_sql = tree.sql(dialect="sqlite")
    warnings = []
    if dqs_changes:
        warnings.append("sqlite_dqs_fallback_applied")
    return tree, qualified_sql, warnings


def _ordered_column_comparisons(tree: exp.Expression) -> tuple:
    """Fingerprint comparison orientation before optimize() can reorder it."""
    builder = TreeBuilder(tree)
    operators = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
    result = []
    for item in tree.walk():
        if not isinstance(item, operators):
            continue
        if not isinstance(item.this, exp.Column) or not isinstance(item.expression, exp.Column):
            continue
        ancestor = item.parent
        while ancestor is not None and id(ancestor) not in builder.scopes:
            ancestor = ancestor.parent
        scope = builder.scopes.get(id(ancestor))
        if scope is None:
            raise Unscorable("comparison_without_scope")
        left = builder.build(item.this, scope).name
        right = builder.build(item.expression, scope).name
        result.append((builder.ids[id(scope.expression)], item.key, left, right))
    return tuple(sorted(result))


class TreeBuilder:
    def __init__(self, tree: exp.Expression):
        scopes = traverse_scope(tree)
        self.scopes = {id(scope.expression): scope for scope in scopes}
        queries = [item for item in tree.walk(bfs=False) if id(item) in self.scopes]
        self.ids = {id(item): f"q{i}" for i, item in enumerate(queries)}
        self.sources = {
            id(scope): {name: (i, source) for i, (name, (_, source))
                        in enumerate(scope.selected_sources.items())}
            for scope in scopes
        }

    def output_index(self, scope: Scope, name: str) -> int:
        names = list(scope.outer_columns or scope.expression.named_selects)
        matches = [i for i, value in enumerate(names) if _fold(value) == _fold(name)]
        if len(matches) != 1:
            raise Unscorable(f"ambiguous_output_binding:{name}")
        return matches[0]

    def binding(self, scope: Scope, table: str) -> tuple[Scope, int, Any]:
        current = scope
        while current is not None:
            match = self.sources[id(current)].get(table)
            if match is not None:
                return current, *match
            current = current.parent
        raise Unscorable(f"unresolved_relation:{table}")

    def build(self, item: Any, scope: Scope | None = None, region: str = "") -> Node:
        if not isinstance(item, exp.Expression):
            if isinstance(item, Enum):
                return Node(_label("ENUM", type(item).__name__, item.value), region=region)
            if not isinstance(item, (str, int, float, bool, type(None))):
                raise Unscorable(f"unsupported_ast_attribute:{type(item).__name__}")
            return Node(_label("VALUE", type(item).__name__, item), region=region)
        if id(item) in self.scopes:
            scope = self.scopes[id(item)]
            region = self.ids[id(item)] + ":QUERY"
        if isinstance(item, exp.Paren):
            return self.build(item.this, scope, region)
        if isinstance(item, exp.Alias):
            # References to aliases are resolved separately, including derived columns.
            return self.build(item.this, scope, region)
        if isinstance(item, exp.Column):
            if scope is None:
                raise Unscorable("column_without_scope")
            if not item.table:
                # qualify preserves SELECT-output references in ORDER BY.
                index = self.output_index(scope, item.name)
                projection = scope.expression.selects[index]
                return self.build(projection, scope, region)
            owner, index, source = self.binding(scope, item.table)
            origin = self.ids[id(owner.expression)] + f"/r{index}"
            column = item.name
            if isinstance(source, Scope):
                column = f"output_{self.output_index(source, column)}"
            return Node(_label("COLUMN", origin, column), region=region)
        if isinstance(item, exp.Literal):
            kind = "text" if item.is_string else ("integer" if item.is_int else "number")
            return Node(_label("LITERAL", kind, item.this), region=region)
        if isinstance(item, exp.Identifier):
            return Node(_label("IDENTIFIER", item.this), region=region)
        if isinstance(item, exp.Table):
            _, index, source = self.binding(scope, item.alias_or_name)
            if isinstance(source, Scope):
                name = _label("DERIVED", self.ids[id(source.expression)], index)
            else:
                name = _label("TABLE", item.catalog, item.db, item.name, index)
            # SQL modifiers (e.g. sampling) must not disappear from comparison.
            children = self.arguments(item, scope, region, {"this", "db", "catalog", "alias"})
            return Node(name, children, region=region)
        if isinstance(item, (exp.And, exp.Or)) and _safe_boolean(item):
            flat = []

            def flatten(current: exp.Expression) -> None:
                while isinstance(current, exp.Paren):
                    current = current.this
                if isinstance(current, type(item)):
                    flatten(current.this)
                    flatten(current.expression)
                else:
                    flat.append(self.build(current, scope, region))

            flatten(item)
            return Node(_label(item.key.upper()), sorted(flat, key=Node.key), region=region)
        # Only reverse literal-vs-column comparisons; never commute two columns
        # (their SQLite collations may differ) or explicit COLLATE expressions.
        reverse = {exp.LT: exp.GT, exp.LTE: exp.GTE, exp.GT: exp.LT,
                   exp.GTE: exp.LTE, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}
        if type(item) in reverse and _literal(item.this) and isinstance(item.expression, exp.Column):
            swapped = reverse[type(item)](this=item.expression.copy(), expression=item.this.copy())
            return self.build(swapped, scope, region)
        children = self.arguments(item, scope, region)
        return Node(_label(item.key.upper()), children, region=region)

    def arguments(self, item: exp.Expression, scope: Scope, region: str,
                  skip: set[str] | None = None) -> list[Node]:
        children = []
        for key in sorted(item.args):
            value = item.args[key]
            if key in (skip or set()) or key == "alias" or value is None or value == []:
                continue
            # Parser stores omitted ASC as None and explicit ASC as False.
            if isinstance(item, exp.Ordered) and key == "desc" and value is False:
                continue
            child_region = region
            if isinstance(item, exp.Select):
                child_region = self.ids[id(item)] + ":" + key.upper().rstrip("_")
            values = value if isinstance(value, list) else [value]
            nodes = [self.build(part, scope, child_region) for part in values]
            if isinstance(item, exp.In) and key == "expressions" and all(_literal(v) for v in values):
                nodes.sort(key=Node.key)
            children.append(Node(_label("ROLE", key), nodes, region=child_region))
        return children


def _index(tree: Node, max_nodes: int) -> int:
    stack = [(tree, "root")]
    count = 0
    while stack:
        node, path = stack.pop()
        node.path = path
        count += 1
        if count > max_nodes:
            raise Unscorable("comparison_node_limit")
        stack.extend((child, f"{path}/{i}") for i, child in enumerate(node.children))
    return count


def _normalize_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        raise Unscorable("schema_must_be_table_column_type_mapping")
    result = {}
    for table, columns in schema.items():
        if not isinstance(table, str) or not isinstance(columns, dict) or not columns:
            raise Unscorable("schema_must_be_table_column_type_mapping")
        folded = {}
        for column, dtype in columns.items():
            if not isinstance(column, str) or not isinstance(dtype, str):
                raise Unscorable("schema_columns_require_string_types")
            if _fold(column) in folded:
                raise Unscorable("duplicate_schema_column")
            folded[_fold(column)] = dtype
        if _fold(table) in result:
            raise Unscorable("duplicate_schema_table")
        result[_fold(table)] = folded
    return result


def sql_similarity(pred_sql: str, gold_sql: str, schema: dict, *,
                   max_nodes: int = 600, include_trees: bool = False,
                   sqlite_dqs: bool = True) -> dict:
    """Return a JSON-serializable score or status=unscorable with null score.

    schema: flat {table: {column: SQL_type}}; SQLite dialect. No SQL is executed.
    edit_script is an optimal mapping delta with immutable source/target paths,
    NOT a sequential patch program. insert/delete promote/adopt child nodes.
    """
    result = {
        "metric_version": METRIC_VERSION,
        "dependencies": {"sqlglot": sqlglot.__version__, "apted": version("apted")},
        "optimizer": {"name": "sqlglot.optimize", "rules": [rule.__name__ for rule in RULES]},
        "dialect": "sqlite", "status": "unscorable", "similarity": None, "ted": None,
        "sqlite_dqs": sqlite_dqs,
        "pred_sql": pred_sql, "gold_sql": gold_sql,
        "limitations": LIMITATIONS,
    }
    stage = "schema"
    try:
        if max_nodes < 1:
            raise Unscorable("max_nodes_must_be_positive")
        schema = _normalize_schema(schema)
        # Column order affects SELECT * expansion, so it is part of the fingerprint.
        fingerprint = [(table, list(schema[table].items())) for table in sorted(schema)]
        result["schema_hash"] = hashlib.sha256(
            json.dumps(fingerprint, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        expressions, warnings = [], []
        for side, sql in [("pred", pred_sql), ("gold", gold_sql)]:
            stage = side
            expression, qualified_sql, notices = _prepare(
                sql, schema, max_nodes, sqlite_dqs
            )
            expressions.append(expression)
            warnings.extend(notices)
            result[side + "_qualified_sql"] = qualified_sql
        stage = "normalization"
        if _canonicalize_matched_inner_joins(*expressions):
            warnings.append("matched_simple_inner_joins_canonicalized")
        # SQLite may apply different collations to a=b and b=a. Capture
        # operand order after paired join canonicalization but before optimize.
        comparison_signatures = [_ordered_column_comparisons(expression)
                                 for expression in expressions]
        trees, sizes = [], []
        for side, expression in zip(("pred", "gold"), expressions):
            stage = side + "_optimizer"
            expression = optimize(expression, schema=schema, dialect="sqlite")
            if sum(1 for _ in expression.walk()) > max_nodes * 4:
                raise Unscorable("optimized_node_limit")
            if any(isinstance(item, exp.Func) for item in expression.walk()):
                warnings.append("function_semantics_not_proven; conservative_reordering_only")
            if any(isinstance(item, exp.Join) for item in expression.walk()):
                warnings.append("join_semantics_require_review")
            tree = TreeBuilder(expression).build(expression)
            sizes.append(_index(tree, max_nodes))
            trees.append(tree)
            result[side + "_optimized_sql"] = expression.sql(dialect="sqlite")
            result[side + "_tree_hash"] = hashlib.sha256(repr(tree.key()).encode()).hexdigest()
            if include_trees:
                result[side + "_tree"] = tree.as_dict()
        stage = "distance"
        apted = APTED(trees[0], trees[1], Config())
        distance = apted.compute_edit_distance()
        orientation_conflict = comparison_signatures[0] != comparison_signatures[1]
        if orientation_conflict and distance == 0:
            raise Unscorable("optimizer_erased_column_comparison_orientation")
        if orientation_conflict:
            warnings.append("preoptimizer_column_comparisons_differ; inspect_sqlite_collation")
        edits = []
        regions = {"pred": set(), "gold": set()}
        for source, target in apted.compute_edit_mapping():
            if source is not None and target is not None and source.name == target.name:
                continue
            op = "insert" if source is None else "delete" if target is None else "replace"
            edit = {"op": op, "cost": 1}
            for side, node in [("pred", source), ("gold", target)]:
                edit[side] = None if node is None else {
                    "path": node.path, "label": json.loads(node.name), "region": node.region,
                }
                if node is not None:
                    regions[side].add(node.region)
            edits.append(edit)
        if len(edits) != distance:
            raise RuntimeError("APTED mapping cost differs from computed distance")
        result.update(
            status="ok", ted=distance, similarity=max(0.0, 1.0 - distance / max(sizes)),
            pred_node_count=sizes[0], gold_node_count=sizes[1],
            edit_counts={op: sum(e["op"] == op for e in edits)
                         for op in ("insert", "delete", "replace")},
            edit_script=edits, changed_regions={side: sorted(v) for side, v in regions.items()},
            warnings=sorted(set(warnings)),
            preoptimizer_comparison_orientation_conflict=orientation_conflict,
        )
    except (Unscorable, SqlglotError, RecursionError) as error:
        result.update(error_stage=stage, error=f"{type(error).__name__}: {error}")
    return result


def schema_from_sqlite(path: str | Path) -> dict:
    """Read table metadata in read-only mode. Does not run prediction or gold."""
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        names = connection.execute(
            "SELECT name FROM sqlite_schema WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        result = {}
        for (name,) in names:
            columns = connection.execute(
                "SELECT name FROM pragma_table_xinfo(?) WHERE hidden != 1 ORDER BY cid", (name,)
            ).fetchall()
            # Qualification only needs names. Avoid rigid-type optimizer assumptions
            # about SQLite's dynamic typing or unusual declared type spellings.
            result[name] = {column: "UNKNOWN" for (column,) in columns}
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    schema_group = parser.add_mutually_exclusive_group(required=True)
    schema_group.add_argument("--schema", type=Path, help="JSON {table:{column:type}}")
    schema_group.add_argument("--db", type=Path, help="SQLite file, metadata read only")
    parser.add_argument("--pred")
    parser.add_argument("--gold")
    parser.add_argument("--input", type=Path, help="JSONL rows with pred_sql and gold_sql")
    parser.add_argument("--output", type=Path, help="New output JSONL file; never overwrite")
    parser.add_argument("--include-trees", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=600)
    parser.add_argument("--strict-quotes", action="store_true",
                        help="Disable SQLite legacy double-quoted string fallback")
    args = parser.parse_args()
    if args.input and (args.pred is not None or args.gold is not None):
        parser.error("use --input OR --pred/--gold")
    if not args.input and (args.pred is None or args.gold is None):
        parser.error("provide --input or both --pred and --gold")
    if args.max_nodes < 1:
        parser.error("--max-nodes must be positive")
    schema = (json.loads(args.schema.read_text(encoding="utf-8-sig")) if args.schema
              else schema_from_sqlite(args.db))
    output = None
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.open("x", encoding="utf-8")

        def emit(row: dict) -> None:
            encoded = json.dumps(row, ensure_ascii=False)
            if output:
                output.write(encoded + "\n")
            else:
                print(encoded)

        def score(row: dict) -> dict:
            value = sql_similarity(row.get("pred_sql"), row.get("gold_sql"), schema,
                                   max_nodes=args.max_nodes, include_trees=args.include_trees,
                                   sqlite_dqs=not args.strict_quotes)
            for key in ("task_id", "attempt_id", "db_id"):
                if key in row:
                    value[key] = row[key]
            return value

        if args.input:
            with args.input.open(encoding="utf-8-sig") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("row must be a JSON object")
                    except ValueError as error:
                        emit({"status": "unscorable", "similarity": None, "ted": None,
                              "line_number": line_number, "error": str(error)})
                        continue
                    emit(score(row))
        else:
            emit(score({"pred_sql": args.pred, "gold_sql": args.gold}))
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
