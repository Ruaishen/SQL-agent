"""Spider 1.0 official SQL hardness calculation.

This is the isolated hardness portion of taoyds/spider/evaluation.py. Keeping
it independent avoids importing the legacy evaluator's CLI and parser globals.
"""

from __future__ import annotations

from typing import Any

UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")


def _nested_sql(sql: dict[str, Any]) -> list[dict[str, Any]]:
    nested: list[dict[str, Any]] = []
    conditions = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    for condition in conditions:
        if isinstance(condition[3], dict):
            nested.append(condition[3])
        if isinstance(condition[4], dict):
            nested.append(condition[4])
    for operator in ("intersect", "except", "union"):
        if sql[operator] is not None:
            nested.append(sql[operator])
    return nested


def _has_aggregation(unit: Any) -> bool:
    return unit[0] != AGG_OPS.index("none")


def _count_component1(sql: dict[str, Any]) -> int:
    count = int(bool(sql["where"]))
    count += int(bool(sql["groupBy"]))
    count += int(bool(sql["orderBy"]))
    count += int(sql["limit"] is not None)
    count += max(0, len(sql["from"]["table_units"]) - 1)
    connectors = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += sum(token == "or" for token in connectors)
    conditions = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    like_id = WHERE_OPS.index("like")
    count += sum(condition[1] == like_id for condition in conditions)
    return count


def _count_others(sql: dict[str, Any]) -> int:
    agg_count = sum(_has_aggregation(unit) for unit in sql["select"][1])
    agg_count += sum(_has_aggregation(unit) for unit in sql["where"][::2])
    agg_count += sum(_has_aggregation(unit) for unit in sql["groupBy"])
    if sql["orderBy"]:
        units = sql["orderBy"][1]
        agg_count += sum(_has_aggregation(value) for unit in units for value in unit[1:] if value)
    agg_count += sum(_has_aggregation(unit) for unit in sql["having"])
    count = int(agg_count > 1)
    count += int(len(sql["select"][1]) > 1)
    count += int(len(sql["where"]) > 1)
    count += int(len(sql["groupBy"]) > 1)
    return count


def eval_hardness(sql: dict[str, Any]) -> str:
    component1 = _count_component1(sql)
    component2 = len(_nested_sql(sql))
    others = _count_others(sql)
    if component1 <= 1 and others == 0 and component2 == 0:
        return "easy"
    if (others <= 2 and component1 <= 1 and component2 == 0) or (
        component1 <= 2 and others < 2 and component2 == 0
    ):
        return "medium"
    if (
        (others > 2 and component1 <= 2 and component2 == 0)
        or (2 < component1 <= 3 and others <= 2 and component2 == 0)
        or (component1 <= 1 and others == 0 and component2 <= 1)
    ):
        return "hard"
    return "extra"
