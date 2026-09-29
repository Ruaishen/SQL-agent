from __future__ import annotations

from data.preprocess_spider import _split_databases
from third_party.spider_eval.hardness import eval_hardness


def test_database_split_is_deterministic_and_disjoint() -> None:
    db_ids = {f"db_{index:03d}" for index in range(146)}
    first = _split_databases(db_ids, 42)
    second = _split_databases(db_ids, 42)
    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 102,
        "internal_validation": 22,
        "internal_holdout": 22,
    }
    assert not (set(first["train"]) & set(first["internal_validation"]))
    assert not (set(first["train"]) & set(first["internal_holdout"]))


def test_official_hardness_minimal_query_is_easy() -> None:
    parsed = {
        "select": [False, [[0, [0, [0, 1, False], None]]]],
        "from": {"table_units": [["table_unit", 0]], "conds": []},
        "where": [],
        "groupBy": [],
        "orderBy": [],
        "having": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }
    assert eval_hardness(parsed) == "easy"
