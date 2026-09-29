import json
import sqlite3
import subprocess
import sys

import pytest

from evaluation.sql_similarity import schema_from_sqlite, sql_similarity

SCHEMA = {"student": {"id": "INT", "name": "TEXT", "age": "INT", "manager_id": "INT"},
          "class": {"id": "INT", "name": "TEXT"}}


def score(a, b, **kwargs):
    result = sql_similarity(a, b, SCHEMA, **kwargs)
    assert result["status"] == "ok", result
    assert sum(result["edit_counts"].values()) == result["ted"]
    assert len(result["edit_script"]) == result["ted"]
    return result


@pytest.mark.parametrize("a,b", [
    ("SELECT s.name FROM student s", " select stu.name from STUDENT AS stu; -- hi"),
    ("SELECT name FROM student", 'SELECT "NAME" FROM "STUDENT"'),
    ("SELECT name FROM student WHERE age>18 AND id=1",
     "SELECT name FROM student WHERE id=1 AND (age>18)"),
    ("SELECT name FROM student WHERE (age>18 AND id=1) AND manager_id=2",
     "SELECT name FROM student WHERE manager_id=2 AND (id=1 AND age>18)"),
    ("SELECT name FROM student WHERE id=1 OR id=2", "SELECT name FROM student WHERE id=2 OR id=1"),
    ("SELECT name FROM student WHERE age>18", "SELECT name FROM student WHERE 18<age"),
    ("SELECT name FROM student WHERE age<>18", "SELECT name FROM student WHERE age!=18"),
    ("SELECT name FROM student WHERE id IN (1,2,NULL)",
     "SELECT name FROM student WHERE id IN (NULL,2,1)"),
    ("SELECT name FROM student ORDER BY age", "SELECT name FROM student ORDER BY age ASC"),
    ("SELECT name AS a FROM student ORDER BY a", "SELECT name AS b FROM student ORDER BY b"),
    ("SELECT name AS a FROM student ORDER BY a", "SELECT name FROM student ORDER BY name"),
    ("SELECT name FROM student ORDER BY 1", "SELECT name FROM student ORDER BY name"),
    ("SELECT x.n FROM (SELECT name AS n FROM student) x",
     "SELECT y.m FROM (SELECT name AS m FROM student) y"),
    ("WITH x AS (SELECT name FROM student) SELECT name FROM x",
     "WITH y AS (SELECT name FROM student) SELECT name FROM y"),
    ("SELECT a.name FROM student a JOIN student b ON a.manager_id=b.id",
     "SELECT x.name FROM student x JOIN student y ON x.manager_id=y.id"),
    ("SELECT s.name FROM student s JOIN class c ON s.id=c.id WHERE c.name='A'",
     "SELECT s.name FROM class c JOIN student s ON s.id=c.id AND c.name='A'"),
    ("SELECT a.name FROM student a WHERE EXISTS (SELECT 1 FROM student b WHERE b.id=a.manager_id)",
     "SELECT x.name FROM student x WHERE EXISTS (SELECT 1 FROM student y WHERE y.id=x.manager_id)"),
    ("SELECT * FROM student", "SELECT id,name,age,manager_id FROM student"),
    ("SELECT student.* FROM student", "SELECT s.* FROM student s"),
    ('SELECT name FROM student WHERE name="Rylan"', "SELECT name FROM student WHERE name='Rylan'"),
    ('SELECT "Rylan" FROM student', "SELECT 'Rylan' FROM student"),
    ('SELECT "Rylan" AS Rylan FROM student', "SELECT 'Rylan' AS Rylan FROM student"),
])
def test_supported_equivalence(a, b):
    r = score(a, b)
    assert r["ted"] == 0, r
    assert r["similarity"] == 1
    assert r["pred_tree_hash"] == r["gold_tree_hash"]


@pytest.mark.parametrize("a,b", [
    ("SELECT name FROM student WHERE age>18", "SELECT name FROM student WHERE age>20"),
    ("SELECT name FROM student", "SELECT age FROM student"),
    ("SELECT name FROM student WHERE age>18", "SELECT name FROM student WHERE age>=18"),
    ("SELECT name FROM student WHERE name='A'", "SELECT name FROM student WHERE name='a'"),
])
def test_single_replacement(a, b):
    r = score(a, b)
    assert r["ted"] == 1
    assert r["edit_counts"] == {"insert": 0, "delete": 0, "replace": 1}
    assert r["similarity"] == 1 - 1 / max(r["pred_node_count"], r["gold_node_count"])


@pytest.mark.parametrize("a,b", [
    ("SELECT name FROM student", "SELECT DISTINCT name FROM student"),
    ("SELECT COUNT(*) FROM student", "SELECT COUNT(id) FROM student"),
    ("SELECT COUNT(id) FROM student", "SELECT COUNT(DISTINCT id) FROM student"),
    ("SELECT MAX(age) FROM student", "SELECT MIN(age) FROM student"),
    ("SELECT CAST(age AS INT) FROM student", "SELECT CAST(age AS TEXT) FROM student"),
    ("SELECT name FROM student LIMIT 1", "SELECT name FROM student LIMIT 2"),
    ("SELECT name FROM student ORDER BY age ASC", "SELECT name FROM student ORDER BY age DESC"),
    ("SELECT name,age FROM student", "SELECT age,name FROM student"),
    ("SELECT name FROM student ORDER BY age,id", "SELECT name FROM student ORDER BY id,age"),
    ("SELECT name FROM student WHERE age=18", "SELECT name FROM student WHERE age='18'"),
    ("SELECT name FROM student WHERE age=18", "SELECT name FROM student WHERE NOT age=18"),
    ("SELECT a.name FROM student a JOIN student b ON a.manager_id=b.id",
     "SELECT b.name FROM student a JOIN student b ON a.manager_id=b.id"),
    ("SELECT a.name FROM student a JOIN class b ON a.id=b.id",
     "SELECT a.name FROM student a LEFT JOIN class b ON a.id=b.id"),
    ("SELECT * FROM student s JOIN class c ON s.id=c.id",
     "SELECT * FROM class c JOIN student s ON s.id=c.id"),
    ("SELECT s.name FROM student s JOIN class c ON s.id=c.id LIMIT 1",
     "SELECT s.name FROM class c JOIN student s ON s.id=c.id LIMIT 1"),
    ("SELECT name FROM student WHERE name COLLATE NOCASE='a'",
     "SELECT name FROM student WHERE name COLLATE BINARY='a'"),
    ("SELECT a.name FROM student a WHERE EXISTS (SELECT 1 FROM student b WHERE b.id=a.manager_id)",
     "SELECT a.name FROM student a WHERE EXISTS (SELECT 1 FROM student b WHERE b.id=b.manager_id)"),
    ("SELECT name FROM student UNION SELECT name FROM class",
     "SELECT name FROM student UNION ALL SELECT name FROM class"),
    ("SELECT ROW_NUMBER() OVER (ORDER BY age) FROM student",
     "SELECT ROW_NUMBER() OVER (ORDER BY age DESC) FROM student"),
    ("SELECT age, COUNT(*) FROM student GROUP BY age HAVING COUNT(*)>1",
     "SELECT age, COUNT(*) FROM student WHERE age>1 GROUP BY age"),
])
def test_real_differences_and_symmetry(a, b):
    forward = score(a, b)
    backward = score(b, a)
    assert forward["ted"] > 0
    assert forward["ted"] == backward["ted"]
    assert forward["similarity"] == backward["similarity"]
    assert forward["edit_counts"]["insert"] == backward["edit_counts"]["delete"]


@pytest.mark.parametrize("sql", [
    "", "SELECT FROM", "SELECT 1; SELECT 2", "DELETE FROM student",
    "SELECT missing FROM student", "SELECT COUNT(*) FROM unknown_table",
    "SELECT [missing] FROM student", "SELECT `missing` FROM student",
    "SELECT name FROM student JOIN class ON student.id=class.id",
    "WITH RECURSIVE x AS (SELECT 1 UNION ALL SELECT 1 FROM x) SELECT * FROM x",
])
def test_unscorable(sql):
    r = sql_similarity(sql, "SELECT name FROM student", SCHEMA)
    assert r["status"] == "unscorable", r
    assert r["similarity"] is None
    assert r["ted"] is None


def test_size_limit():
    r = sql_similarity("SELECT name FROM student", "SELECT name FROM student", SCHEMA, max_nodes=2)
    assert r["status"] == "unscorable"


def test_dqs_setting_and_identifier_precedence():
    a = 'SELECT name FROM student WHERE name="Rylan"'
    strict = sql_similarity(a, a, SCHEMA, sqlite_dqs=False)
    assert strict["status"] == "unscorable"
    assert score('SELECT "name" FROM student', "SELECT 'name' FROM student")["ted"] > 0


def test_normalization_with_nulls_and_mixed_values():
    a = "SELECT id FROM student WHERE age>18 AND id IN (1,2,NULL)"
    b = "SELECT id FROM student WHERE id IN (NULL,2,1) AND 18<age"
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE student(id, name, age, manager_id)")
        db.executemany("INSERT INTO student VALUES (?,NULL,?,NULL)",
                       [(1, 19), (2, None), (3, 30), (None, 20), (2, '21')])
        assert db.execute(a).fetchall() == db.execute(b).fetchall()
    assert score(a, b)["ted"] == 0


def test_determinism_and_trees():
    args = ("SELECT name FROM student WHERE age>18", "SELECT name FROM student WHERE age>19")
    first = score(*args, include_trees=True)
    assert first == score(*args, include_trees=True)
    assert first["pred_tree"] != first["gold_tree"]
    assert first["changed_regions"] == {"pred": ["q0:WHERE"], "gold": ["q0:WHERE"]}
    second = score(first["pred_qualified_sql"], first["gold_qualified_sql"])
    assert second["pred_tree_hash"] == first["pred_tree_hash"]
    assert second["gold_tree_hash"] == first["gold_tree_hash"]


def test_collation_counterexample():
    schema = {"t": {"a": "TEXT", "b": "TEXT"}}
    a, b = "SELECT a FROM t WHERE a=b", "SELECT a FROM t WHERE b=a"
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE t(a TEXT COLLATE BINARY, b TEXT COLLATE NOCASE)")
        db.execute("INSERT INTO t VALUES ('a', 'A')")
        assert db.execute(a).fetchall() != db.execute(b).fetchall()
    result = sql_similarity(a, b, schema)
    assert result["status"] == "unscorable"
    assert "optimizer_erased_column_comparison_orientation" in result["error"]


@pytest.mark.parametrize("a,b", [
    ("SELECT name FROM student WHERE age>18 AND age>18",
     "SELECT name FROM student WHERE age>18"),
    ("SELECT name FROM student WHERE age>18 AND 1=1",
     "SELECT name FROM student WHERE age>18"),
    ("WITH x AS (SELECT name FROM student) SELECT name FROM x",
     "SELECT name FROM student"),
    ("SELECT x.n FROM (SELECT name AS n FROM student) x",
     "SELECT name FROM student"),
])
def test_full_optimizer_rewrites_before_ted(a, b):
    result = score(a, b)
    assert result["ted"] == 0, result
    assert result["pred_qualified_sql"] != result["pred_optimized_sql"]
    assert result["optimizer"]["name"] == "sqlglot.optimize"


def test_optimizer_does_not_hide_real_errors():
    bad = score("SELECT name FROM student WHERE age>18",
                "SELECT name FROM student WHERE age>20")
    assert bad["ted"] > 0
    assert bad["similarity"] < 1


def test_join_canonicalization_requires_both_sides_to_match():
    mismatched_tables = score(
        "SELECT s.name FROM student s JOIN class c ON s.id=c.id",
        "SELECT s.name FROM student s JOIN student t ON s.manager_id=t.id",
    )
    assert 'FROM "student"' in mismatched_tables["pred_optimized_sql"]
    outer_vs_inner = score(
        "SELECT s.name FROM student s LEFT JOIN class c ON s.id=c.id",
        "SELECT s.name FROM student s JOIN class c ON s.id=c.id",
    )
    assert 'FROM "student"' in outer_vs_inner["gold_optimized_sql"]
    assert "matched_simple_inner_joins_canonicalized" not in outer_vs_inner["warnings"]


def test_schema_fingerprint_preserves_column_order():
    a = sql_similarity("SELECT * FROM t", "SELECT * FROM t", {"t": {"a": "INT", "b": "INT"}})
    b = sql_similarity("SELECT * FROM t", "SELECT * FROM t", {"t": {"b": "INT", "a": "INT"}})
    assert a["schema_hash"] != b["schema_hash"]
    assert a["pred_tree_hash"] != b["pred_tree_hash"]


def test_metadata_and_cli(tmp_path):
    db = tmp_path / "sample.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE student(id INTEGER, name TEXT)")
        connection.execute("INSERT INTO student VALUES (1,'A')")
    before = db.read_bytes()
    schema = schema_from_sqlite(db)
    assert schema == {"student": {"id": "UNKNOWN", "name": "UNKNOWN"}}
    source, output = tmp_path / "input.jsonl", tmp_path / "scores.jsonl"
    source.write_text(json.dumps({"task_id": "x", "pred_sql": "SELECT id FROM student",
                                  "gold_sql": "SELECT name FROM student"}) + '\nnot json\n',
                      encoding="utf-8")
    command = [sys.executable, "-m", "evaluation.sql_similarity", "--db", str(db),
               "--input", str(source), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["task_id"] == "x" and rows[0]["ted"] == 1
    assert rows[1]["status"] == "unscorable"
    assert db.read_bytes() == before
    assert subprocess.run(command, capture_output=True).returncode != 0
