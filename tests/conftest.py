from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from sql_agent.config import EnvConfig
from sql_agent.models import TaskRecord


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    db_dir = tmp_path / "database" / "sample"
    db_dir.mkdir(parents=True)
    path = db_dir / "sample.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE departments(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE employees(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER,
            salary REAL,
            note TEXT,
            FOREIGN KEY(department_id) REFERENCES departments(id)
        );
        CREATE TABLE invalid_text(value TEXT);
        INSERT INTO invalid_text VALUES (CAST(X'80' AS TEXT));
        CREATE TABLE binary_data(value BLOB);
        INSERT INTO binary_data VALUES (X'00FF');
        INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Sales');
        """
    )
    connection.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?)",
        [
            (
                index,
                f"employee-{index:02d}",
                1 if index % 2 else 2,
                1000.0 + index,
                "x" * 400 if index == 1 else "duplicate",
            )
            for index in range(1, 26)
        ],
    )
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def config(tmp_path: Path) -> EnvConfig:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    return replace(
        EnvConfig(),
        spider_root=tmp_path,
        processed_data_root=tmp_path / "processed",
        tokenizer_path=tokenizer_dir,
        query_timeout_seconds=0.05,
        verifier_timeout_seconds=0.2,
    )


@pytest.fixture
def task() -> TaskRecord:
    return TaskRecord(
        task_id="sample_00000",
        source="test",
        db_id="sample",
        db_path="database/sample/sample.sqlite",
        question="How many employees are there?",
        reference_sql="SELECT count(*) FROM employees",
        difficulty="easy",
        split="internal_validation",
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
