from __future__ import annotations

import hashlib
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)


def tokenizer_fingerprint(model_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in TOKENIZER_FILES:
        path = model_dir / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise FileNotFoundError(f"no tokenizer files found in {model_dir}")
    return result


def assert_tokenizers_identical(student_model: Path, teacher_model: Path) -> dict[str, str]:
    student = tokenizer_fingerprint(student_model)
    teacher = tokenizer_fingerprint(teacher_model)
    if student != teacher:
        differing = sorted(set(student) | set(teacher))
        differing = [name for name in differing if student.get(name) != teacher.get(name)]
        raise ValueError(f"student/teacher tokenizer mismatch: {', '.join(differing)}")
    return student
