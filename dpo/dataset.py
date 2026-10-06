"""Load the complete accepted subset, bound to its saved blind audit evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pair(pair: dict) -> None:
    task = pair.get("task_id")
    fork = pair.get("fork_turn")
    if not isinstance(task, str) or pair.get("split") != "train":
        raise ValueError("Expected a training pair with task_id")
    if type(fork) is not int or fork < 1 or pair.get("loss_from_turn") != fork:
        raise ValueError(f"Invalid fork or loss boundary: {task}")
    if pair.get("chosen_verification", {}).get("correct") is not True:
        raise ValueError(f"Chosen is not database verified: {task}")
    if (pair.get("rejected_verification") or {}).get("correct") is True:
        raise ValueError(f"Rejected trajectory is correct: {task}")
    prefix = pair.get("prompt_messages")
    for branch in ("chosen_messages", "rejected_messages"):
        messages = pair[branch]
        if (not isinstance(prefix, list) or len(prefix) != 2 * fork
                or messages[:2 * fork] != prefix or len(messages) <= 2 * fork
                or messages[0]["role"] != "system" or messages[1]["role"] != "user"
                or any(m["role"] != ("assistant" if i % 2 == 0 else "user")
                       for i, m in enumerate(messages[2:], 2))):
            raise ValueError(f"Shared prompt or assistant fork changed: {task}")
    if pair["chosen_messages"] == pair["rejected_messages"]:
        raise ValueError(f"Identical preference branches: {task}")


def load_audited_pairs(pairs_jsonl: Path, audit_roots: list[Path]) -> tuple[list[dict], dict]:
    # Import only for this strict mode; legacy tokenization keeps its dependencies.
    from dpo.audit import audit_input, validate_review

    if not audit_roots:
        raise ValueError("Audited JSONL requires --audit-roots")
    accepted = {}
    evidence = []
    for root in audit_roots:
        report_path = root / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") != "completed" or report.get("audit_errors_pending", 0)
                or report.get("errors", 0) or report.get("tasks_remaining", 0)):
            raise ValueError(f"Audit run is incomplete: {root}")
        paths = sorted((root / "accepted/pairs").glob("*.json"))
        expected = report.get("new_accepted", report.get("accepted_files"))
        if expected is None or len(paths) != expected:
            raise ValueError(f"Accepted files disagree with audit report: {root}")
        evidence.append({"root": str(root.resolve()), "report_sha256": sha256(report_path)})
        for path in paths:
            pair = json.loads(path.read_text(encoding="utf-8"))
            validate_pair(pair)
            if pair.get("sanitized_reasoning_turns"):
                raise ValueError(f"Strict dataset contains edited reasoning: {path}")
            if "new_accepted" in report:
                attempt = pair.get("retry_provenance", {}).get("attempt")
                if type(attempt) is not int or attempt not in (1, 2):
                    raise ValueError(f"Invalid retry provenance: {path}")
                review_path = root / "attempts" / f"{attempt:02d}" / "reviews" / path.name
            else:
                review_path = root / "reviews" / path.name
            review = json.loads(review_path.read_text(encoding="utf-8"))
            data = audit_input(pair)
            validate_review(review["review"], data)
            if (review.get("task_id") != pair["task_id"] or review.get("decision") != "accept"
                    or review.get("issue_codes") != []
                    or review["review"]["verdict"] != "accept"
                    or data["mechanical_checks"]["issues"]
                    or review.get("mechanical_checks") != data["mechanical_checks"]):
                raise ValueError(f"Pair does not match a clean accepted audit: {path}")
            task = pair["task_id"]
            if task in accepted:
                raise ValueError(f"Duplicate accepted task: {task}")
            accepted[task] = pair
            evidence.append({"task_id": task, "pair_sha256": sha256(path),
                             "review_sha256": sha256(review_path)})
    pairs = []
    seen = set()
    for line in pairs_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        task = pair.get("task_id")
        if task in seen:
            raise ValueError(f"Duplicate JSONL task: {task}")
        if task not in accepted or pair != accepted[task]:
            raise ValueError(f"JSONL pair is absent from or changed since accepted audit: {task}")
        seen.add(task)
        pairs.append(pair)
    if not pairs or seen != set(accepted):
        raise ValueError("JSONL does not cover the complete accepted subset")
    return pairs, {"source_kind": "audited_jsonl", "pairs_jsonl": str(pairs_jsonl.resolve()),
                   "pairs_jsonl_sha256": sha256(pairs_jsonl), "audit_evidence": evidence}
