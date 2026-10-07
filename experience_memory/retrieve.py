from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable


def terms(text: str) -> Counter:
    result = Counter(re.findall(r"[a-z_][a-z_0-9]*", text.lower()))
    for span in re.findall(r"[\u4e00-\u9fff]+", text):
        result.update(span[i : i + 2] for i in range(len(span) - 1))
        if len(span) == 1:
            result[span] += 1
    # Bridge common English questions and Chinese teacher experiences.
    aliases = {
        "count": ("count", "number", "many", "数量", "计数", "统计"),
        "join": ("join", "连接", "关联"),
        "distinct": ("distinct", "unique", "去重", "唯一"),
        "group": ("group", "each", "每个", "分组", "聚合"),
        "max": ("max", "highest", "most", "最大", "最高"),
        "min": ("min", "lowest", "least", "最小", "最低"),
        "order": ("order", "sort", "排序"),
    }
    for concept, words in aliases.items():
        if any(result[word] for word in words):
            result["concept:" + concept] += 1
    return result


def retrieve(query: str, memories: list[dict], top_k: int = 3) -> list[dict]:
    """Deterministic TF-IDF cosine search over experience text only."""
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if not memories or top_k == 0:
        return []
    documents = [terms(memory["experience"]) for memory in memories]
    frequencies = Counter(term for document in documents for term in document)
    weights = {
        term: math.log((1 + len(documents)) / (1 + count)) + 1
        for term, count in frequencies.items()
    }

    def vector(document):
        return {
            term: (1 + math.log(count)) * weights[term]
            for term, count in document.items()
            if term in weights
        }

    query_vector = vector(terms(query))
    query_norm = math.sqrt(sum(weight**2 for weight in query_vector.values()))
    if not query_norm:
        return []
    ranked = []
    for index, document in enumerate(documents):
        values = vector(document)
        norm = math.sqrt(sum(weight**2 for weight in values.values()))
        score = sum(weight * values.get(term, 0) for term, weight in query_vector.items())
        if norm and score > 0:
            ranked.append((score / (norm * query_norm), index))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [memories[index] for _, index in ranked[:top_k]]


HEADER = (
    "历史通用经验（仅供参考，请判断适用条件）：\n"
    "SQL 示例来自历史题目，表名和字段名需映射到当前 schema；不要直接照抄。"
)


def render_context(
    memories: Iterable[dict], *, max_tokens: int, count_tokens: Callable[[str], int]
) -> str:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    blocks = []
    for memory in memories:
        block = (
            f"经验：{memory['experience']}\n"
            f"修改前 SQL：{memory['sql_before']}\n修改后 SQL：{memory['sql_after']}"
        )
        candidate = HEADER + "\n\n" + "\n\n".join([*blocks, block])
        if count_tokens(candidate) <= max_tokens:
            blocks.append(block)
    return HEADER + "\n\n" + "\n\n".join(blocks) if blocks else ""


def contexts_for_tasks(
    tasks, memories: list[dict], *, top_k: int, max_tokens: int, count_tokens: Callable[[str], int]
) -> dict[str, str]:
    return {
        task.task_id: render_context(
            retrieve(task.question, memories, top_k),
            max_tokens=max_tokens,
            count_tokens=count_tokens,
        )
        for task in tasks
    }
