from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = (
    "Given a natural-language SQL task, retrieve general SQL reasoning experiences "
    "whose applicability conditions and advice help solve the task."
)
SQL_QUERY_INSTRUCTION = (
    "Given a SQL query, retrieve SQL queries with similar structure and logic."
)


class EmbeddingEncoder:
    """Frozen Qwen3 embedding model; lazy loading keeps empty-memory runs lightweight."""

    def __init__(
        self, model: str = DEFAULT_EMBEDDING_MODEL, *, device: str = "cpu",
        batch_size: int = 8, max_length: int = 2048,
        query_instruction: str = QUERY_INSTRUCTION,
    ):
        if batch_size < 1 or max_length < 1:
            raise ValueError("Invalid embedding limits")
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_instruction = query_instruction
        self._model = None
        self._tokenizer = None

    def encode(self, texts: list[str], *, is_query: bool):
        import torch
        import torch.nn.functional as F

        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float32 if torch.device(self.device).type == "cpu" else torch.float16
                ),
            ).to(self.device).eval()
            self._model.requires_grad_(False)
        if is_query:
            texts = [f"Instruct: {self.query_instruction}\nQuery:{text}" for text in texts]
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = self._tokenizer(
                    texts[start:start + self.batch_size], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                # Official Qwen3 recipe: left padding and last-token pooling.
                hidden = self._model(**batch).last_hidden_state[:, -1]
                embeddings.append(F.normalize(hidden.float(), p=2, dim=1).cpu())
        return torch.cat(embeddings)


class EmbeddingRetriever:
    """Encode source questions once; retrieve using cosine similarity."""

    def __init__(self, memories: list[dict], encoder=None, *, text_field: str = "question"):
        self.memories = memories
        self.text_field = text_field
        self.encoder = encoder if encoder is not None else EmbeddingEncoder()
        self._vectors = None

    @staticmethod
    def _normalized(vectors, rows):
        import torch
        import torch.nn.functional as F

        values = torch.as_tensor(vectors, dtype=torch.float32).cpu()
        if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
            raise ValueError("Invalid embedding shape")
        if not torch.isfinite(values).all() or (values.norm(dim=1) == 0).any():
            raise ValueError("Invalid embedding values")
        return F.normalize(values, p=2, dim=1)

    def prepare(self):
        if self.memories and self._vectors is None:
            texts = [memory.get(self.text_field) for memory in self.memories]
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError(f"Memory needs a nonempty {self.text_field} field")
            self._vectors = self._normalized(
                self.encoder.encode(
                    texts, is_query=False
                ), len(self.memories),
            )

    def search(self, queries: list[str], top_k: int = 5, *, return_scores: bool = False):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not queries or not self.memories or top_k == 0:
            return [[] for _ in queries]
        import torch

        self.prepare()
        results = []
        for start in range(0, len(queries), self.encoder.batch_size):
            batch = queries[start:start + self.encoder.batch_size]
            vectors = self._normalized(self.encoder.encode(batch, is_query=True), len(batch))
            if vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("Query and memory embedding dimensions differ")
            scores = vectors @ self._vectors.T
            for row in scores:
                # Stable ties follow memory insertion order; keep previous positive-score rule.
                indices = torch.argsort(row, descending=True, stable=True).tolist()
                results.append([
                    {"memory": self.memories[index], "score": float(row[index])}
                    if return_scores else self.memories[index]
                    for index in indices[:top_k] if row[index] > 0
                ])
        return results


class TfidfRetriever:
    """Memory-only vocabulary/IDF; lowercase word and CJK character unigrams."""

    def __init__(self, memories: list[dict], *, text_field: str = "experience"):
        self.memories = memories
        self.text_field = text_field
        texts = [memory.get(text_field) for memory in memories]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError(f"Memory needs a nonempty {text_field} field")
        counts = [Counter(self.tokenize(text)) for text in texts]
        document_frequency = Counter(token for row in counts for token in row)
        self.idf = {
            token: 1 + math.log((1 + len(memories)) / (1 + frequency))
            for token, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(row) for row in counts]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]", text.lower())

    def _vector(self, counts: Counter) -> dict[str, float]:
        values = {token: count * self.idf[token] for token, count in counts.items()
                  if token in self.idf}
        norm = math.sqrt(sum(value * value for value in values.values()))
        return {token: value / norm for token, value in values.items()} if norm else {}

    def search(self, queries: list[str], top_k: int = 5, *, return_scores: bool = False) -> list[list[dict]]:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        results = []
        for query in queries:
            vector = self._vector(Counter(self.tokenize(query)))
            scores = [sum(value * memory.get(token, 0) for token, value in vector.items())
                      for memory in self.vectors]
            indices = sorted(range(len(scores)), key=lambda index: -scores[index])
            results.append([{"memory": self.memories[index], "score": scores[index]}
                            if return_scores else self.memories[index] for index in indices[:top_k]
                            if scores[index] > 0])
        return results


class BM25Retriever:
    """Okapi BM25 fitted only on the chosen frozen memory field."""

    def __init__(self, memories: list[dict], *, text_field: str = "sql_before",
                 k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.memories = memories
        self.text_field = text_field
        self.k1, self.b = k1, b
        texts = [memory.get(text_field) for memory in memories]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError(f"Memory needs a nonempty {text_field} field")
        self.counts = [Counter(TfidfRetriever.tokenize(text)) for text in texts]
        self.lengths = [sum(row.values()) for row in self.counts]
        self.avg_length = sum(self.lengths) / len(memories) if memories else 0.0
        frequencies = Counter(token for row in self.counts for token in row)
        self.idf = {
            token: math.log(1 + (len(memories) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in frequencies.items()
        }

    def search(self, queries: list[str], top_k: int = 5, *, return_scores: bool = False):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        results = []
        for query in queries:
            # Each distinct query term contributes once (no query-TF weighting).
            tokens = dict.fromkeys(TfidfRetriever.tokenize(query))
            scores = []
            for counts, length in zip(self.counts, self.lengths, strict=True):
                length_ratio = length / self.avg_length if self.avg_length else 0.0
                penalty = self.k1 * (1 - self.b + self.b * length_ratio)
                score = sum(
                    self.idf[token] * counts[token] * (self.k1 + 1) / (counts[token] + penalty)
                    for token in tokens if counts[token]
                )
                scores.append(score)
            indices = sorted(range(len(scores)), key=lambda index: -scores[index])
            results.append([
                {"memory": self.memories[index], "score": scores[index]}
                if return_scores else self.memories[index]
                for index in indices[:top_k] if scores[index] > 0
            ])
        return results


def make_retriever(memories: list[dict], *, retriever: str = "embedding", encoder=None):
    if retriever == "embedding":
        return EmbeddingRetriever(memories, encoder)
    if retriever == "embedding_experience":
        return EmbeddingRetriever(memories, encoder, text_field="experience")
    if retriever == "tfidf":
        return TfidfRetriever(memories)
    raise ValueError(f"Unknown memory retriever: {retriever}")


def retrieve(
    query: str, memories: list[dict], top_k: int = 5, *, encoder=None,
    retriever: str = "embedding",
) -> list[dict]:
    return make_retriever(memories, retriever=retriever, encoder=encoder).search([query], top_k)[0]


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
        # Keep the student's display identical to the previous combined format.
        experience_text = memory["experience"]
        if "question" in memory:
            experience_text = f"原始问题：{memory['question']}\n通用经验：{experience_text}"
        block = (
            f"经验：{experience_text}\n"
            f"修改前 SQL：{memory['sql_before']}\n修改后 SQL：{memory['sql_after']}"
        )
        candidate = HEADER + "\n\n" + "\n\n".join([*blocks, block])
        if count_tokens(candidate) <= max_tokens:
            blocks.append(block)
    return HEADER + "\n\n" + "\n\n".join(blocks) if blocks else ""


class SqlMemoryRetriever:
    """Retrieve current SQL against stored pre-correction SQLs."""

    def __init__(self, memories, encoder=None, *, top_k, max_tokens, count_tokens,
                 retriever="embedding", text_field="sql_before"):
        if text_field not in ("sql_before", "sql_first_execute"):
            raise ValueError(f"Unknown SQL memory field: {text_field}")
        self.text_field = text_field
        # An absent first execution is not replaced by the final failed SQL.
        indexed_memories = ([m for m in memories if m.get(text_field) is not None]
                            if text_field == "sql_first_execute" else memories)
        self.indexed_memory_count = len(indexed_memories)
        self.excluded_missing_sql_count = len(memories) - len(indexed_memories)
        if retriever == "embedding":
            self.retriever = EmbeddingRetriever(indexed_memories, encoder, text_field=text_field)
        elif retriever == "tfidf":
            self.retriever = TfidfRetriever(indexed_memories, text_field=text_field)
        elif retriever == "bm25":
            self.retriever = BM25Retriever(indexed_memories, text_field=text_field)
        else:
            raise ValueError(f"Unknown SQL memory retriever: {retriever}")
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens

    def prepare(self):
        if self.top_k and isinstance(self.retriever, EmbeddingRetriever):
            self.retriever.prepare()

    def __call__(self, sqls: list[str]) -> list[dict]:
        matches = self.retriever.search(sqls, self.top_k, return_scores=True)
        return [
            {
                "query_sql": sql,
                "memory_ids": [item["memory"]["memory_id"] for item in items],
                "scores": [item["score"] for item in items],
                "context": render_context(
                    [item["memory"] for item in items], max_tokens=self.max_tokens,
                    count_tokens=self.count_tokens,
                ),
            }
            for sql, items in zip(sqls, matches, strict=True)
        ]


def contexts_for_tasks(
    tasks, memories: list[dict], *, top_k: int, max_tokens: int,
    count_tokens: Callable[[str], int], encoder=None, retriever: str = "embedding",
) -> dict[str, str]:
    tasks = list(tasks)
    matches = make_retriever(memories, retriever=retriever, encoder=encoder).search(
        [task.question for task in tasks], top_k
    )
    return {
        task.task_id: render_context(found, max_tokens=max_tokens, count_tokens=count_tokens)
        for task, found in zip(tasks, matches, strict=True)
    }
