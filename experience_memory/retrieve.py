from __future__ import annotations

from collections.abc import Callable, Iterable

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = (
    "Given a natural-language SQL task, retrieve general SQL reasoning experiences "
    "whose applicability conditions and advice help solve the task."
)


class EmbeddingEncoder:
    """Frozen Qwen3 embedding model; lazy loading keeps empty-memory runs lightweight."""

    def __init__(
        self, model: str = DEFAULT_EMBEDDING_MODEL, *, device: str = "cpu",
        batch_size: int = 8, max_length: int = 2048,
    ):
        if batch_size < 1 or max_length < 1:
            raise ValueError("Invalid embedding limits")
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
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
            texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}" for text in texts]
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
    """Encode a memory snapshot once; search only experience text using cosine similarity."""

    def __init__(self, memories: list[dict], encoder=None):
        self.memories = memories
        self.encoder = encoder if encoder is not None else EmbeddingEncoder()
        self._vectors = None

    def search(self, queries: list[str], top_k: int = 10) -> list[list[dict]]:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not queries or not self.memories or top_k == 0:
            return [[] for _ in queries]
        import torch
        import torch.nn.functional as F

        def normalized(vectors, rows):
            values = torch.as_tensor(vectors, dtype=torch.float32).cpu()
            if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
                raise ValueError("Invalid embedding shape")
            if not torch.isfinite(values).all() or (values.norm(dim=1) == 0).any():
                raise ValueError("Invalid embedding values")
            return F.normalize(values, p=2, dim=1)

        if self._vectors is None:
            self._vectors = normalized(
                self.encoder.encode(
                    [memory["experience"] for memory in self.memories], is_query=False
                ), len(self.memories),
            )
        results = []
        for start in range(0, len(queries), self.encoder.batch_size):
            batch = queries[start:start + self.encoder.batch_size]
            vectors = normalized(self.encoder.encode(batch, is_query=True), len(batch))
            if vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("Query and memory embedding dimensions differ")
            scores = vectors @ self._vectors.T
            for row in scores:
                # Stable ties follow memory insertion order; keep previous positive-score rule.
                indices = torch.argsort(row, descending=True, stable=True).tolist()
                results.append([
                    self.memories[index] for index in indices[:top_k] if row[index] > 0
                ])
        return results


def retrieve(query: str, memories: list[dict], top_k: int = 10, *, encoder=None) -> list[dict]:
    return EmbeddingRetriever(memories, encoder).search([query], top_k)[0]


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
    tasks, memories: list[dict], *, top_k: int, max_tokens: int,
    count_tokens: Callable[[str], int], encoder=None,
) -> dict[str, str]:
    tasks = list(tasks)
    matches = EmbeddingRetriever(memories, encoder).search(
        [task.question for task in tasks], top_k
    )
    return {
        task.task_id: render_context(found, max_tokens=max_tokens, count_tokens=count_tokens)
        for task, found in zip(tasks, matches, strict=True)
    }
