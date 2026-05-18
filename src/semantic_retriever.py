"""MiniLM semantic retrieval for the StackLite question corpus."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any

from src.bm25_retriever import Document, make_snippet, tokenize


DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbedder:
    """Lazy wrapper around SentenceTransformers so tests do not download a model."""

    def __init__(self, model_name: str = DEFAULT_MINILM_MODEL) -> None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "MiniLM semantic search requires sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> Sequence[Sequence[float]]:
        """Encode texts with normalized MiniLM sentence embeddings."""

        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class SemanticRetriever:
    """Dense retriever that ranks documents by cosine similarity."""

    def __init__(
        self,
        embedder: Any | None = None,
        model_name: str = DEFAULT_MINILM_MODEL,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.embedder = embedder
        self.model_name = model_name
        self.batch_size = batch_size
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []

    def fit(self, documents: Sequence[Document]) -> "SemanticRetriever":
        """Build a dense embedding index over the document corpus."""

        self.documents = list(documents)
        if not self.documents:
            raise ValueError("Cannot fit SemanticRetriever on an empty document list")

        texts = [semantic_document_text(document) for document in self.documents]
        self.embeddings = self._encode(texts)
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
        """Return top-k semantic matches for a user query."""

        if not self.documents or not self.embeddings:
            raise RuntimeError("SemanticRetriever must be fit before calling search")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self._encode([query])[0]
        scores = [dot_product(query_embedding, embedding) for embedding in self.embeddings]
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (
                scores[index],
                self.documents[index].score,
                self.documents[index].answer_count,
            ),
            reverse=True,
        )[:top_k]

        query_tokens = set(tokenize(query))
        results: list[dict[str, object]] = []
        for rank, doc_index in enumerate(ranked_indices, start=1):
            document = self.documents[doc_index]
            results.append(
                {
                    "rank": rank,
                    "score": round(scores[doc_index], 6),
                    "doc_id": document.doc_id,
                    "source": document.source,
                    "question_id": document.question_id,
                    "title": document.title,
                    "tags": ", ".join(document.tags),
                    "link": document.link,
                    "snippet": make_snippet(semantic_document_text(document), query_tokens),
                    "stack_score": document.score,
                    "answer_count": document.answer_count,
                    "view_count": document.view_count,
                }
            )
        return results

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self.embedder is None:
            self.embedder = MiniLMEmbedder(self.model_name)

        raw_embeddings = self.embedder.encode(list(texts), batch_size=self.batch_size)
        return [normalize_vector(vector) for vector in raw_embeddings]


def semantic_document_text(document: Document) -> str:
    """Create a natural text field for dense retrieval."""

    tags_text = ", ".join(document.tags)
    return f"Title: {document.title}\nBody: {document.body}\nTags: {tags_text}".strip()


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Return a unit-length list representation of an embedding vector."""

    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute the dot product between two same-length vectors."""

    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return sum(float(a) * float(b) for a, b in zip(left, right))
