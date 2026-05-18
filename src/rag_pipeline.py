"""Retrieval-augmented generation pipeline with citation-grounded answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


DEFAULT_RRF_K = 60
DEFAULT_HF_RAG_MODEL = "google/flan-t5-small"


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
        """Return ranked retrieval results."""


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: Sequence[dict[str, object]]) -> str:
        """Generate a cited answer from retrieved contexts."""


@dataclass(frozen=True)
class RAGResponse:
    """A generated answer plus the contexts used to support it."""

    question: str
    answer: str
    contexts: list[dict[str, object]]

    @property
    def citations(self) -> list[dict[str, object]]:
        return [
            {
                "citation": f"[{index}]",
                "doc_id": context["doc_id"],
                "title": context["title"],
                "link": context["link"],
            }
            for index, context in enumerate(self.contexts, start=1)
        ]


class CitationAnswerGenerator:
    """Deterministic citation-grounded generator for reliable demos and tests."""

    def __init__(self, max_contexts: int = 3, snippet_chars: int = 260) -> None:
        if max_contexts <= 0:
            raise ValueError("max_contexts must be positive")
        if snippet_chars <= 0:
            raise ValueError("snippet_chars must be positive")
        self.max_contexts = max_contexts
        self.snippet_chars = snippet_chars

    def generate(self, question: str, contexts: Sequence[dict[str, object]]) -> str:
        if not contexts:
            return (
                "I could not find enough retrieved evidence in the StackLite corpus "
                "to answer this question with citations."
            )

        selected = list(contexts[: self.max_contexts])
        lead = (
            "The retrieved StackLite evidence points to the following cited source "
            f"{'passage' if len(selected) == 1 else 'passages'} as the best grounding for the answer."
        )
        evidence_sentences = []
        for index, context in enumerate(selected, start=1):
            snippet = compact_text(str(context.get("snippet", "")), self.snippet_chars)
            title = compact_text(str(context.get("title", "")), 120)
            if snippet:
                evidence_sentences.append(f"[{index}] {title}: {snippet}")
            else:
                evidence_sentences.append(f"[{index}] {title}")

        closing = (
            f"For the question '{question}', use these citations as the supporting evidence "
            "and avoid adding claims that are not supported by the retrieved passages."
        )
        return f"{lead}\n\n" + "\n".join(evidence_sentences) + f"\n\n{closing}"


class HuggingFaceText2TextGenerator:
    """Optional open-source LLM generator using a HuggingFace text2text model."""

    def __init__(
        self,
        model_name: str = DEFAULT_HF_RAG_MODEL,
        max_new_tokens: int = 180,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "HuggingFace RAG generation requires transformers. "
                "Install project requirements first."
            ) from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.pipeline = pipeline("text2text-generation", model=model_name)

    def generate(self, question: str, contexts: Sequence[dict[str, object]]) -> str:
        if not contexts:
            return "I could not find enough retrieved evidence to answer with citations."

        context_block = "\n".join(
            f"[{index}] {context['title']}: {context['snippet']}"
            for index, context in enumerate(contexts, start=1)
        )
        prompt = (
            "Answer the question using only the cited context. "
            "Include citation markers like [1] and [2].\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_block}\n\n"
            "Answer:"
        )
        outputs = self.pipeline(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
        answer = str(outputs[0]["generated_text"]).strip()
        if not re.search(r"\[\d+\]", answer):
            answer = f"{answer} " + " ".join(f"[{index}]" for index in range(1, min(len(contexts), 2) + 1))
        return answer


class HybridRAGPipeline:
    """Hybrid retrieval plus answer generation."""

    def __init__(
        self,
        bm25_retriever: Retriever,
        semantic_retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
        candidate_k: int = 10,
        context_k: int = 3,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if context_k <= 0:
            raise ValueError("context_k must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.generator = generator or CitationAnswerGenerator(max_contexts=context_k)
        self.candidate_k = candidate_k
        self.context_k = context_k
        self.rrf_k = rrf_k

    def retrieve(self, question: str) -> list[dict[str, object]]:
        result_sets = {
            "bm25": self.bm25_retriever.search(question, top_k=self.candidate_k),
        }
        if self.semantic_retriever is not None:
            result_sets["semantic"] = self.semantic_retriever.search(question, top_k=self.candidate_k)
        return reciprocal_rank_fusion(result_sets, top_k=self.context_k, rrf_k=self.rrf_k)

    def answer(self, question: str) -> RAGResponse:
        contexts = self.retrieve(question)
        answer = self.generator.generate(question, contexts)
        return RAGResponse(question=question, answer=answer, contexts=contexts)


def reciprocal_rank_fusion(
    result_sets: dict[str, Sequence[dict[str, object]]],
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, object]]:
    """Fuse multiple ranked result lists with reciprocal rank fusion."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    fused: dict[str, dict[str, object]] = {}
    for retriever_name, results in result_sets.items():
        for rank, result in enumerate(results, start=1):
            doc_id = str(result["doc_id"])
            if doc_id not in fused:
                fused[doc_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "retrievers": [],
                    "component_ranks": {},
                    "component_scores": {},
                }

            fused_result = fused[doc_id]
            fused_result["rrf_score"] = float(fused_result["rrf_score"]) + 1 / (rrf_k + rank)
            fused_result["retrievers"].append(retriever_name)
            fused_result["component_ranks"][retriever_name] = rank
            fused_result["component_scores"][retriever_name] = result.get("score", 0.0)

    ranked = sorted(
        fused.values(),
        key=lambda result: (
            float(result["rrf_score"]),
            int(result.get("stack_score", 0) or 0),
            int(result.get("answer_count", 0) or 0),
        ),
        reverse=True,
    )[:top_k]

    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
        result["rrf_score"] = round(float(result["rrf_score"]), 6)
        result["retrievers"] = ", ".join(sorted(set(result["retrievers"])))
        result["component_ranks"] = format_component_map(result["component_ranks"])
        result["component_scores"] = format_component_map(result["component_scores"])
    return ranked


def compact_text(text: str, max_chars: int) -> str:
    """Normalize whitespace and trim text for answer prompts and reports."""

    compacted = " ".join(text.split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."


def format_component_map(values: Any) -> str:
    if not isinstance(values, dict):
        return str(values)
    return "; ".join(f"{key}:{value}" for key, value in sorted(values.items()))
