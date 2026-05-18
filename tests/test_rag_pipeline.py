from __future__ import annotations

import unittest

from src.rag_pipeline import CitationAnswerGenerator, HybridRAGPipeline, reciprocal_rank_fusion


class StaticRetriever:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
        del query
        return self.results[:top_k]


class RAGPipelineTests(unittest.TestCase):
    def test_rrf_promotes_documents_seen_by_multiple_retrievers(self) -> None:
        bm25_results = [
            make_result("doc-a", 1, 9.0),
            make_result("doc-b", 2, 8.0),
        ]
        semantic_results = [
            make_result("doc-c", 1, 0.9),
            make_result("doc-a", 2, 0.8),
        ]

        fused = reciprocal_rank_fusion(
            {"bm25": bm25_results, "semantic": semantic_results},
            top_k=3,
        )

        self.assertEqual(fused[0]["doc_id"], "doc-a")
        self.assertEqual(fused[0]["retrievers"], "bm25, semantic")
        self.assertEqual(fused[0]["component_ranks"], "bm25:1; semantic:2")

    def test_citation_generator_returns_numbered_evidence(self) -> None:
        generator = CitationAnswerGenerator(max_contexts=2, snippet_chars=60)
        contexts = [
            make_result("doc-a", 1, 9.0, title="Dying ReLU", snippet="ReLU units can stop activating."),
            make_result("doc-b", 2, 8.0, title="Neural Networks", snippet="Bias and learning rate can matter."),
        ]

        answer = generator.generate("What is dying ReLU?", contexts)

        self.assertIn("[1] Dying ReLU", answer)
        self.assertIn("[2] Neural Networks", answer)
        self.assertIn("avoid adding claims", answer)

    def test_pipeline_returns_answer_contexts_and_citation_metadata(self) -> None:
        bm25 = StaticRetriever([make_result("doc-a", 1, 9.0), make_result("doc-b", 2, 8.0)])
        semantic = StaticRetriever([make_result("doc-b", 1, 0.9), make_result("doc-c", 2, 0.8)])
        pipeline = HybridRAGPipeline(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            generator=CitationAnswerGenerator(max_contexts=2),
            candidate_k=2,
            context_k=2,
        )

        response = pipeline.answer("How should I cite retrieved contexts?")

        self.assertEqual(response.question, "How should I cite retrieved contexts?")
        self.assertEqual(len(response.contexts), 2)
        self.assertEqual(response.citations[0]["citation"], "[1]")
        self.assertIn("[1]", response.answer)


def make_result(
    doc_id: str,
    rank: int,
    score: float,
    title: str | None = None,
    snippet: str | None = None,
) -> dict[str, object]:
    return {
        "rank": rank,
        "score": score,
        "doc_id": doc_id,
        "source": "datascience",
        "question_id": rank,
        "title": title or f"Title {doc_id}",
        "tags": "machine-learning",
        "link": f"https://example.com/{doc_id}",
        "snippet": snippet or f"Snippet for {doc_id}",
        "stack_score": 1,
        "answer_count": 1,
        "view_count": 10,
    }


if __name__ == "__main__":
    unittest.main()
