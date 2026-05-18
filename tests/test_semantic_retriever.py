from __future__ import annotations

import unittest

from src.bm25_retriever import Document
from src.semantic_retriever import SemanticRetriever


class KeywordVectorEmbedder:
    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float("neural" in lower or "relu" in lower),
                    float("keras" in lower or "class weights" in lower),
                    float("artificial intelligence" in lower or "machine learning" in lower),
                ]
            )
        return vectors


class SemanticRetrieverTests(unittest.TestCase):
    def test_search_returns_dense_similarity_ranking(self) -> None:
        documents = [
            make_document("datascience:5706", "What is the dying ReLU problem?", "Neural network ReLU units can die."),
            make_document("datascience:13490", "How to set class weights in Keras?", "Keras supports class weights."),
            make_document("ai:35", "AI vs machine learning", "Artificial intelligence and machine learning differ."),
        ]
        retriever = SemanticRetriever(embedder=KeywordVectorEmbedder()).fit(documents)

        results = retriever.search("Why do ReLU neurons die in neural networks?", top_k=2)

        self.assertEqual(results[0]["doc_id"], "datascience:5706")
        self.assertEqual(len(results), 2)

    def test_fit_rejects_empty_corpus(self) -> None:
        retriever = SemanticRetriever(embedder=KeywordVectorEmbedder())

        with self.assertRaises(ValueError):
            retriever.fit([])


def make_document(doc_id: str, title: str, body: str) -> Document:
    source, question_id = doc_id.split(":")
    return Document(
        doc_id=doc_id,
        source=source,
        question_id=int(question_id),
        title=title,
        body=body,
        tags=("machine-learning",),
        link=f"https://example.com/{doc_id}",
        score=1,
        answer_count=1,
        view_count=1,
    )


if __name__ == "__main__":
    unittest.main()
