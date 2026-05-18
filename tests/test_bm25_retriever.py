from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.bm25_retriever import (
    BM25Retriever,
    Document,
    clean_html,
    load_stacklite_zip,
    tokenize,
    write_results_csv,
)


class BM25RetrieverUnitTests(unittest.TestCase):
    def test_clean_html_extracts_readable_text(self) -> None:
        raw_html = "<p>Use <code>class_weight</code> for imbalanced classes.</p>"

        self.assertEqual(
            clean_html(raw_html),
            "Use class_weight for imbalanced classes.",
        )

    def test_tokenize_keeps_technical_terms_and_removes_stopwords(self) -> None:
        tokens = tokenize("How does C++ compare with C# and .NET?")

        self.assertIn("c++", tokens)
        self.assertIn("c#", tokens)
        self.assertIn("net", tokens)
        self.assertNotIn("how", tokens)

    def test_search_ranks_matching_document_first(self) -> None:
        documents = [
            make_document("datascience:1", "Dying ReLU problem", "Neural units can stop updating."),
            make_document("datascience:2", "Keras class weights", "Class weights help imbalanced labels."),
        ]
        retriever = BM25Retriever().fit(documents)

        results = retriever.search("What is the dying ReLU problem?", top_k=2)

        self.assertEqual(results[0]["doc_id"], "datascience:1")
        self.assertEqual(len(results), 2)

    def test_invalid_retriever_arguments_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            BM25Retriever(k1=0)
        with self.assertRaises(ValueError):
            BM25Retriever(b=1.5)
        with self.assertRaises(ValueError):
            BM25Retriever().fit([])
        with self.assertRaises(RuntimeError):
            BM25Retriever().search("query")

    def test_load_stacklite_zip_reads_stack_exchange_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "sample.zip"
            records = [
                {
                    "question_id": 42,
                    "title": "What is BM25?",
                    "body": "<p>A lexical retrieval function.</p>",
                    "tags": ["search", "ranking"],
                    "link": "https://example.com/questions/42",
                    "score": 7,
                    "answer_count": 2,
                    "view_count": 100,
                }
            ]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("top_datascience_questions.json", json.dumps(records))

            documents = load_stacklite_zip(zip_path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].doc_id, "datascience:42")
        self.assertEqual(documents[0].body, "A lexical retrieval function.")

    def test_write_results_csv_creates_flat_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.csv"
            write_results_csv(
                {
                    "query": [
                        {
                            "rank": 1,
                            "score": 2.5,
                            "doc_id": "datascience:1",
                            "source": "datascience",
                            "question_id": 1,
                            "title": "Title",
                            "tags": "tag",
                            "link": "https://example.com",
                            "snippet": "Snippet",
                            "stack_score": 1,
                            "answer_count": 1,
                            "view_count": 1,
                        }
                    ]
                },
                output_path,
            )

            content = output_path.read_text(encoding="utf-8")

        self.assertIn("query,rank,score,doc_id", content)
        self.assertIn("datascience:1", content)


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
