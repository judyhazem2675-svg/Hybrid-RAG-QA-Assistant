from __future__ import annotations

import unittest
from pathlib import Path

from src.bm25_retriever import BM25Retriever, load_stacklite_zip
from src.evaluation import evaluate_retrieval, load_evaluation_questions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "DataSet.zip"


class BM25EvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(
                "DataSet.zip is not available; run `dvc pull DataSet.zip.dvc` "
                "to enable dataset integration tests."
            )
        cls.questions = load_evaluation_questions(
            PROJECT_ROOT / "evaluation" / "bm25_eval_queries.json"
        )
        cls.documents = load_stacklite_zip(DATASET_PATH)
        cls.retriever = BM25Retriever().fit(cls.documents)

    def test_bm25_baseline_scores_high_on_curated_questions(self) -> None:
        run = {
            question.query_id: [
                str(result["doc_id"])
                for result in self.retriever.search(question.query, top_k=10)
            ]
            for question in self.questions
        }

        metrics, rows = evaluate_retrieval(run, self.questions, k=10)

        self.assertEqual(len(rows), 5)
        self.assertGreaterEqual(metrics["MAP@10"], 0.8)
        self.assertGreaterEqual(metrics["MRR@10"], 0.9)


if __name__ == "__main__":
    unittest.main()

