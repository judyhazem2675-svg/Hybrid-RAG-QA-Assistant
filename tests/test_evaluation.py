from __future__ import annotations

import unittest

from src.evaluation import (
    EvaluationQuestion,
    average_precision_at_k,
    evaluate_retrieval,
    reciprocal_rank_at_k,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_average_precision_at_k_rewards_early_relevant_hits(self) -> None:
        retrieved = ["d1", "d2", "d3", "d4"]
        relevant = {"d2", "d4"}

        self.assertAlmostEqual(
            average_precision_at_k(retrieved, relevant, k=4),
            ((1 / 2) + (2 / 4)) / 2,
        )

    def test_reciprocal_rank_at_k_uses_first_relevant_hit(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d3"}

        self.assertAlmostEqual(reciprocal_rank_at_k(retrieved, relevant, k=3), 1 / 3)

    def test_evaluate_retrieval_returns_map_and_mrr(self) -> None:
        questions = [
            EvaluationQuestion("q1", "first query", frozenset({"d1"})),
            EvaluationQuestion("q2", "second query", frozenset({"d3"})),
        ]
        run = {
            "q1": ["d1", "d2"],
            "q2": ["d2", "d3"],
        }

        metrics, rows = evaluate_retrieval(run, questions, k=2)

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(metrics["MAP@2"], (1.0 + 0.5) / 2)
        self.assertAlmostEqual(metrics["MRR@2"], (1.0 + 0.5) / 2)


if __name__ == "__main__":
    unittest.main()

