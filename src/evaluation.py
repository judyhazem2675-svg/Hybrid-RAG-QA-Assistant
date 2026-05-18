"""Retrieval evaluation metrics for the Hybrid RAG QA Assistant."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EvaluationQuestion:
    """A query and its manually judged relevant document IDs."""

    query_id: str
    query: str
    relevant_doc_ids: frozenset[str]
    notes: str = ""


def load_evaluation_questions(path: str | Path) -> list[EvaluationQuestion]:
    """Load query relevance judgments from a JSON file."""

    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    questions: list[EvaluationQuestion] = []
    for record in records:
        relevant_doc_ids = frozenset(record["relevant_doc_ids"])
        if not relevant_doc_ids:
            raise ValueError(f"{record['query_id']} must have at least one relevant document")
        questions.append(
            EvaluationQuestion(
                query_id=record["query_id"],
                query=record["query"],
                relevant_doc_ids=relevant_doc_ids,
                notes=record.get("notes", ""),
            )
        )
    return questions


def average_precision_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Iterable[str],
    k: int = 10,
) -> float:
    """Compute AP@k for one query."""

    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant), k)


def reciprocal_rank_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Iterable[str],
    k: int = 10,
) -> float:
    """Compute RR@k for one query."""

    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Iterable[str],
    k: int = 10,
) -> float:
    """Compute binary nDCG@k for one query."""

    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_retrieval(
    run: dict[str, Sequence[str]],
    questions: Sequence[EvaluationQuestion],
    k: int = 10,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Compute MAP@k and MRR@k plus per-query details."""

    if not questions:
        raise ValueError("questions must not be empty")

    rows: list[dict[str, object]] = []
    for question in questions:
        retrieved_doc_ids = list(run.get(question.query_id, []))
        ap_at_k = average_precision_at_k(
            retrieved_doc_ids,
            question.relevant_doc_ids,
            k=k,
        )
        rr_at_k = reciprocal_rank_at_k(
            retrieved_doc_ids,
            question.relevant_doc_ids,
            k=k,
        )
        ndcg = ndcg_at_k(
            retrieved_doc_ids,
            question.relevant_doc_ids,
            k=k,
        )
        rows.append(
            {
                "query_id": question.query_id,
                "query": question.query,
                f"AP@{k}": round(ap_at_k, 6),
                f"RR@{k}": round(rr_at_k, 6),
                f"nDCG@{k}": round(ndcg, 6),
                "relevant_doc_ids": ", ".join(sorted(question.relevant_doc_ids)),
                "retrieved_doc_ids": ", ".join(retrieved_doc_ids[:k]),
                "notes": question.notes,
            }
        )
    metrics = {
        f"MAP@{k}": round(sum(float(row[f"AP@{k}"]) for row in rows) / len(rows), 6),
        f"MRR@{k}": round(sum(float(row[f"RR@{k}"]) for row in rows) / len(rows), 6),
        f"nDCG@{k}": round(sum(float(row[f"nDCG@{k}"]) for row in rows) / len(rows), 6),
        "queries": float(len(rows)),
    }
    return metrics, rows


def write_dict_csv(rows: Sequence[dict[str, object]], output_path: str | Path) -> None:
    """Write a sequence of dictionaries to CSV."""

    if not rows:
        raise ValueError("rows must not be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

