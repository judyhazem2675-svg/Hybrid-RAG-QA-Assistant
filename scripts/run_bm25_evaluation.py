"""Evaluate Jomana's BM25 retriever with MAP@10 and MRR@10."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25Retriever, load_stacklite_dataset, write_results_csv  # noqa: E402
from src.evaluation import evaluate_retrieval, load_evaluation_questions, write_dict_csv  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data" / "stacklite_questions.csv"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "bm25_eval_queries.json"
PER_QUERY_OUTPUT = PROJECT_ROOT / "results" / "bm25_evaluation_per_query.csv"
RUN_OUTPUT = PROJECT_ROOT / "results" / "bm25_evaluation_top10.csv"
REPORT_OUTPUT = PROJECT_ROOT / "reports" / "bm25_evaluation_report.md"


def main() -> None:
    questions = load_evaluation_questions(QUESTIONS_PATH)
    documents = load_stacklite_dataset(DATASET_PATH)
    retriever = BM25Retriever(k1=1.5, b=0.75).fit(documents)

    query_to_results = {
        question.query: retriever.search(question.query, top_k=10)
        for question in questions
    }
    run = {
        question.query_id: [
            str(result["doc_id"]) for result in query_to_results[question.query]
        ]
        for question in questions
    }
    metrics, per_query_rows = evaluate_retrieval(run, questions, k=10)

    write_results_csv(query_to_results, RUN_OUTPUT)
    write_dict_csv(per_query_rows, PER_QUERY_OUTPUT)
    write_report(metrics, per_query_rows, REPORT_OUTPUT)

    print("BM25 retrieval evaluation")
    print(f"Queries: {int(metrics['queries'])}")
    print(f"MAP@10: {metrics['MAP@10']:.6f}")
    print(f"MRR@10: {metrics['MRR@10']:.6f}")
    print(f"Per-query metrics: {PER_QUERY_OUTPUT}")
    print(f"Top-10 run: {RUN_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")


def write_report(
    metrics: dict[str, float],
    per_query_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BM25 Retrieval Evaluation",
        "",
        "This report covers Abdallah's retrieval evaluation milestone for the BM25 baseline.",
        "",
        "## Metrics",
        "",
        f"- Queries: {int(metrics['queries'])}",
        f"- MAP@10: {metrics['MAP@10']:.6f}",
        f"- MRR@10: {metrics['MRR@10']:.6f}",
        "",
        "## Per-Query Results",
        "",
        "| Query ID | AP@10 | RR@10 | Relevant Documents |",
        "|---|---:|---:|---|",
    ]
    for row in per_query_rows:
        lines.append(
            f"| {row['query_id']} | {row['AP@10']:.6f} | "
            f"{row['RR@10']:.6f} | {row['relevant_doc_ids']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- MAP@10 measures how highly relevant documents are ranked across the top 10 results.",
            "- MRR@10 measures how early the first relevant document appears.",
            "- Relevance judgments are stored in `evaluation/bm25_eval_queries.json`.",
            "- This evaluation is limited to BM25 retrieval and does not include semantic search or RAG generation.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

