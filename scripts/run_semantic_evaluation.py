"""Evaluate MiniLM semantic retrieval and compare it with the BM25 baseline."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25Retriever, load_stacklite_dataset, write_results_csv  # noqa: E402
from src.evaluation import evaluate_retrieval, load_evaluation_questions, write_dict_csv  # noqa: E402
from src.semantic_retriever import DEFAULT_MINILM_MODEL, SemanticRetriever  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data" / "stacklite_questions.csv"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "bm25_eval_queries.json"
COMPARISON_OUTPUT = PROJECT_ROOT / "results" / "semantic_vs_bm25_comparison.csv"
SEMANTIC_RUN_OUTPUT = PROJECT_ROOT / "results" / "semantic_evaluation_top10.csv"
REPORT_OUTPUT = PROJECT_ROOT / "reports" / "semantic_search_report.md"


def main() -> None:
    questions = load_evaluation_questions(QUESTIONS_PATH)
    documents = load_stacklite_dataset(DATASET_PATH)

    bm25_retriever = BM25Retriever(k1=1.5, b=0.75).fit(documents)
    semantic_retriever = SemanticRetriever(model_name=DEFAULT_MINILM_MODEL).fit(documents)

    bm25_query_to_results = {
        question.query: bm25_retriever.search(question.query, top_k=10)
        for question in questions
    }
    semantic_query_to_results = {
        question.query: semantic_retriever.search(question.query, top_k=10)
        for question in questions
    }

    bm25_run = make_run(questions, bm25_query_to_results)
    semantic_run = make_run(questions, semantic_query_to_results)
    bm25_metrics, bm25_rows = evaluate_retrieval(bm25_run, questions, k=10)
    semantic_metrics, semantic_rows = evaluate_retrieval(semantic_run, questions, k=10)

    comparison_rows = build_comparison_rows(bm25_rows, semantic_rows)
    write_results_csv(semantic_query_to_results, SEMANTIC_RUN_OUTPUT)
    write_dict_csv(comparison_rows, COMPARISON_OUTPUT)
    write_report(bm25_metrics, semantic_metrics, comparison_rows, REPORT_OUTPUT)

    print("MiniLM semantic search evaluation")
    print(f"Model: {DEFAULT_MINILM_MODEL}")
    print(f"Queries: {int(semantic_metrics['queries'])}")
    print(f"BM25 nDCG@10: {bm25_metrics['nDCG@10']:.6f}")
    print(f"Semantic nDCG@10: {semantic_metrics['nDCG@10']:.6f}")
    print(f"BM25 MAP@10: {bm25_metrics['MAP@10']:.6f}")
    print(f"Semantic MAP@10: {semantic_metrics['MAP@10']:.6f}")
    print(f"Comparison CSV: {COMPARISON_OUTPUT}")
    print(f"Semantic top-10 run: {SEMANTIC_RUN_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")


def make_run(
    questions: list,
    query_to_results: dict[str, list[dict[str, object]]],
) -> dict[str, list[str]]:
    return {
        question.query_id: [
            str(result["doc_id"]) for result in query_to_results[question.query]
        ]
        for question in questions
    }


def build_comparison_rows(
    bm25_rows: list[dict[str, object]],
    semantic_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    semantic_by_query = {str(row["query_id"]): row for row in semantic_rows}
    for bm25_row in bm25_rows:
        semantic_row = semantic_by_query[str(bm25_row["query_id"])]
        rows.append(
            {
                "query_id": bm25_row["query_id"],
                "query": bm25_row["query"],
                "bm25_AP@10": bm25_row["AP@10"],
                "semantic_AP@10": semantic_row["AP@10"],
                "bm25_RR@10": bm25_row["RR@10"],
                "semantic_RR@10": semantic_row["RR@10"],
                "bm25_nDCG@10": bm25_row["nDCG@10"],
                "semantic_nDCG@10": semantic_row["nDCG@10"],
                "relevant_doc_ids": bm25_row["relevant_doc_ids"],
                "bm25_retrieved_doc_ids": bm25_row["retrieved_doc_ids"],
                "semantic_retrieved_doc_ids": semantic_row["retrieved_doc_ids"],
            }
        )
    return rows


def write_report(
    bm25_metrics: dict[str, float],
    semantic_metrics: dict[str, float],
    comparison_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MiniLM Semantic Search Evaluation",
        "",
        "This report covers Kadry Mostafa's semantic retrieval milestone.",
        "",
        "## Method",
        "",
        f"- Dense model: `{DEFAULT_MINILM_MODEL}`",
        "- Corpus: StackLite questions from `data/stacklite_questions.csv`",
        "- Baseline: Jomana's Okapi BM25 retriever",
        "- Relevance judgments: Abdallah's curated evaluation queries",
        "",
        "## Metrics",
        "",
        "| Retriever | MAP@10 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|",
        f"| BM25 | {bm25_metrics['MAP@10']:.6f} | {bm25_metrics['MRR@10']:.6f} | "
        f"{bm25_metrics['nDCG@10']:.6f} |",
        f"| MiniLM semantic | {semantic_metrics['MAP@10']:.6f} | {semantic_metrics['MRR@10']:.6f} | "
        f"{semantic_metrics['nDCG@10']:.6f} |",
        "",
        "## Per-Query nDCG@10",
        "",
        "| Query ID | BM25 nDCG@10 | Semantic nDCG@10 |",
        "|---|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['query_id']} | {float(row['bm25_nDCG@10']):.6f} | "
            f"{float(row['semantic_nDCG@10']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- nDCG@10 rewards relevant documents ranked near the top of the first page.",
            "- The semantic retriever is intentionally isolated from RAG generation and UI work.",
            "- BM25 can remain stronger on exact technical titles; MiniLM is added as the dense retrieval baseline.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
