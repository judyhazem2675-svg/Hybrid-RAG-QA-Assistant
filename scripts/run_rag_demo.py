"""Run Judy's RAG integration milestone with cited answers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25Retriever, load_stacklite_dataset  # noqa: E402
from src.evaluation import load_evaluation_questions  # noqa: E402
from src.rag_pipeline import (  # noqa: E402
    CitationAnswerGenerator,
    DEFAULT_HF_RAG_MODEL,
    HuggingFaceText2TextGenerator,
    HybridRAGPipeline,
    compact_text,
)
from src.semantic_retriever import DEFAULT_MINILM_MODEL, SemanticRetriever  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data" / "stacklite_questions.csv"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "bm25_eval_queries.json"
ANSWERS_OUTPUT = PROJECT_ROOT / "results" / "rag_sample_answers.csv"
CONTEXTS_OUTPUT = PROJECT_ROOT / "results" / "rag_retrieved_contexts.csv"
REPORT_OUTPUT = PROJECT_ROOT / "reports" / "rag_integration_report.md"


def main() -> None:
    args = parse_args()
    validate_args(args)
    questions = load_evaluation_questions(QUESTIONS_PATH)
    documents = load_stacklite_dataset(DATASET_PATH)

    bm25_retriever = BM25Retriever(k1=1.5, b=0.75).fit(documents)
    semantic_retriever = None
    if args.retriever == "hybrid":
        semantic_retriever = SemanticRetriever(model_name=DEFAULT_MINILM_MODEL).fit(documents)

    generator = build_generator(args.generator)
    pipeline = HybridRAGPipeline(
        bm25_retriever=bm25_retriever,
        semantic_retriever=semantic_retriever,
        generator=generator,
        candidate_k=args.candidate_k,
        context_k=args.context_k,
    )

    responses = [pipeline.answer(question.query) for question in questions[: args.limit]]
    write_answer_outputs(responses, ANSWERS_OUTPUT)
    write_context_outputs(responses, CONTEXTS_OUTPUT)
    write_report(args, responses, REPORT_OUTPUT)

    print("Judy RAG integration demo")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Documents: {len(documents)}")
    print(f"Retriever mode: {args.retriever}")
    print(f"Generator mode: {args.generator}")
    print(f"Questions answered: {len(responses)}")
    print(f"Answers CSV: {ANSWERS_OUTPUT}")
    print(f"Contexts CSV: {CONTEXTS_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")
    print()
    for response in responses:
        print(f"Question: {response.question}")
        print(compact_text(response.answer, 260))
        print("Citations:", ", ".join(citation["citation"] for citation in response.citations))
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=["hybrid", "bm25"], default="hybrid")
    parser.add_argument("--generator", choices=["extractive", "huggingface"], default="extractive")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--context-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def build_generator(generator_name: str):
    if generator_name == "huggingface":
        return HuggingFaceText2TextGenerator(model_name=DEFAULT_HF_RAG_MODEL)
    return CitationAnswerGenerator(max_contexts=3)


def validate_args(args: argparse.Namespace) -> None:
    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be positive")
    if args.context_k <= 0:
        raise ValueError("--context-k must be positive")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")


def write_answer_outputs(responses, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for response in responses:
        rows.append(
            {
                "question": response.question,
                "answer": response.answer,
                "citations": " ".join(citation["citation"] for citation in response.citations),
                "citation_doc_ids": ", ".join(str(citation["doc_id"]) for citation in response.citations),
                "citation_links": ", ".join(str(citation["link"]) for citation in response.citations),
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_context_outputs(responses, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for response in responses:
        for context in response.contexts:
            rows.append(
                {
                    "question": response.question,
                    "rank": context["rank"],
                    "rrf_score": context["rrf_score"],
                    "retrievers": context["retrievers"],
                    "doc_id": context["doc_id"],
                    "title": context["title"],
                    "link": context["link"],
                    "snippet": context["snippet"],
                    "component_ranks": context["component_ranks"],
                    "component_scores": context["component_scores"],
                }
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(args: argparse.Namespace, responses, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Integration Demo",
        "",
        "This report covers Judy Hazem's RAG integration milestone.",
        "",
        "## Method",
        "",
        "- Corpus: `data/stacklite_questions.csv`",
        "- Lexical retriever: Okapi BM25",
        f"- Dense retriever: `{DEFAULT_MINILM_MODEL}`" if args.retriever == "hybrid" else "- Dense retriever: not used",
        "- Fusion: Reciprocal Rank Fusion over BM25 and semantic results",
        f"- Generator: `{args.generator}`",
        "- Note: StackLite provides question-post passages, so citations point to retrieved posts used as grounding.",
        "",
        "## Sample Answers",
        "",
    ]
    for index, response in enumerate(responses, start=1):
        citations = ", ".join(
            f"{citation['citation']} {citation['doc_id']}" for citation in response.citations
        )
        lines.extend(
            [
                f"### Question {index}",
                "",
                response.question,
                "",
                response.answer,
                "",
                f"**Citations:** {citations}",
                "",
            ]
        )
    lines.extend(
        [
            "## Output Files",
            "",
            f"- Answers: `{ANSWERS_OUTPUT.relative_to(PROJECT_ROOT).as_posix()}`",
            f"- Retrieved contexts: `{CONTEXTS_OUTPUT.relative_to(PROJECT_ROOT).as_posix()}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
