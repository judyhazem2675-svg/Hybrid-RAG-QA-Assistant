"""Gradio interface for live Hybrid RAG Q&A."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25Retriever, Document, load_stacklite_dataset  # noqa: E402
from src.rag_pipeline import (  # noqa: E402
    CitationAnswerGenerator,
    DEFAULT_HF_RAG_MODEL,
    HuggingFaceText2TextGenerator,
    HybridRAGPipeline,
)
from src.semantic_retriever import DEFAULT_MINILM_MODEL, SemanticRetriever  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data" / "stacklite_questions.csv"
DEFAULT_RETRIEVER_MODE = "Hybrid: BM25 + MiniLM"
BM25_RETRIEVER_MODE = "BM25 only"
DEFAULT_GENERATOR_MODE = "Extractive citations"
HF_GENERATOR_MODE = "HuggingFace FLAN-T5"
CONTEXT_HEADERS = [
    "Rank",
    "Retriever(s)",
    "Doc ID",
    "Title",
    "RRF score",
    "Link",
    "Snippet",
]
EXAMPLE_QUESTIONS = [
    ["What is the dying ReLU problem in neural networks?"],
    ["How do I set class weights for imbalanced classes in Keras?"],
    ["What is the difference between artificial intelligence and machine learning?"],
    ["What are deconvolutional layers in convolutional neural networks?"],
]


@lru_cache(maxsize=1)
def load_documents() -> tuple[Document, ...]:
    """Load the DVC-tracked StackLite corpus once per app process."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset CSV not found at data/stacklite_questions.csv. "
            "Run `pip install dvc` and `dvc pull data/stacklite_questions.csv.dvc` "
            "from the repository root before using the live app."
        )
    return tuple(load_stacklite_dataset(DATASET_PATH))


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """Build and cache the lexical retriever."""

    return BM25Retriever(k1=1.5, b=0.75).fit(load_documents())


@lru_cache(maxsize=1)
def get_semantic_retriever() -> SemanticRetriever:
    """Build and cache the MiniLM retriever when hybrid mode is selected."""

    return SemanticRetriever(model_name=DEFAULT_MINILM_MODEL).fit(load_documents())


@lru_cache(maxsize=1)
def get_hf_generator() -> HuggingFaceText2TextGenerator:
    """Load the optional HuggingFace generator only when requested."""

    return HuggingFaceText2TextGenerator(model_name=DEFAULT_HF_RAG_MODEL)


def answer_question(
    question: str,
    retriever_mode: str,
    generator_mode: str,
    candidate_k: int | float,
    context_k: int | float,
) -> tuple[str, list[list[object]], str, str]:
    """Run live retrieval-augmented QA and return Gradio component values."""

    question = question.strip()
    if not question:
        return "Enter a question to start.", [], "", ""

    try:
        candidate_count = int(candidate_k)
        context_count = int(context_k)
        bm25_retriever = get_bm25_retriever()
        semantic_retriever = (
            get_semantic_retriever()
            if retriever_mode == DEFAULT_RETRIEVER_MODE
            else None
        )
        generator = (
            get_hf_generator()
            if generator_mode == HF_GENERATOR_MODE
            else CitationAnswerGenerator(max_contexts=context_count)
        )
        pipeline = HybridRAGPipeline(
            bm25_retriever=bm25_retriever,
            semantic_retriever=semantic_retriever,
            generator=generator,
            candidate_k=candidate_count,
            context_k=context_count,
        )
        response = pipeline.answer(question)
    except FileNotFoundError as exc:
        return dataset_missing_answer(str(exc)), [], "", "Dataset setup is required."
    except ImportError as exc:
        return dependency_error_answer(str(exc)), [], "", "Install the missing dependency and restart the app."
    except Exception as exc:  # pragma: no cover - UI safety net
        return (
            f"The app could not answer this question.\n\nError: {type(exc).__name__}: {exc}",
            [],
            "",
            "Runtime error.",
        )

    return (
        response.answer,
        context_rows(response.contexts),
        citation_markdown(response.citations),
        f"Used {retriever_mode} with {len(response.contexts)} cited contexts.",
    )


def context_rows(contexts: list[dict[str, object]]) -> list[list[object]]:
    """Convert retrieved contexts into a compact table for the UI."""

    rows: list[list[object]] = []
    for context in contexts:
        rows.append(
            [
                context.get("rank", ""),
                context.get("retrievers", ""),
                context.get("doc_id", ""),
                context.get("title", ""),
                context.get("rrf_score", context.get("score", "")),
                context.get("link", ""),
                context.get("snippet", ""),
            ]
        )
    return rows


def citation_markdown(citations: list[dict[str, object]]) -> str:
    """Format citation links for the Gradio markdown panel."""

    if not citations:
        return ""
    lines = []
    for citation in citations:
        marker = citation["citation"]
        title = citation["title"]
        doc_id = citation["doc_id"]
        link = citation["link"]
        lines.append(f"- {marker} [{title}]({link}) ({doc_id})")
    return "\n".join(lines)


def dataset_missing_answer(message: str) -> str:
    return (
        "The StackLite CSV is not available locally, so the retrievers cannot be built yet.\n\n"
        f"{message}\n\n"
        "Run this from the repository root:\n\n"
        "```bash\n"
        "pip install dvc\n"
        "dvc pull data/stacklite_questions.csv.dvc\n"
        "python app.py\n"
        "```"
    )


def dependency_error_answer(message: str) -> str:
    return (
        "A required UI or model dependency is missing.\n\n"
        f"{message}\n\n"
        "Install the project dependencies with:\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "```"
    )


def clear_interface() -> tuple[str, str, list[list[object]], str, str]:
    """Reset the visible app outputs."""

    return "", "", [], "", ""


def build_demo() -> gr.Blocks:
    """Create the Gradio Blocks app."""

    with gr.Blocks(title="Hybrid RAG QA Assistant") as demo:
        gr.Markdown("# Hybrid RAG QA Assistant")
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="Ask a technical machine learning or AI question...",
                lines=3,
                scale=3,
            )
        with gr.Row():
            retriever_mode = gr.Radio(
                choices=[DEFAULT_RETRIEVER_MODE, BM25_RETRIEVER_MODE],
                value=DEFAULT_RETRIEVER_MODE,
                label="Retriever",
            )
            generator_mode = gr.Radio(
                choices=[DEFAULT_GENERATOR_MODE, HF_GENERATOR_MODE],
                value=DEFAULT_GENERATOR_MODE,
                label="Generator",
            )
        with gr.Row():
            candidate_k = gr.Slider(
                minimum=3,
                maximum=20,
                value=10,
                step=1,
                label="Candidate results",
            )
            context_k = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Cited contexts",
            )
        with gr.Row():
            submit = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear")

        answer = gr.Markdown(label="Answer")
        contexts = gr.Dataframe(
            headers=CONTEXT_HEADERS,
            label="Retrieved evidence",
            interactive=False,
            wrap=True,
        )
        citations = gr.Markdown(label="Citation links")
        status = gr.Markdown(label="Status")

        gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question)

        inputs = [question, retriever_mode, generator_mode, candidate_k, context_k]
        outputs = [answer, contexts, citations, status]
        submit.click(fn=answer_question, inputs=inputs, outputs=outputs)
        question.submit(fn=answer_question, inputs=inputs, outputs=outputs)
        clear.click(
            fn=clear_interface,
            inputs=[],
            outputs=[question, answer, contexts, citations, status],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
