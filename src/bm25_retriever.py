"""BM25 lexical retrieval for the StackLite question corpus.

This module covers Jomana's assigned project scope:

- load and preprocess the provided StackLite JSON files
- build a BM25 keyword index over question titles, bodies, and tags
- return top-k lexical matches with citation-ready metadata

The implementation is intentionally dependency-light so it runs cleanly in
Colab and in a local Python environment.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Sequence


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+#.\-]*", re.IGNORECASE)

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


class _HTMLTextExtractor(HTMLParser):
    """Small HTML-to-text converter using Python's standard library."""

    _BLOCK_TAGS = {"br", "p", "div", "li", "pre", "code", "tr", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append(" ")

    def get_text(self) -> str:
        return " ".join(" ".join(self._parts).split())


def clean_html(raw_html: str | None) -> str:
    """Convert Stack Exchange HTML question bodies into normalized plain text."""

    if not raw_html:
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(html.unescape(raw_html))
    parser.close()
    return parser.get_text()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize technical text while preserving useful tokens like c++, c#, and .net."""

    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("._-")
        if not token:
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        if len(token) == 1 and token not in {"c", "r"}:
            continue
        tokens.append(token)
    return tokens


@dataclass(frozen=True)
class Document:
    """A searchable StackLite question with citation metadata."""

    doc_id: str
    source: str
    question_id: int
    title: str
    body: str
    tags: tuple[str, ...]
    link: str
    score: int
    answer_count: int
    view_count: int

    @property
    def text(self) -> str:
        tags_text = " ".join(self.tags)
        title_boost = f"{self.title}\n{self.title}"
        return f"{title_boost}\n\n{self.body}\n\nTags: {tags_text}".strip()


def load_stacklite_zip(zip_path: str | Path) -> list[Document]:
    """Load all StackLite JSON files from the provided ZIP archive."""

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")

    documents: list[Document] = []
    with zipfile.ZipFile(zip_path) as archive:
        json_files = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".json") and not name.startswith("__MACOSX/")
        )
        if not json_files:
            raise ValueError(f"No JSON corpus files found in {zip_path}")

        for name in json_files:
            source = Path(name).stem.replace("top_", "").replace("_questions", "")
            records = json.loads(archive.read(name).decode("utf-8"))
            for record in records:
                question_id = int(record.get("question_id", 0))
                documents.append(
                    Document(
                        doc_id=f"{source}:{question_id}",
                        source=source,
                        question_id=question_id,
                        title=html.unescape(str(record.get("title", "")).strip()),
                        body=clean_html(record.get("body", "")),
                        tags=tuple(record.get("tags", []) or []),
                        link=str(record.get("link", "")).strip(),
                        score=int(record.get("score", 0) or 0),
                        answer_count=int(record.get("answer_count", 0) or 0),
                        view_count=int(record.get("view_count", 0) or 0),
                    )
                )
    return documents


class BM25Retriever:
    """Okapi BM25 retriever for keyword-based lexical search."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.documents: list[Document] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def fit(self, documents: Sequence[Document]) -> "BM25Retriever":
        """Build the BM25 index."""

        self.documents = list(documents)
        if not self.documents:
            raise ValueError("Cannot fit BM25Retriever on an empty document list")

        tokenized_docs = [self.tokenizer(document.text) for document in self.documents]
        self.doc_lengths = [len(tokens) for tokens in tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        document_frequency: Counter[str] = Counter()
        self.postings = defaultdict(list)

        for doc_index, tokens in enumerate(tokenized_docs):
            term_counts = Counter(tokens)
            document_frequency.update(term_counts.keys())
            for term, frequency in term_counts.items():
                self.postings[term].append((doc_index, frequency))

        corpus_size = len(self.documents)
        self.idf = {
            term: math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
        """Return top-k BM25 results for a user query."""

        if not self.documents:
            raise RuntimeError("BM25Retriever must be fit before calling search")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = Counter(self.tokenizer(query))
        scores = [0.0] * len(self.documents)

        for term, query_frequency in query_terms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_index, term_frequency in self.postings.get(term, []):
                doc_length = self.doc_lengths[doc_index] or 1
                normalization = self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                numerator = term_frequency * (self.k1 + 1)
                scores[doc_index] += query_frequency * idf * numerator / (
                    term_frequency + normalization
                )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (
                scores[index],
                self.documents[index].score,
                self.documents[index].answer_count,
            ),
            reverse=True,
        )[:top_k]

        query_tokens = set(query_terms)
        results: list[dict[str, object]] = []
        for rank, doc_index in enumerate(ranked_indices, start=1):
            document = self.documents[doc_index]
            results.append(
                {
                    "rank": rank,
                    "score": round(scores[doc_index], 6),
                    "doc_id": document.doc_id,
                    "source": document.source,
                    "question_id": document.question_id,
                    "title": document.title,
                    "tags": ", ".join(document.tags),
                    "link": document.link,
                    "snippet": make_snippet(document.text, query_tokens),
                    "stack_score": document.score,
                    "answer_count": document.answer_count,
                    "view_count": document.view_count,
                }
            )
        return results


def make_snippet(text: str, query_tokens: set[str], window: int = 180) -> str:
    """Create a compact citation snippet around the first matched query term."""

    normalized = " ".join(text.split())
    lower_text = normalized.lower()
    hit_positions = [
        lower_text.find(term)
        for term in query_tokens
        if term and lower_text.find(term) >= 0
    ]
    if not hit_positions:
        return normalized[:window].rstrip() + ("..." if len(normalized) > window else "")

    center = min(hit_positions)
    start = max(0, center - window // 2)
    end = min(len(normalized), start + window)
    snippet = normalized[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(normalized):
        snippet = snippet + "..."
    return snippet


def write_results_csv(
    query_to_results: dict[str, list[dict[str, object]]],
    output_path: str | Path,
) -> None:
    """Write sample query results to CSV for easy inspection and reporting."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for query, results in query_to_results.items():
        for result in results:
            rows.append({"query": query, **result})

    fieldnames = [
        "query",
        "rank",
        "score",
        "doc_id",
        "source",
        "question_id",
        "title",
        "tags",
        "link",
        "snippet",
        "stack_score",
        "answer_count",
        "view_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
