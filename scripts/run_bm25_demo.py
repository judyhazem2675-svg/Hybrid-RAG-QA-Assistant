"""Run Jomana's BM25 milestone demo and export top-10 sample results."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25Retriever, load_stacklite_zip, write_results_csv  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "DataSet.zip"
OUTPUT_PATH = PROJECT_ROOT / "results/bm25_sample_top10.csv"

SAMPLE_QUERIES = [
    "Why are micro average precision recall and F1 equal in multiclass classification?",
    "What is the difference between artificial intelligence and machine learning?",
    "What are deconvolutional layers in convolutional neural networks?",
    "How do I set class weights for imbalanced classes in Keras?",
    "What is the dying ReLU problem in neural networks?",
]


def main() -> None:
    documents = load_stacklite_zip(DATASET_PATH)
    retriever = BM25Retriever(k1=1.5, b=0.75).fit(documents)
    query_to_results = {
        query: retriever.search(query, top_k=10) for query in SAMPLE_QUERIES
    }
    write_results_csv(query_to_results, OUTPUT_PATH)

    print(f"Loaded documents: {len(documents)}")
    print(f"Average BM25 document length: {retriever.avg_doc_length:.2f} tokens")
    print(f"Exported top-10 results for {len(SAMPLE_QUERIES)} queries to {OUTPUT_PATH}")
    print()
    for query, results in query_to_results.items():
        best = results[0]
        print(f"Query: {query}")
        print(f"Top hit: [{best['source']}] {best['title']} ({best['score']})")
        print(f"Citation: {best['link']}")
        print()


if __name__ == "__main__":
    main()
