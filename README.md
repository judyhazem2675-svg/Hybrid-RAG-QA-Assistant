# Hybrid-RAG-QA-Assistant

A hybrid retrieval-based question answering assistant using BM25, MiniLM semantic search, RRF fusion, and RAG with Streamlit/Gradio interface.

This repository now includes:

- the CI/CD scaffold from the project infrastructure work
- Jomana's assigned first milestone: **BM25 retrieval for keyword-based lexical search**
- Abdallah's next milestone in progress: **retrieval evaluation with MAP@10 and MRR@10**

## Jomana BM25 Scope

- Load the provided `DataSet.zip` StackLite corpus.
- Preprocess Stack Exchange question titles, HTML bodies, tags, and citation metadata.
- Build an Okapi BM25 keyword index.
- Return top-10 lexical search results for sample technical questions.
- Provide a Colab-ready notebook with saved outputs for the BM25 retrieval demo.

Out of scope for Jomana's BM25 milestone:

- semantic search
- hybrid fusion
- RAG generation
- retrieval evaluation metrics
- citation-quality review
- UI integration

## BM25 Files

- `src/bm25_retriever.py` - reusable BM25 loader, preprocessing, indexing, search, and CSV export code.
- `scripts/run_bm25_demo.py` - local demo script that exports top-10 results for sample questions.
- `scripts/execute_notebook_inplace.py` - helper script to run the notebook and save outputs inside it.
- `notebooks/Jomana_BM25_Retrieval.ipynb` - Colab/Jupyter notebook for the milestone.
- `results/bm25_sample_top10.csv` - generated top-10 sample retrieval results.
- `DataSet.zip` - provided corpus archive.

## Run BM25 Locally

```bash
python scripts/run_bm25_demo.py
```

The script loads `DataSet.zip`, builds the BM25 index, and writes:

```text
results/bm25_sample_top10.csv
```

To rerun the notebook and save its outputs:

```bash
python scripts/execute_notebook_inplace.py
```

## BM25 Method Summary

The retriever indexes each StackLite question as:

```text
lightly boosted title + cleaned body text + tags
```

It uses Okapi BM25 with:

- `k1 = 1.5`
- `b = 0.75`
- IDF smoothing: `log(1 + (N - df + 0.5) / (df + 0.5))`

Each result includes citation-ready metadata: title, source, Stack Exchange link, tags, score, answer count, and snippet.

## CI/CD

This repository includes a GitHub Actions pipeline in `.github/workflows/ci.yml`.

The CI stages are:

1. Lint
2. Unit tests with coverage
3. Data validation
4. Model or retriever validation

See `docs/ci_cd_setup.md` for setup notes and branch protection instructions.

## Abdallah Retrieval Evaluation

Abdallah's immediate milestone evaluates the BM25 baseline with manually judged relevance questions.

Evaluation files:

- `evaluation/bm25_eval_queries.json` - curated evaluation queries and relevant document IDs.
- `src/evaluation.py` - MAP@10 and MRR@10 metric utilities.
- `scripts/run_bm25_evaluation.py` - runs BM25 evaluation and writes outputs.
- `results/bm25_evaluation_per_query.csv` - per-query AP@10 and RR@10 results.
- `results/bm25_evaluation_top10.csv` - top-10 retrieved documents for evaluation queries.
- `reports/bm25_evaluation_report.md` - short evaluation report.

Run evaluation:

```bash
python scripts/run_bm25_evaluation.py
```
