# Hybrid-RAG-QA-Assistant

A hybrid retrieval-based question answering assistant using BM25, MiniLM semantic search, RRF fusion, and RAG with Streamlit/Gradio interface.

This repository now includes:

- the CI/CD scaffold from the project infrastructure work
- Jomana's assigned first milestone: **BM25 retrieval for keyword-based lexical search**
- Abdallah's milestone: **retrieval evaluation with MAP@10 and MRR@10**
- Kadry Mostafa's milestone: **MiniLM semantic search with nDCG@10 comparison**
- Judy Hazem's milestone: **RAG integration with citation-grounded answers**

## Jomana BM25 Scope

- Load the DVC-tracked StackLite CSV corpus.
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
- `data/stacklite_questions.csv.dvc` - DVC pointer for the StackLite CSV corpus.

## Run BM25 Locally

```bash
python scripts/run_bm25_demo.py
```

The script loads `data/stacklite_questions.csv`, builds the BM25 index, and writes:

```text
results/bm25_sample_top10.csv
```

If `data/stacklite_questions.csv` is not present after cloning, pull it from the configured DVC remote:

```bash
pip install dvc
dvc pull data/stacklite_questions.csv.dvc
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

## Kadry Semantic Search

Kadry's milestone adds dense retrieval with MiniLM embeddings and compares it against the BM25 baseline.

Semantic search files:

- `src/semantic_retriever.py` - MiniLM embedding wrapper and dense cosine-similarity retriever.
- `scripts/run_semantic_evaluation.py` - compares BM25 and MiniLM on the evaluation questions.
- `results/semantic_vs_bm25_comparison.csv` - per-query MAP, MRR, and nDCG comparison.
- `results/semantic_evaluation_top10.csv` - top-10 MiniLM semantic retrieval results.
- `reports/semantic_search_report.md` - short semantic search evaluation report.
- `notebooks/Kadry_Semantic_Search.ipynb` - Colab/Jupyter notebook for the milestone.

Run semantic evaluation:

```bash
python scripts/run_semantic_evaluation.py
```

## Judy RAG Integration

Judy's milestone connects retrieval to answer generation with citation-ready outputs.

RAG files:

- `src/rag_pipeline.py` - RRF fusion, cited answer generation, and optional HuggingFace text2text generation.
- `scripts/run_rag_demo.py` - runs the RAG demo and writes answer/context outputs.
- `results/rag_sample_answers.csv` - generated answers with citation markers and links.
- `results/rag_retrieved_contexts.csv` - retrieved evidence used for each generated answer.
- `reports/rag_integration_report.md` - short RAG integration report.
- `notebooks/Judy_RAG_Integration.ipynb` - Colab/Jupyter notebook for the milestone.

Run the default hybrid RAG demo:

```bash
python scripts/run_rag_demo.py
```

Optional open-source LLM generation:

```bash
python scripts/run_rag_demo.py --generator huggingface
```

## Citation Quality Evaluation

The citation-quality milestone reviews whether generated RAG answers are supported by their retrieved sources.

Files:

- `evaluation/citation_quality_questions.json` - five citation-quality evaluation questions.
- `reports/citation_quality_examples.md` - good/bad citation examples and judgment notes.

The review uses the outputs from Judy's RAG notebook and demo script:

- `results/rag_sample_answers.csv`
- `results/rag_retrieved_contexts.csv`

## Interactive Gradio UI

The live Q&A app is implemented in `app.py`.

Run it locally:

```bash
pip install -r requirements.txt
pip install dvc
dvc pull data/stacklite_questions.csv.dvc
python app.py
```

The UI supports:

- hybrid BM25 + MiniLM retrieval
- BM25-only retrieval for faster lexical demos
- deterministic extractive citation answers
- optional HuggingFace FLAN-T5 generation
- retrieved evidence tables with document IDs, links, snippets, retriever sources, and RRF scores

If `data/stacklite_questions.csv` is missing, the app shows the DVC setup command instead of failing silently.

## Final Report and Demo

- Final Overleaf-ready report: `reports/final_report.tex`
- Video walkthrough script: `docs/video_demo_walkthrough.md`

The walkthrough is structured for a 3-5 minute demo covering retrieval, evaluation, RAG citations, and the Gradio interface.

## Team Log

- Jomana: BM25
- Abdallah: retrieval evaluation
- Kadry: semantic search
- Judy: RAG
- Judy + Abdallah: citation quality
- Kadry: UI
