# Video Demo Walkthrough

Target length: 3 to 5 minutes.

## 0:00-0:30 - Project Setup

- Open the repository and show `src/`, `scripts/`, `results/`, `reports/`, and `app.py`.
- State that the system is a Hybrid RAG QA assistant over the StackLite corpus.
- Mention that the dataset is DVC-tracked and should be pulled with:

```bash
dvc pull data/stacklite_questions.csv.dvc
```

## 0:30-1:15 - Retrieval Components

- Show `src/bm25_retriever.py`.
- Explain that BM25 indexes title, cleaned body text, and tags.
- Show `src/semantic_retriever.py`.
- Explain that MiniLM embeddings support dense semantic search.

## 1:15-2:00 - Evaluation Results

- Open `reports/bm25_evaluation_report.md`.
- Point out MAP@10 = 1.000000 and MRR@10 = 1.000000 on five curated queries.
- Open `reports/semantic_search_report.md`.
- Point out BM25 and MiniLM both reach nDCG@10 = 1.000000 on the current benchmark.
- Clarify that the evaluation set is small and curated, so it validates the milestone but is not a broad benchmark.

## 2:00-2:45 - RAG and Citations

- Show `src/rag_pipeline.py`.
- Explain reciprocal rank fusion between BM25 and semantic results.
- Open `results/rag_sample_answers.csv` and `results/rag_retrieved_contexts.csv`.
- Point out citation markers, source links, document IDs, component ranks, and snippets.

## 2:45-4:15 - Gradio UI Demo

- Run:

```bash
python app.py
```

- Ask: "What is the dying ReLU problem in neural networks?"
- Keep the default hybrid retriever and extractive citation generator.
- Show the generated answer, the evidence table, and the citation links.
- Ask: "How do I set class weights for imbalanced classes in Keras?"
- Switch to BM25-only mode to show the faster lexical baseline.
- Compare how the evidence table changes.

## 4:15-5:00 - Wrap-Up

- Open `reports/final_report.tex` and mention that it is ready for Overleaf.
- Open the README team log.
- Summarize the final deliverables: BM25, evaluation, semantic search, RAG, citation review, Gradio UI, final report, and this demo script.
