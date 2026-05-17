# BM25 Retrieval Evaluation

This report covers Abdallah's retrieval evaluation milestone for the BM25 baseline.

## Metrics

- Queries: 5
- MAP@10: 1.000000
- MRR@10: 1.000000

## Per-Query Results

| Query ID | AP@10 | RR@10 | Relevant Documents |
|---|---:|---:|---|
| q1_micro_macro_average | 1.000000 | 1.000000 | datascience:15989 |
| q2_ai_vs_ml | 1.000000 | 1.000000 | ai:35, datascience:19077 |
| q3_deconvolution_layers | 1.000000 | 1.000000 | datascience:6107 |
| q4_keras_class_weights | 1.000000 | 1.000000 | datascience:13490 |
| q5_dying_relu | 1.000000 | 1.000000 | datascience:18810, datascience:5706 |

## Notes

- MAP@10 measures how highly relevant documents are ranked across the top 10 results.
- MRR@10 measures how early the first relevant document appears.
- Relevance judgments are stored in `evaluation/bm25_eval_queries.json`.
- This evaluation is limited to BM25 retrieval and does not include semantic search or RAG generation.
