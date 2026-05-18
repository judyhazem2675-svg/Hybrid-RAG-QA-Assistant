# MiniLM Semantic Search Evaluation

This report covers Kadry Mostafa's semantic retrieval milestone.

## Method

- Dense model: `sentence-transformers/all-MiniLM-L6-v2`
- Corpus: StackLite questions from `DataSet.zip`
- Baseline: Jomana's Okapi BM25 retriever
- Relevance judgments: Abdallah's curated evaluation queries

## Metrics

| Retriever | MAP@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 1.000000 | 1.000000 | 1.000000 |
| MiniLM semantic | 1.000000 | 1.000000 | 1.000000 |

## Per-Query nDCG@10

| Query ID | BM25 nDCG@10 | Semantic nDCG@10 |
|---|---:|---:|
| q1_micro_macro_average | 1.000000 | 1.000000 |
| q2_ai_vs_ml | 1.000000 | 1.000000 |
| q3_deconvolution_layers | 1.000000 | 1.000000 |
| q4_keras_class_weights | 1.000000 | 1.000000 |
| q5_dying_relu | 1.000000 | 1.000000 |

## Notes

- nDCG@10 rewards relevant documents ranked near the top of the first page.
- The semantic retriever is intentionally isolated from RAG generation and UI work.
- BM25 can remain stronger on exact technical titles; MiniLM is added as the dense retrieval baseline.
