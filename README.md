# Hybrid-RAG-QA-Assistant
A hybrid retrieval-based question answering assistant using BM25, MiniLM semantic search, RRF fusion, and RAG with Streamlit/Gradio interface.

## CI/CD

This repository includes a GitHub Actions pipeline in `.github/workflows/ci.yml`.

The CI stages are:

1. Lint
2. Unit tests with coverage
3. Data validation
4. Model or retriever validation

See `docs/ci_cd_setup.md` for setup notes and branch protection instructions.
