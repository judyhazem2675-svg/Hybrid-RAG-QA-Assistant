# CI/CD Setup

This branch adds only the CI/CD component for the Hybrid RAG QA Assistant project.

Jomana should add the BM25 retrieval implementation in her own branch/PR. The workflow here is prepared to validate those files once they are added.

## Workflow

```text
.github/workflows/ci.yml
```

The workflow runs on:

- every push to `main`
- every pull request targeting `main`

## Pipeline Stages

1. **Lint**
   - Runs Ruff on Python files in `src`, `scripts`, `tests`, and `app` when those files exist.

2. **Unit Tests With Coverage**
   - Runs pytest when tests exist.
   - Enforces `70%` minimum coverage once a `src` package and tests are present.
   - Uploads `coverage.xml` as an artifact when coverage is generated.

3. **Data Validation**
   - Runs `tests/test_data_validation.py` when the project adds dataset validation.
   - Intended for StackLite schema and quality checks.

4. **Model Validation**
   - Runs `tests/test_model_validation.py` when a model validation test exists.
   - Falls back to `tests/test_retriever_validation.py` for retrieval-only milestones such as BM25.

## Why Some Steps Are Conditional

This PR is CI-only. The repository currently does not contain all project implementation files. Conditional checks let the CI scaffold merge cleanly first, then become strict as each teammate adds their part.

## Branch Protection Rule

After this PR is merged:

1. Go to GitHub repository **Settings**.
2. Open **Branches**.
3. Add a rule for `main`.
4. Enable **Require a pull request before merging**.
5. Enable **Require status checks to pass before merging**.
6. Select these checks:
   - `Lint`
   - `Unit Tests With Coverage`
   - `Data Validation`
   - `Model Validation`

## Evidence to Submit

- `.github/workflows/ci.yml`
- Screenshot of a green CI run
- Screenshot of branch protection enabled on `main`
- Coverage artifact once implementation tests are added

