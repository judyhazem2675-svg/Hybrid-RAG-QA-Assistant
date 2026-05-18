# DVC Dataset Tracking

The StackLite corpus CSV is tracked with DVC instead of Git.

Git tracks:

- `data/stacklite_questions.csv.dvc` - the small DVC pointer file
- `.dvc/config` - the DagsHub remote configuration
- `data/.gitignore` - ignores the local CSV materialized by DVC

The actual dataset file is restored locally with:

```bash
pip install dvc
dvc pull data/stacklite_questions.csv.dvc
```

The configured DVC remote is:

```text
https://dagshub.com/kadry720/Hybrid-RAG-QA-Assistant.dvc
```

To upload the dataset to DagsHub, configure credentials locally without committing them:

```bash
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user YOUR_DAGSHUB_USERNAME
dvc remote modify dagshub --local password YOUR_DAGSHUB_TOKEN
dvc push
```

Do not commit `.dvc/config.local`; it contains local credentials.

## GitHub Actions

The CI workflow can pull the dataset during tests when this repository has a GitHub Actions secret named:

```text
DAGSHUB_TOKEN
```

Use a DagsHub token for the `kadry720` account, because the DVC remote is hosted under that account.
If the secret is not configured, CI skips the DVC pull step and dataset-dependent integration tests skip.
