# DVC Dataset Tracking

The StackLite corpus ZIP is tracked with DVC instead of Git.

Git tracks:

- `DataSet.zip.dvc` - the small DVC pointer file
- `.dvc/config` - the DagsHub remote configuration
- `.gitignore` - ignores the local `DataSet.zip`

The actual dataset file is restored locally with:

```bash
pip install dvc
dvc pull DataSet.zip.dvc
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
