"""Build the DVC-tracked StackLite CSV from raw StackLite JSON files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "stacklite_questions.csv"

FIELDNAMES = [
    "source",
    "question_id",
    "title",
    "body",
    "tags",
    "link",
    "score",
    "answer_count",
    "view_count",
]


def main() -> None:
    json_paths = sorted(DATA_DIR.glob("top_*_questions.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"No raw StackLite JSON files found in {DATA_DIR}. "
            "Expected files like top_ai_questions.json."
        )

    rows: list[dict[str, object]] = []
    for json_path in json_paths:
        source = json_path.stem.replace("top_", "").replace("_questions", "")
        records = json.loads(json_path.read_text(encoding="utf-8"))
        for record in records:
            rows.append(
                {
                    "source": source,
                    "question_id": int(record.get("question_id", 0) or 0),
                    "title": str(record.get("title", "")).strip(),
                    "body": str(record.get("body", "") or ""),
                    "tags": json.dumps(record.get("tags", []) or [], ensure_ascii=False),
                    "link": str(record.get("link", "")).strip(),
                    "score": int(record.get("score", 0) or 0),
                    "answer_count": int(record.get("answer_count", 0) or 0),
                    "view_count": int(record.get("view_count", 0) or 0),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} StackLite records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
