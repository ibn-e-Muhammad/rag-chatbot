"""Glaive dataset ingestion and cleaning utilities."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Trim and collapse internal whitespace to a single space."""
    return _WHITESPACE_RE.sub(" ", value).strip()


def _get_field(row: Mapping[str, Any], key: str) -> str:
    for row_key, value in row.items():
        if row_key is not None and row_key.strip().lower() == key:
            return "" if value is None else str(value)
    return ""


def process_glaive(input_path: str | Path, output_path: str | Path, max_rows: int | None = None) -> int:
    """Stream-process a Glaive CSV into JSONL question-answer documents.

    Each output JSONL line has:
    - "document": "Question: ...\\nAnswer: ..."
    - "question": cleaned question text
    - "answer": cleaned answer text
    """
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be None or a non-negative integer.")

    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with source.open("r", encoding="utf-8-sig", newline="") as infile, target.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)

        for row in reader:
            if max_rows is not None and written >= max_rows:
                break

            question = normalize_text(_get_field(row, "question"))
            answer = normalize_text(_get_field(row, "answer"))

            if not question and not answer:
                continue

            payload = {
                "document": f"Question: {question}\nAnswer: {answer}",
                "question": question,
                "answer": answer,
            }
            outfile.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    return written
