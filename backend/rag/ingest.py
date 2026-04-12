"""Dataset ingestion entrypoint for Glaive and StackOverflow cleaning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glaive_ingest import process_glaive
from stackoverflow_processing import process_stackoverflow


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_ingestion(
    glaive_input: Path,
    glaive_output: Path,
    glaive_max_rows: int | None,
    so_questions: Path,
    so_answers: Path,
    so_output: Path,
    so_max_questions: int,
    so_batch_size: int,
    skip_glaive: bool = False,
    skip_stackoverflow: bool = False,
) -> dict:
    summary: dict[str, object] = {}

    if not skip_glaive:
        written = process_glaive(
            input_path=glaive_input,
            output_path=glaive_output,
            max_rows=glaive_max_rows,
        )
        summary["glaive"] = {
            "written_documents": written,
            "output_path": str(glaive_output),
        }

    if not skip_stackoverflow:
        so_summary = process_stackoverflow(
            questions_path=str(so_questions),
            answers_path=str(so_answers),
            output_path=str(so_output),
            max_questions=so_max_questions,
            batch_size=so_batch_size,
        )
        so_summary["output_path"] = str(so_output)
        summary["stackoverflow"] = so_summary

    return summary


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Ingest and clean Glaive + StackOverflow datasets."
    )
    parser.add_argument(
        "--glaive-input",
        default=str(root / "data" / "glaive" / "train.csv"),
        help="Path to Glaive train.csv",
    )
    parser.add_argument(
        "--glaive-output",
        default=str(root / "data" / "processed" / "glaive_cleaned.jsonl"),
        help="Output JSONL for cleaned Glaive documents",
    )
    parser.add_argument(
        "--glaive-max-rows",
        type=int,
        default=None,
        help="Optional row limit for Glaive processing",
    )
    parser.add_argument(
        "--so-questions",
        default=str(root / "data" / "stackoverflow" / "Questions.csv"),
        help="Path to StackOverflow Questions.csv",
    )
    parser.add_argument(
        "--so-answers",
        default=str(root / "data" / "stackoverflow" / "Answers.csv"),
        help="Path to StackOverflow Answers.csv",
    )
    parser.add_argument(
        "--so-output",
        default=str(root / "data" / "processed" / "stackoverflow_docs.jsonl"),
        help="Output JSONL for cleaned StackOverflow documents",
    )
    parser.add_argument(
        "--so-max-questions",
        type=int,
        default=100000,
        help="Maximum number of StackOverflow questions to process",
    )
    parser.add_argument(
        "--so-batch-size",
        type=int,
        default=5000,
        help="Batch size for chunked CSV processing",
    )
    parser.add_argument(
        "--skip-glaive",
        action="store_true",
        help="Skip Glaive ingestion",
    )
    parser.add_argument(
        "--skip-stackoverflow",
        action="store_true",
        help="Skip StackOverflow ingestion",
    )
    args = parser.parse_args()

    summary = run_ingestion(
        glaive_input=Path(args.glaive_input),
        glaive_output=Path(args.glaive_output),
        glaive_max_rows=args.glaive_max_rows,
        so_questions=Path(args.so_questions),
        so_answers=Path(args.so_answers),
        so_output=Path(args.so_output),
        so_max_questions=args.so_max_questions,
        so_batch_size=args.so_batch_size,
        skip_glaive=args.skip_glaive,
        skip_stackoverflow=args.skip_stackoverflow,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
