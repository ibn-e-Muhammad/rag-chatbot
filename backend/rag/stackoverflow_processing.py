"""StackOverflow Questions/Answers ingestion and cleaning.

Output format: JSONL, one record per line with:
- question_id
- answer_id
- score
- document: "Question: ...\nAnswer: ..."
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
from html import unescape


CODE_BLOCK_RE = re.compile(r"(?is)<pre\b[^>]*>.*?</pre>")
INLINE_CODE_RE = re.compile(r"(?is)<code\b[^>]*>.*?</code>")
HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def _batch_reader(path: Path, batch_size: int) -> Iterator[List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        batch: List[Dict[str, str]] = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = CODE_BLOCK_RE.sub(" ", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def process_stackoverflow(
    questions_path: str,
    answers_path: str,
    output_path: str,
    max_questions: int = 100000,
    batch_size: int = 5000,
) -> Dict[str, int]:
    """Process StackOverflow CSVs and write cleaned QA documents as JSONL."""
    questions_file = Path(questions_path)
    answers_file = Path(answers_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    questions: Dict[int, str] = {}
    processed_questions = 0

    for batch in _batch_reader(questions_file, batch_size=batch_size):
        for row in batch:
            if processed_questions >= max_questions:
                break
            qid = _to_int(row.get("Id"))
            if qid <= 0:
                continue
            title = row.get("Title", "") or ""
            body = row.get("Body", "") or ""
            question_text = _clean_text(f"{title}\n{body}")
            if not question_text:
                continue
            questions[qid] = question_text
            processed_questions += 1
        if processed_questions >= max_questions:
            break

    best_answers: Dict[int, Tuple[int, int, str]] = {}
    scanned_answers = 0
    kept_answers = 0

    for batch in _batch_reader(answers_file, batch_size=batch_size):
        for row in batch:
            scanned_answers += 1
            parent_id = _to_int(row.get("ParentId"))
            if parent_id not in questions:
                continue

            answer_text = _clean_text(row.get("Body", "") or "")
            if not answer_text:
                continue
            if len(answer_text.split()) > 1000:
                continue

            score = _to_int(row.get("Score"))
            answer_id = _to_int(row.get("Id"))
            existing = best_answers.get(parent_id)
            if (
                existing is None
                or score > existing[0]
                or (score == existing[0] and len(answer_text) > len(existing[2]))
            ):
                if existing is None:
                    kept_answers += 1
                best_answers[parent_id] = (score, answer_id, answer_text)

    written_docs = 0
    with out_file.open("w", encoding="utf-8", newline="") as f:
        for qid, question_text in questions.items():
            answer = best_answers.get(qid)
            if answer is None:
                continue
            score, answer_id, answer_text = answer
            record = {
                "question_id": qid,
                "answer_id": answer_id,
                "score": score,
                "document": f"Question: {question_text}\nAnswer: {answer_text}",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_docs += 1

    return {
        "processed_questions": processed_questions,
        "scanned_answers": scanned_answers,
        "matched_questions_with_answers": kept_answers,
        "written_documents": written_docs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process StackOverflow Questions/Answers into cleaned JSONL docs."
    )
    parser.add_argument(
        "--questions-path",
        default="data\\stackoverflow\\Questions.csv",
        help="Path to Questions.csv",
    )
    parser.add_argument(
        "--answers-path",
        default="data\\stackoverflow\\Answers.csv",
        help="Path to Answers.csv",
    )
    parser.add_argument(
        "--output-path",
        default="data\\processed\\stackoverflow_docs.jsonl",
        help="Path to output JSONL file under data\\processed\\",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=100000,
        help="Maximum number of questions to process from the start of Questions.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for chunked CSV processing",
    )
    args = parser.parse_args()

    summary = process_stackoverflow(
        questions_path=args.questions_path,
        answers_path=args.answers_path,
        output_path=args.output_path,
        max_questions=args.max_questions,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False))
