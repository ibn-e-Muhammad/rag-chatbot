"""Build a hierarchical topic tree from cleaned Glaive and StackOverflow datasets."""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import orjson

_WHITESPACE_RE = re.compile(r"\s+")
_QUESTION_ANSWER_RE = re.compile(r"Question:\s*(.*?)\s*Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
_DEDUP_CLEAN_RE = re.compile(r"[^a-z0-9]+")

LOGGER = logging.getLogger(__name__)
_PROGRESS_EVERY = 50_000

ALLOWED_TOPICS: List[str] = [
    "variables",
    "data types",
    "loops",
    "functions",
    "lists",
    "dictionaries",
    "strings",
    "file handling",
    "exceptions",
    "OOP",
    "modules",
    "libraries",
    "debugging",
    "system / OS",
    "APIs",
    "web / backend",
    "general programming",
]

DEFAULT_TOPIC = "general programming"
DEFAULT_SUBTOPIC = "general"

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "variables": [
        "variable",
        "assignment",
        "scope",
        "global variable",
        "local variable",
        "mutable",
        "immutable",
        "constant",
    ],
    "data types": [
        "data type",
        "datatype",
        "int",
        "integer",
        "float",
        "double",
        "boolean",
        "bool",
        "none",
        "null",
        "cast",
        "conversion",
        "type hint",
    ],
    "loops": [
        "for loop",
        "while loop",
        "for-loop",
        "while-loop",
        "foreach",
        "for each",
        "loop",
        "iterate",
        "iteration",
        "for(",
        "while(",
    ],
    "functions": [
        "function",
        "def",
        "lambda",
        "argument",
        "parameter",
        "return",
        "callable",
        "function call",
    ],
    "lists": [
        "list",
        "array",
        "slice",
        "index",
        "append",
        "extend",
        "list comprehension",
    ],
    "dictionaries": [
        "dict",
        "dictionary",
        "hash map",
        "hashmap",
        "mapping",
        "key value",
        "key-value",
    ],
    "strings": [
        "string",
        "str",
        "substring",
        "split",
        "join",
        "replace",
        "format",
        "f-string",
        "regex",
        "regular expression",
    ],
    "file handling": [
        "file",
        "open(",
        "readline",
        "writeline",
        "append",
        "directory",
        "folder",
    ],
    "exceptions": [
        "exception",
        "error handling",
        "try",
        "except",
        "raise",
        "finally",
        "traceback",
    ],
    "OOP": [
        "class",
        "inheritance",
        "polymorphism",
        "encapsulation",
        "self",
        "instance",
    ],
    "modules": [
        "import",
        "module",
        "namespace",
        "__init__",
        "relative import",
    ],
    "libraries": [
        "library",
        "framework",
        "dependency",
        "pip",
        "pip install",
        "package manager",
    ],
    "debugging": [
        "debug",
        "debugger",
        "breakpoint",
        "logging",
        "log",
        "traceback",
    ],
    "system / OS": [
        "windows",
        "linux",
        "mac",
        "osx",
        "unix",
        "bash",
        "shell",
        "terminal",
        "command line",
        "environment variable",
        "process",
        "permission",
    ],
    "APIs": [
        "api",
        "endpoint",
        "rest",
        "graphql",
        "request",
        "response",
    ],
    "web / backend": [
        "web",
        "backend",
        "server",
        "http",
        "url",
        "route",
        "fastapi",
        "flask",
        "django",
        "node",
    ],
    "general programming": [],
}

_LOOP_SUBTOPIC_RULES: List[Tuple[str, List[str]]] = [
    ("for loops", ["for loop", "for-loop", "foreach", "for each", "for(", "for i in"]),
    ("while loops", ["while loop", "while-loop", "do while", "do-while", "while("]),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    only_words = re.fullmatch(r"[A-Za-z0-9 ]+", keyword) is not None
    escaped = re.escape(keyword).replace("\\ ", r"\s+")
    if only_words:
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


_TOPIC_PATTERNS: List[Tuple[str, List[re.Pattern[str]]]] = [
    (topic, [_keyword_pattern(keyword) for keyword in keywords])
    for topic, keywords in _TOPIC_KEYWORDS.items()
]

_SUBTOPIC_PATTERNS: List[Tuple[str, List[re.Pattern[str]]]] = [
    (name, [_keyword_pattern(keyword) for keyword in keywords])
    for name, keywords in _LOOP_SUBTOPIC_RULES
]


def _score_patterns(text: str, patterns: Iterable[re.Pattern[str]]) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _select_topic(text: str) -> str:
    normalized = _normalize_text(text).lower()
    scores: Dict[str, int] = {}
    for topic, patterns in _TOPIC_PATTERNS:
        if not patterns:
            continue
        score = _score_patterns(normalized, patterns)
        if score:
            scores[topic] = score
    if not scores:
        return DEFAULT_TOPIC
    max_score = max(scores.values())
    for topic, _ in _TOPIC_PATTERNS:
        if scores.get(topic) == max_score:
            return topic
    return DEFAULT_TOPIC


def _select_subtopic(topic: str, text: str) -> str:
    if topic != "loops":
        return DEFAULT_SUBTOPIC
    normalized = _normalize_text(text).lower()
    scores: Dict[str, int] = {}
    for name, patterns in _SUBTOPIC_PATTERNS:
        score = _score_patterns(normalized, patterns)
        if score:
            scores[name] = score
    if not scores:
        return DEFAULT_SUBTOPIC
    max_score = max(scores.values())
    for name, _ in _SUBTOPIC_PATTERNS:
        if scores.get(name) == max_score:
            return name
    return DEFAULT_SUBTOPIC


def _dedup_key(question: str, answer: str) -> str:
    combined = f"{question} {answer}".lower()
    combined = _DEDUP_CLEAN_RE.sub(" ", combined)
    return _normalize_text(combined)


def _split_document(document: str) -> Tuple[str, str]:
    if not document:
        return "", ""
    match = _QUESTION_ANSWER_RE.search(document)
    if not match:
        return _normalize_text(document), ""
    return _normalize_text(match.group(1)), _normalize_text(match.group(2))


def _extract_question_answer(record: Dict[str, object]) -> Tuple[str, str]:
    question = str(record.get("question") or "")
    answer = str(record.get("answer") or "")

    # Common alternate schemas used across cleaned/processed datasets.
    if not question:
        question = str(
            record.get("title")
            or record.get("prompt")
            or record.get("instruction")
            or ""
        )
    if not answer:
        answer = str(
            record.get("response")
            or record.get("output")
            or record.get("accepted_answer")
            or record.get("body")
            or ""
        )

    if not question or not answer:
        document = str(record.get("document") or record.get("content") or record.get("text") or "")
        doc_question, doc_answer = _split_document(document)
        if not question:
            question = doc_question
        if not answer:
            answer = doc_answer

    # Last resort: keep the raw document-like text as answer if available.
    if not answer:
        answer = _normalize_text(str(record.get("document") or record.get("content") or record.get("text") or ""))

    question = _normalize_text(question)
    answer = _normalize_text(answer)
    if not getattr(_extract_question_answer, "_printed", False):
        LOGGER.info("First extracted Q/A sample: question=%r answer_len=%s", question, len(answer))
        _extract_question_answer._printed = True
    return question, answer


def _build_node(question: str, answer: str, record: Dict[str, object], source: str) -> Dict[str, object]:
    node: Dict[str, object] = {"question": question, "answer": answer, "source": source}
    if source == "stackoverflow":
        score = record.get("score")
        if score is not None:
            node["score"] = score
    return node


def _read_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    printed = False
    LOGGER.info("Reading JSONL input: %s", path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = orjson.loads(stripped)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON at %s:%s (%s)", path, line_number, exc)
                continue
            if not printed:
                LOGGER.info("First parsed record from %s: %s", path, record)
                printed = True
            yield record
    LOGGER.info("Finished reading JSONL input: %s", path)


def _build_source_tree(
    path: Path,
    source: str,
    max_records_per_source: int | None = None,
) -> Tuple[Dict[str, Dict[str, List[Dict[str, object]]]], int, int, int]:
    tree: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    read = 0
    skipped_no_qa = 0
    deduped_local = 0
    seen_local: set[str] = set()

    for record in _read_jsonl(path):
        if max_records_per_source is not None and read >= max_records_per_source:
            LOGGER.info("Reached %s max_records_per_source=%s", source, max_records_per_source)
            break
        read += 1

        question, answer = _extract_question_answer(record)
        if not question or not answer:
            skipped_no_qa += 1
            continue

        dedup = _dedup_key(question, answer)
        if dedup in seen_local:
            deduped_local += 1
            continue
        seen_local.add(dedup)

        text = f"{question} {answer}"
        topic = _select_topic(text)
        if topic not in ALLOWED_TOPICS:
            topic = DEFAULT_TOPIC
        subtopic = _select_subtopic(topic, text)
        node = _build_node(question, answer, record, source)
        tree.setdefault(topic, {}).setdefault(subtopic, []).append(node)

        if read % _PROGRESS_EVERY == 0:
            LOGGER.info(
                "[%s] Progress: read=%s skipped_no_qa=%s deduped_local=%s",
                source,
                read,
                skipped_no_qa,
                deduped_local,
            )

    return tree, read, skipped_no_qa, deduped_local


def _merge_source_tree(
    destination: Dict[str, Dict[str, List[Dict[str, object]]]],
    source_tree: Dict[str, Dict[str, List[Dict[str, object]]]],
    seen_global: set[str],
) -> Tuple[int, int]:
    kept = 0
    deduped_global = 0
    for topic, subtopics in source_tree.items():
        topic_bucket = destination.setdefault(topic, {})
        for subtopic, entries in subtopics.items():
            subtopic_bucket = topic_bucket.setdefault(subtopic, [])
            for node in entries:
                dedup = _dedup_key(str(node.get("question") or ""), str(node.get("answer") or ""))
                if dedup in seen_global:
                    deduped_global += 1
                    continue
                seen_global.add(dedup)
                kept += 1
                subtopic_bucket.append(node)
    return kept, deduped_global


def build_tree_index(
    glaive_path: Path,
    stackoverflow_path: Path,
    max_records_per_source: int | None = None,
) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    LOGGER.info("Starting tree index build")
    tree: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    seen_global: set[str] = set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_glaive = executor.submit(_build_source_tree, glaive_path, "glaive", max_records_per_source)
        future_stackoverflow = executor.submit(
            _build_source_tree,
            stackoverflow_path,
            "stackoverflow",
            max_records_per_source,
        )
        glaive_tree, glaive_read, glaive_skipped_no_qa, glaive_deduped_local = future_glaive.result()
        stack_tree, stack_read, stack_skipped_no_qa, stack_deduped_local = future_stackoverflow.result()

    kept_glaive, deduped_cross_glaive = _merge_source_tree(tree, glaive_tree, seen_global)
    kept_stack, deduped_cross_stack = _merge_source_tree(tree, stack_tree, seen_global)

    total_read = glaive_read + stack_read
    kept = kept_glaive + kept_stack
    skipped_no_qa = glaive_skipped_no_qa + stack_skipped_no_qa
    deduped = glaive_deduped_local + stack_deduped_local + deduped_cross_glaive + deduped_cross_stack

    LOGGER.info(
        "Tree index stats: total_read=%s kept=%s skipped-no-QA=%s deduped=%s",
        total_read,
        kept,
        skipped_no_qa,
        deduped,
    )
    return tree


def write_tree_index(tree: Dict[str, Dict[str, List[Dict[str, object]]]], output_path: Path) -> None:
    LOGGER.info("Writing tree index to: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(orjson.dumps(tree, option=orjson.OPT_INDENT_2).decode("utf-8"))
    LOGGER.info("Finished writing tree index")


def _resolve_tree_index_path(output_path: Path) -> Path:
    if output_path.name == "tree_index.json":
        return output_path
    if output_path.suffix:
        return output_path.with_name("tree_index.json")
    return output_path / "tree_index.json"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Build hierarchical tree index for RAG retrieval.")
    parser.add_argument(
        "--glaive-input",
        default=str(root / "data" / "processed" / "glaive_cleaned.jsonl"),
        help="Path to cleaned Glaive JSONL file",
    )
    parser.add_argument(
        "--stackoverflow-input",
        default=str(root / "data" / "processed" / "stackoverflow_docs.jsonl"),
        help="Path to cleaned StackOverflow JSONL file",
    )
    parser.add_argument(
        "--output",
        default=str(root / "vectorstore" / "tree_index.json"),
        help="Path to output tree_index.json",
    )
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional cap for records read from each source (useful for quick test runs)",
    )
    args = parser.parse_args()
    LOGGER.info("Resolved repository root: %s", root)
    LOGGER.info("Input paths: glaive=%s stackoverflow=%s", args.glaive_input, args.stackoverflow_input)
    LOGGER.info("Requested output path: %s", args.output)

    tree = build_tree_index(
        glaive_path=Path(args.glaive_input),
        stackoverflow_path=Path(args.stackoverflow_input),
        max_records_per_source=args.max_records_per_source,
    )
    output_path = _resolve_tree_index_path(Path(args.output))
    LOGGER.info("Resolved output path: %s", output_path)
    write_tree_index(tree, output_path)


if __name__ == "__main__":
    main()
