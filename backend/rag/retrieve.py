"""Hybrid retrieval combining tree results with vector fallback."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Tuple

from .retrieve_tree import retrieve_tree
from .vector import retrieve_vector

_WHITESPACE_RE = re.compile(r"\s+")

MAX_TOTAL_RESULTS = 6
TREE_STRONG_THRESHOLD = 3

_TECHNICAL_HINTS = {
    "api",
    "backend",
    "class",
    "code",
    "debug",
    "error",
    "exception",
    "fastapi",
    "flask",
    "framework",
    "function",
    "header",
    "http",
    "import",
    "json",
    "loop",
    "middleware",
    "package",
    "python",
    "request",
    "response",
    "socket",
    "sql",
    "thread",
    "traceback",
    "vector",
    "web",
    "while",
}


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _dedup_key(item: Mapping[str, object]) -> Tuple[str, str]:
    question = _normalize_text(str(item.get("question") or ""))
    answer = _normalize_text(str(item.get("answer") or ""))
    return question, answer


def _source_label(item: Mapping[str, object]) -> str:
    label = str(item.get("source_label") or "").strip()
    if label:
        lowered = label.lower()
        if lowered.startswith("stack"):
            return "StackOverflow"
        if lowered.startswith("glaive"):
            return "Glaive"

    source = str(item.get("source") or "").lower()
    if "stack" in source:
        return "StackOverflow"
    if "glaive" in source:
        return "Glaive"

    if any(key in item for key in ("question_id", "answer_id", "score")):
        return "StackOverflow"
    return "Glaive"


def _is_technical_query(query: str) -> bool:
    tokens = set(_normalize_text(query).lower().split())
    return any(token in _TECHNICAL_HINTS for token in tokens)


def _format_item(item: Mapping[str, object]) -> str:
    question = _normalize_text(str(item.get("question") or ""))
    answer = _normalize_text(str(item.get("answer") or ""))
    if not question and not answer:
        return ""
    label = _source_label(item)
    return f"Source: [{label}]\nQuestion: {question}\nAnswer: {answer}\n"


def _merge_results(
    primary: List[Dict[str, object]],
    secondary: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    seen: set[Tuple[str, str]] = set()
    merged: List[Dict[str, object]] = []
    for item in primary + secondary:
        key = _dedup_key(item)
        if not any(key):
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= MAX_TOTAL_RESULTS:
            break
    return merged


def retrieve(query: str) -> str:
    """Retrieve and format hybrid RAG context."""
    normalized = _normalize_text(query)
    if not normalized:
        return ""

    tree_results = retrieve_tree(normalized)
    use_vector = bool(tree_results) and len(tree_results) < TREE_STRONG_THRESHOLD and _is_technical_query(normalized)
    vector_results = retrieve_vector(normalized) if use_vector else []

    merged = _merge_results(tree_results, vector_results)
    blocks = [_format_item(item) for item in merged]
    blocks = [block for block in blocks if block]
    if not blocks:
        return ""
    return "\n".join(blocks).strip()
