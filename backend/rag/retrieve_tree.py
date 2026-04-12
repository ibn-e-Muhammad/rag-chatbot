"""Tree-based retrieval over the prebuilt hierarchical index."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#._-]*")

DEFAULT_TOPIC = "general programming"
DEFAULT_SUBTOPIC = "general"
MAX_RESULTS = 5

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "variables": ["variable", "assignment", "scope", "global variable", "local variable", "constant"],
    "data types": ["data type", "datatype", "int", "integer", "float", "double", "boolean", "bool", "cast"],
    "loops": ["for loop", "while loop", "foreach", "loop", "iterate", "iteration", "for(", "while("],
    "functions": ["function", "def", "lambda", "argument", "parameter", "return", "callable"],
    "lists": ["list", "array", "slice", "index", "append", "extend", "list comprehension"],
    "dictionaries": ["dict", "dictionary", "hash map", "hashmap", "mapping", "key value"],
    "strings": ["string", "str", "substring", "split", "join", "replace", "format", "regex"],
    "file handling": ["file", "open(", "readline", "writeline", "directory", "folder", "path"],
    "exceptions": ["exception", "error handling", "try", "except", "raise", "finally", "traceback"],
    "OOP": ["class", "inheritance", "polymorphism", "encapsulation", "self", "instance", "object oriented"],
    "modules": ["import", "module", "namespace", "__init__", "relative import"],
    "libraries": ["library", "framework", "dependency", "pip", "pip install", "package manager"],
    "debugging": ["debug", "debugger", "breakpoint", "logging", "traceback"],
    "system / OS": ["windows", "linux", "mac", "unix", "bash", "shell", "terminal", "process", "permission"],
    "APIs": ["api", "apis", "endpoint", "rest", "graphql", "request", "response"],
    "web / backend": ["web", "backend", "server", "http", "url", "route", "fastapi", "flask", "django"],
    "general programming": [],
}

_LOOP_SUBTOPIC_RULES: List[Tuple[str, List[str]]] = [
    ("for loops", ["for loop", "for-loop", "foreach", "for each", "for(", "for i in"]),
    ("while loops", ["while loop", "while-loop", "do while", "do-while", "while("]),
]

_STOPWORDS = {
    "a",
    "an",
    "and",
    "can",
    "are",
    "as",
    "at",
    "be",
    "could",
    "by",
    "for",
    "from",
    "how",
    "did",
    "do",
    "does",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "would",
    "when",
    "where",
    "why",
    "with",
}


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
_LOOP_SUBTOPIC_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    subtopic: [_keyword_pattern(keyword) for keyword in keywords]
    for subtopic, keywords in _LOOP_SUBTOPIC_RULES
}


def _score_patterns(text: str, patterns: Iterable[re.Pattern[str]]) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _extract_keywords(query: str) -> List[str]:
    words = _WORD_RE.findall(_normalize_text(query).lower())
    return [word for word in words if (len(word) > 1 or word in {"c", "r"}) and word not in _STOPWORDS]


def _build_query_phrases(keywords: Sequence[str]) -> List[str]:
    phrases: List[str] = []
    for size in (3, 2):
        for idx in range(0, max(0, len(keywords) - size + 1)):
            phrase = " ".join(keywords[idx : idx + size])
            if phrase:
                phrases.append(phrase)
    return phrases


def _topic_token_set(topic: str) -> set[str]:
    return set(_WORD_RE.findall(topic.lower()))


@lru_cache(maxsize=1)
def _load_tree_index() -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    path = _repo_root() / "vectorstore" / "tree_index.json"
    with path.open("rb") as handle:
        payload = handle.read()
    try:
        import orjson  # type: ignore

        return orjson.loads(payload)
    except ImportError:
        return json.loads(payload.decode("utf-8"))


def _select_topic(query_text: str, keywords: Sequence[str], tree: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]]) -> str:
    keyword_set = set(keywords)
    scores: Dict[str, int] = {}
    for topic in tree.keys():
        score = 0
        topic_tokens = _topic_token_set(topic)
        score += 3 * len(topic_tokens & keyword_set)
        if topic.lower() in query_text:
            score += 6
        for rule_topic, patterns in _TOPIC_PATTERNS:
            if topic.lower() != rule_topic.lower():
                continue
            score += 5 * _score_patterns(query_text, patterns)
            break
        if score > 0:
            scores[topic] = score

    if not scores:
        if DEFAULT_TOPIC in tree:
            return DEFAULT_TOPIC
        return next(iter(tree.keys()), "")
    return max(scores.items(), key=lambda item: item[1])[0]


def _select_subtopic(topic: str, query_text: str, keywords: Sequence[str], tree: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]]) -> Tuple[str, Dict[str, int]]:
    subtopics = tree.get(topic, {})
    if not subtopics:
        return DEFAULT_SUBTOPIC, {}

    keyword_set = set(keywords)
    scores: Dict[str, int] = {}
    for subtopic in subtopics.keys():
        score = 0
        subtopic_tokens = set(_WORD_RE.findall(subtopic.lower()))
        score += 2 * len(subtopic_tokens & keyword_set)
        if subtopic.lower() in query_text:
            score += 4
        if topic == "loops":
            score += 5 * _score_patterns(query_text, _LOOP_SUBTOPIC_PATTERNS.get(subtopic, []))
        if subtopic == DEFAULT_SUBTOPIC:
            score += 1
        scores[subtopic] = score

    best_subtopic = max(scores.items(), key=lambda item: item[1])[0]
    if scores.get(best_subtopic, 0) <= 0 and DEFAULT_SUBTOPIC in subtopics:
        return DEFAULT_SUBTOPIC, scores
    return best_subtopic, scores


def _score_node(query_text: str, keywords: Sequence[str], phrases: Sequence[str], node: Mapping[str, object]) -> int:
    question = _normalize_text(str(node.get("question") or "")).lower()
    answer = _normalize_text(str(node.get("answer") or "")).lower()
    if not question and not answer:
        return 0

    question_tokens = set(_WORD_RE.findall(question))
    answer_tokens = set(_WORD_RE.findall(answer))
    score = 0
    for phrase in phrases:
        if phrase in question:
            score += 7
        elif phrase in answer:
            score += 3
    for keyword in keywords:
        if keyword in question_tokens:
            score += 3
        elif keyword in answer_tokens:
            score += 1
    score += 2 * len(question_tokens & set(keywords))
    if query_text in question:
        score += 8

    if score <= 0:
        return 0

    stack_score = node.get("score")
    if isinstance(stack_score, (int, float)):
        score += min(int(stack_score // 25), 2)
    return score


def retrieve_tree(query: str) -> List[Dict[str, object]]:
    """Retrieve the most relevant Q&A nodes from tree_index.json without embeddings."""
    normalized_query = _normalize_text(query).lower()
    if not normalized_query:
        return []

    tree = _load_tree_index()
    if not tree:
        return []

    keywords = _extract_keywords(normalized_query)
    phrases = _build_query_phrases(keywords)

    topic = _select_topic(normalized_query, keywords, tree)
    if not topic or topic not in tree:
        return []

    subtopic, subtopic_scores = _select_subtopic(topic, normalized_query, keywords, tree)
    topic_bucket = tree.get(topic, {})
    if not topic_bucket:
        return []

    candidate_subtopics: List[str] = []
    if subtopic in topic_bucket:
        candidate_subtopics.append(subtopic)
    if DEFAULT_SUBTOPIC in topic_bucket and DEFAULT_SUBTOPIC not in candidate_subtopics:
        candidate_subtopics.append(DEFAULT_SUBTOPIC)
    for name, _ in sorted(subtopic_scores.items(), key=lambda item: item[1], reverse=True):
        if name in topic_bucket and name not in candidate_subtopics and subtopic_scores.get(name, 0) > 0:
            candidate_subtopics.append(name)
        if len(candidate_subtopics) >= 3:
            break

    ranked: List[Tuple[int, str, Dict[str, object]]] = []
    seen: set[Tuple[str, str]] = set()
    minimum_score = 8 if topic == DEFAULT_TOPIC else 1
    for branch in candidate_subtopics:
        for node in topic_bucket.get(branch, []):
            score = _score_node(normalized_query, keywords, phrases, node)
            if score < minimum_score:
                continue
            question = str(node.get("question") or "")
            answer = str(node.get("answer") or "")
            dedup_key = (question, answer)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            enriched = dict(node)
            enriched["topic"] = topic
            enriched["subtopic"] = branch
            ranked.append((score, branch, enriched))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[2] for item in ranked[:MAX_RESULTS]]

