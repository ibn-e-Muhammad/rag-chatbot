"""Gemini-based response generation for the RAG system."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import List

_WHITESPACE_RE = re.compile(r"\s+")

FALLBACK_MESSAGE = "I don't have enough information to answer that."
DEFAULT_MODEL = "gemini-1.5-flash"


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#._-]*")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


def _context_is_empty(context: str) -> bool:
    normalized = _normalize_text(context)
    if not normalized:
        return True
    scrubbed = re.sub(
        r"(Source:\s*\[[^\]]+\]|Question:|Answer:)",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    return not re.search(r"\w", scrubbed)


def _tokenize(value: str) -> set[str]:
    return {token for token in _WORD_RE.findall(_normalize_text(value).lower()) if token not in _STOPWORDS}


def _parse_context_blocks(context: str) -> List[dict[str, str]]:
    blocks: List[dict[str, str]] = []
    chunks = re.split(r"(?=Source:\s*\[[^\]]+\])", context, flags=re.IGNORECASE)
    for chunk in chunks:
        text = _normalize_text(chunk)
        if not text:
            continue
        source_match = re.search(r"Source:\s*\[([^\]]+)\]", text, flags=re.IGNORECASE)
        question_match = re.search(r"Question:\s*(.*?)\s*Answer:", text, flags=re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"Answer:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        blocks.append(
            {
                "source": source_match.group(1).strip() if source_match else "",
                "question": _normalize_text(question_match.group(1)) if question_match else "",
                "answer": _normalize_text(answer_match.group(1)) if answer_match else text,
                "raw": text,
            }
        )
    return blocks


def _context_query_overlap(query: str, context: str) -> int:
    query_tokens = _tokenize(query)
    context_tokens = _tokenize(context)
    return len(query_tokens & context_tokens)


def _choose_context_block(query: str, context: str) -> dict[str, str] | None:
    blocks = _parse_context_blocks(context)
    if not blocks:
        return None

    query_tokens = _tokenize(query)

    def score(block: dict[str, str]) -> tuple[int, int, int]:
        source = block.get("source", "")
        source_bonus = 2 if source.lower().startswith("glaive") else 1 if source.lower().startswith("stack") else 0
        block_text = f"{block.get('question', '')} {block.get('answer', '')}"
        overlap = len(query_tokens & _tokenize(block_text))
        length_bonus = 1 if len(block.get("answer", "")) > 80 else 0
        return (source_bonus, overlap, length_bonus)

    return max(blocks, key=score)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", _normalize_text(text))
    return [part.strip() for part in parts if part.strip()]


def _extract_example(answer: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        snippet = _normalize_text(fenced.group(1))
        if snippet:
            return snippet
    example_match = re.search(r"Example:\s*(.*)", answer, flags=re.IGNORECASE | re.DOTALL)
    if example_match:
        snippet = _normalize_text(example_match.group(1))
        if snippet:
            return snippet
    return "No example provided in the retrieved context."


def _offline_response(query: str, context: str) -> str:
    if _context_query_overlap(query, context) == 0:
        return FALLBACK_MESSAGE

    block = _choose_context_block(query, context)
    if not block:
        return FALLBACK_MESSAGE

    source = block.get("source", "").strip() or "Unknown"
    question = block.get("question", "").strip() or query
    answer = block.get("answer", "").strip()
    if not answer:
        return FALLBACK_MESSAGE

    sentences = _split_sentences(answer)
    short_definition = sentences[0] if sentences else answer
    if len(sentences) > 1 and not sentences[1].lower().startswith("example:"):
        simple_explanation = sentences[1]
    else:
        simple_explanation = short_definition
    example = _extract_example(answer)

    return (
        f"Source: [{source}]\n"
        f"Question: {question}\n"
        f"Answer:\n"
        f"Short definition: {short_definition}\n"
        f"Simple explanation: {simple_explanation}\n"
        f"Example: {example}"
    )


def _resolve_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    return api_key


class _GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = self._build_client()

    def _build_client(self):
        try:
            from google import genai  # type: ignore

            return genai.Client(api_key=self._api_key)
        except ImportError as exc:
            raise RuntimeError("Missing google-genai library. Install it using 'pip install google-genai'.") from exc

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            chunk = parts[0]
            part_text = getattr(chunk, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text.strip()
        return ""


@lru_cache(maxsize=1)
def _get_client() -> _GeminiClient:
    model = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    return _GeminiClient(api_key=_resolve_api_key(), model=model)


def _build_prompt(query: str, context: str) -> str:
    instructions: List[str] = [
        "You are a domain-restricted Python tutor.",
        "Answer ONLY from the provided context.",
        f"If the answer is not present, say: '{FALLBACK_MESSAGE}'.",
        "Do NOT use external knowledge or guess.",
        "Include source labels like [Glaive] or [StackOverflow] when citing facts.",
        "Follow this structure:",
        "1. Short definition",
        "2. Simple explanation",
        "3. Example (if applicable)",
        "Use friendly, clear, educational language and avoid unnecessary jargon.",
        "Prefer Glaive context over StackOverflow when both are present.",
    ]
    return (
        "Instructions:\n"
        + "\n".join(instructions)
        + "\n\nContext:\n"
        + context
        + "\n\nUser query:\n"
        + query
    )


def generate_response(query: str, context: str) -> str:
    """Generate a response using Gemini grounded in retrieved context."""
    normalized_query = _normalize_text(query)
    if not normalized_query or _context_is_empty(context):
        return FALLBACK_MESSAGE

    try:
        prompt = _build_prompt(normalized_query, context)
        response = _get_client().generate(prompt)
        if response:
            return response
    except Exception:
        pass

    return _offline_response(normalized_query, context)
