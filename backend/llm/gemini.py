"""Gemini-based response generation for the RAG system."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal

_WHITESPACE_RE = re.compile(r"\s+")

FALLBACK_MESSAGE = "I don't have enough information to answer that."
DEFAULT_MODEL = "gemini-2.5-flash-lite"
MAX_CONTEXT_BLOCKS = 3
MAX_ANSWER_CHARS_PER_BLOCK = 320
MIN_QUERY_CONTEXT_OVERLAP = 2

LOGGER = logging.getLogger(__name__)

ResponseMode = Literal["online", "offline_fallback", "hard_fail"]


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


@lru_cache(maxsize=1)
def _load_local_env() -> None:
    """Load local .env in dev without overriding real host environment vars."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return

    repo_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_env.exists():
        load_dotenv(dotenv_path=repo_env, override=False)


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


def _response_policy() -> ResponseMode:
    raw = _normalize_text(os.getenv("RAG_LLM_RESPONSE_POLICY") or "graceful").lower()
    if raw in {"strict", "hard_fail", "hard-fail"}:
        return "hard_fail"
    return "offline_fallback"


def _tokenize(value: str) -> set[str]:
    tokens = []
    for token in _WORD_RE.findall(_normalize_text(value).lower()):
        if token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return set(tokens)


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


def _compress_context_for_prompt(query: str, context: str) -> str:
    blocks = _parse_context_blocks(context)
    if not blocks:
        return _normalize_text(context)

    query_tokens = _tokenize(query)

    def rank(block: Dict[str, str]) -> tuple[int, int, int]:
        source = block.get("source", "")
        source_bonus = 2 if source.lower().startswith("glaive") else 1 if source.lower().startswith("stack") else 0
        text = f"{block.get('question', '')} {block.get('answer', '')}"
        overlap = len(query_tokens & _tokenize(text))
        return (overlap, source_bonus, len(block.get("answer", "")))

    top_blocks = sorted(blocks, key=rank, reverse=True)[:MAX_CONTEXT_BLOCKS]
    compact_lines: List[str] = []
    for idx, block in enumerate(top_blocks, start=1):
        source = block.get("source", "Unknown") or "Unknown"
        question = _normalize_text(block.get("question", ""))
        answer = _normalize_text(block.get("answer", ""))
        answer = answer[:MAX_ANSWER_CHARS_PER_BLOCK]
        compact_lines.append(
            f"Evidence {idx} [{source}]\n"
            f"Q: {question}\n"
            f"A: {answer}"
        )
    return "\n\n".join(compact_lines)


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
    if _context_query_overlap(query, context) < MIN_QUERY_CONTEXT_OVERLAP:
        return FALLBACK_MESSAGE

    block = _choose_context_block(query, context)
    if not block:
        return FALLBACK_MESSAGE

    source = block.get("source", "").strip() or "Unknown"
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
        f"Short definition: {short_definition}\n"
        f"Simple explanation: {simple_explanation}\n"
        f"Example: {example}\n"
        f"Sources: [{source}]"
    )


def _is_context_dump(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    markers = ["Source:", "Question:", "Answer:"]
    marker_hits = sum(normalized.count(marker) for marker in markers)
    return marker_hits >= 3


def _validate_generated_response(query: str, context: str, response: str) -> str:
    normalized = _normalize_text(response)
    if not normalized:
        return ""
    if normalized == FALLBACK_MESSAGE:
        return FALLBACK_MESSAGE
    if _context_query_overlap(query, context) < MIN_QUERY_CONTEXT_OVERLAP:
        return FALLBACK_MESSAGE
    normalized_lower = normalized.lower()
    required_sections = ("short definition:", "simple explanation:", "example:", "sources:")
    if not all(section in normalized_lower for section in required_sections):
        return ""
    if _is_context_dump(normalized):
        return ""
    return response.strip()


def _resolve_api_key() -> str:
    _load_local_env()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    os.environ["GEMINI_API_KEY"] = api_key
    return api_key


class _GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = self._build_client()
        self._resolved_model: str | None = None

    def _build_client(self):
        try:
            from google import genai  # type: ignore

            return genai.Client()
        except ImportError as exc:
            raise RuntimeError("Missing google-genai library. Install it using 'pip install google-genai'.") from exc

    def list_models(self) -> List[object]:
        try:
            return list(self._client.models.list())
        except Exception as exc:
            LOGGER.warning("Unable to list Gemini models: %s", exc)
            return []

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if model_name.startswith("models/"):
            return model_name.split("/", 1)[1]
        return model_name

    def _resolve_generation_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model

        configured = self._model
        configured_normalized = self._normalize_model_name(configured)
        listed_models = self.list_models()

        generation_models: List[str] = []
        for model in listed_models:
            name = str(getattr(model, "name", "") or "")
            actions = getattr(model, "supported_actions", None) or []
            if name and "generateContent" in actions:
                generation_models.append(name)

        resolved = configured
        if generation_models:
            configured_candidates = {configured, configured_normalized, f"models/{configured_normalized}"}
            configured_normalized_candidates = {
                self._normalize_model_name(candidate) for candidate in configured_candidates
            }
            exact_match = next(
                (
                    name
                    for name in generation_models
                    if self._normalize_model_name(name) in configured_normalized_candidates
                ),
                None,
            )
            if exact_match:
                resolved = exact_match
            else:
                preferred_order = [
                    "models/gemini-2.5-flash",
                    "models/gemini-2.0-flash",
                    "models/gemini-1.5-flash-latest",
                    "models/gemini-1.5-flash",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-flash",
                ]
                preferred_match = next(
                    (
                        name
                        for preferred in preferred_order
                        for name in generation_models
                        if self._normalize_model_name(name) == self._normalize_model_name(preferred)
                    ),
                    None,
                )
                resolved = preferred_match or generation_models[0]

        self._resolved_model = resolved
        if self._normalize_model_name(resolved) != configured_normalized:
            LOGGER.info("Using Gemini generation model '%s' instead of '%s'.", resolved, configured)
        return resolved

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        except Exception as exc:
            raise RuntimeError(f"API Error: {exc}") from exc
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
        "Do NOT copy the raw context format (no repeating 'Source:', 'Question:', 'Answer:' blocks).",
        "Follow this structure:",
        "1. Short definition",
        "2. Simple explanation",
        "3. Example (if applicable)",
        "4. Sources: [Label1], [Label2]",
        "Keep answer concise (max 120 words).",
        "Use friendly, clear, educational language and avoid unnecessary jargon.",
        "Prefer Glaive context over StackOverflow when both are present.",
    ]
    compact_context = _compress_context_for_prompt(query, context)
    return (
        "Instructions:\n"
        + "\n".join(instructions)
        + "\n\nContext:\n"
        + compact_context
        + "\n\nUser query:\n"
        + query
    )


@dataclass(frozen=True)
class GenerationMeta:
    mode: ResponseMode
    reason: str


def generate_response_with_meta(query: str, context: str) -> tuple[str, GenerationMeta]:
    normalized_query = _normalize_text(query)
    if not normalized_query or _context_is_empty(context):
        return FALLBACK_MESSAGE, GenerationMeta(mode="offline_fallback", reason="empty_query_or_context")

    policy = _response_policy()
    prompt = _build_prompt(normalized_query, context)
    response = _get_client().generate(prompt)
    validated = _validate_generated_response(normalized_query, context, response)
    if validated:
        return validated, GenerationMeta(mode="online", reason="ok")
    LOGGER.warning("Discarded model response due to validation failure; using offline fallback.")
    if policy == "hard_fail":
        raise RuntimeError("Model response validation failed.")
    return _offline_response(normalized_query, context), GenerationMeta(
        mode="offline_fallback",
        reason="invalid_or_context_dumped_output",
    )


def generate_response(query: str, context: str) -> str:
    """Generate a response using Gemini grounded in retrieved context."""
    response, _meta = generate_response_with_meta(query, context)
    return response
