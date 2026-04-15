import logging
import re
import time
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm.gemini import generate_direct_response, generate_response_with_meta
from rag.retrieve import retrieve

try:
    from google.genai.errors import ClientError as GeminiClientError  # type: ignore
except ImportError:  # pragma: no cover - optional import guard
    GeminiClientError = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SOURCE_RE = re.compile(r"Source:\s*\[([^\]]+)\]")
RATE_LIMIT_MESSAGE = (
    "API rate limit exceeded. I am receiving too many requests right now. "
    "Please wait about 10 seconds and try asking again."
)
SERVICE_UNAVAILABLE_MESSAGE = "The AI service is currently unavailable. Please try again later."


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    direct_answer: str | None = None
    mode: str | None = None
    reason: str | None = None


def _extract_sources(context: str) -> List[str]:
    seen = set()
    sources: List[str] = []
    for match in _SOURCE_RE.findall(context or ""):
        label = match.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        sources.append(label)
    return sources


def _is_gemini_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).upper()
    if GeminiClientError and isinstance(exc, GeminiClientError):
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "TOO MANY REQUESTS" in text


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info("Received chat query: %s", request.query)
    start_time = time.perf_counter()
    try:
        context = retrieve(request.query)
        sources = _extract_sources(context)
        try:
            answer, meta = generate_response_with_meta(request.query, context)
        except Exception as exc:
            if _is_gemini_rate_limit_error(exc):
                logger.exception("Gemini API rate-limited during response generation")
                return ChatResponse(
                    answer=RATE_LIMIT_MESSAGE,
                    sources=sources,
                    mode="hard_fail",
                    reason="gemini_rate_limit",
                )
            logger.exception("Gemini API unavailable during response generation")
            return ChatResponse(
                answer=SERVICE_UNAVAILABLE_MESSAGE,
                sources=sources,
                mode="hard_fail",
                reason="gemini_service_unavailable",
            )

        # --- Direct (non-RAG) answer (non-blocking) ---
        direct_answer = None
        try:
            direct_text = generate_direct_response(request.query)
            if direct_text:
                direct_answer = direct_text
        except Exception:
            logger.warning("Direct API answer generation failed; skipping.", exc_info=True)

        duration = time.perf_counter() - start_time
        logger.info("Generated response in %.2fs (mode=%s reason=%s)", duration, meta.mode, meta.reason)
        return ChatResponse(
            answer=answer,
            sources=sources,
            direct_answer=direct_answer,
            mode=meta.mode,
            reason=meta.reason,
        )
    except Exception as exc:
        logger.exception("Failed to generate response")
        return ChatResponse(
            answer=SERVICE_UNAVAILABLE_MESSAGE,
            sources=[],
            mode="hard_fail",
            reason=f"backend_error: {type(exc).__name__}",
        )
