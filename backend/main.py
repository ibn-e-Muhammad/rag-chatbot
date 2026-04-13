import logging
import re
import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm.gemini import generate_response_with_meta
from rag.retrieve import retrieve

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


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info("Received chat query: %s", request.query)
    start_time = time.perf_counter()
    try:
        context = retrieve(request.query)
        answer, meta = generate_response_with_meta(request.query, context)
        sources = _extract_sources(context)
        duration = time.perf_counter() - start_time
        logger.info("Generated response in %.2fs (mode=%s reason=%s)", duration, meta.mode, meta.reason)
        return ChatResponse(answer=answer, sources=sources, mode=meta.mode, reason=meta.reason)
    except Exception as exc:
        logger.exception("Failed to generate response")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
