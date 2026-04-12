"""Vector-based retrieval fallback using Gemini embeddings + ChromaDB."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_QUESTION_ANSWER_RE = re.compile(r"Question:\s*(.*?)\s*Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)

GLAIVE_COLLECTION = "glaive_collection"
STACKOVERFLOW_COLLECTION = "stackoverflow_collection"

DEFAULT_GLAIVE_TOP_K = 2
DEFAULT_STACKOVERFLOW_TOP_K = 3

DEFAULT_CHUNK_TARGET_WORDS = 300
DEFAULT_CHUNK_MIN_WORDS = 200
DEFAULT_CHUNK_MAX_WORDS = 400
DEFAULT_CHUNK_OVERLAP_WORDS = 40

DEFAULT_BATCH_SIZE = 48


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _data_path(filename: str) -> Path:
    return _repo_root() / "data" / "processed" / filename


def _vectorstore_path() -> Path:
    return _repo_root() / "vectorstore" / "chroma"


try:
    import orjson  # type: ignore

    _JSON_DECODE_ERROR = orjson.JSONDecodeError

    def _loads_json(line: bytes) -> Dict[str, object]:
        return orjson.loads(line)

except ImportError:
    _JSON_DECODE_ERROR = json.JSONDecodeError

    def _loads_json(line: bytes) -> Dict[str, object]:
        return json.loads(line.decode("utf-8"))


def _read_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    if not path.exists():
        LOGGER.warning("JSONL path does not exist: %s", path)
        return
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield _loads_json(line)
            except _JSON_DECODE_ERROR as exc:
                LOGGER.warning("Skipping invalid JSON at %s:%s (%s)", path, line_number, exc)


def _split_document(document: str) -> Tuple[str, str]:
    if not document:
        return "", ""
    match = _QUESTION_ANSWER_RE.search(document)
    if not match:
        return _normalize_text(document), ""
    return _normalize_text(match.group(1)), _normalize_text(match.group(2))


def _extract_question_answer(record: Mapping[str, object]) -> Tuple[str, str]:
    question = str(record.get("question") or "")
    answer = str(record.get("answer") or "")

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

    question = _normalize_text(question)
    answer = _normalize_text(answer)
    return question, answer


def _chunk_answer(question: str, answer: str) -> Iterable[str]:
    question_words = question.split()
    answer_words = answer.split()
    if not question_words and not answer_words:
        return []

    target_answer_words = max(DEFAULT_CHUNK_TARGET_WORDS - len(question_words), 120)
    target_answer_words = max(target_answer_words, DEFAULT_CHUNK_MIN_WORDS - len(question_words))
    target_answer_words = min(target_answer_words, max(DEFAULT_CHUNK_MAX_WORDS - len(question_words), 120))
    step = max(target_answer_words - DEFAULT_CHUNK_OVERLAP_WORDS, 1)

    if not answer_words:
        return [f"Question: {question}\nAnswer: "]

    chunks: List[str] = []
    for start in range(0, len(answer_words), step):
        chunk_words = answer_words[start : start + target_answer_words]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words)
        chunks.append(f"Question: {question}\nAnswer: {chunk_text}")
        if start + target_answer_words >= len(answer_words):
            break
    return chunks


def _build_chunk_id(source: str, record: Mapping[str, object], chunk_index: int, question: str, answer: str) -> str:
    if source == "stackoverflow":
        question_id = record.get("question_id") or record.get("questionId") or record.get("questionID")
        answer_id = record.get("answer_id") or record.get("answerId") or record.get("answerID")
        if question_id and answer_id:
            return f"{source}-{question_id}-{answer_id}-{chunk_index}"
    payload = f"{source}-{question}-{answer}-{chunk_index}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:24]
    return f"{source}-{digest}-{chunk_index}"


def _resolve_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    return api_key


class _GeminiEmbedder:
    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key
        self._client = self._build_client()
        self._resolved_model: str | None = None

    def _build_client(self):
        try:
            from google import genai  # type: ignore

            return genai.Client(api_key=self._api_key)
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

    def _resolve_model_name(self) -> str:
        if self._resolved_model:
            return self._resolved_model

        configured = self._model
        configured_normalized = self._normalize_model_name(configured)
        listed_models = self.list_models()

        embed_models: List[str] = []
        for model in listed_models:
            name = str(getattr(model, "name", "") or "")
            actions = getattr(model, "supported_actions", None) or []
            if name and "embedContent" in actions:
                embed_models.append(name)

        resolved = configured
        if embed_models:
            configured_candidates = {configured, configured_normalized, f"models/{configured_normalized}"}
            configured_normalized_candidates = {
                self._normalize_model_name(candidate) for candidate in configured_candidates
            }
            exact_match = next(
                (
                    name
                    for name in embed_models
                    if self._normalize_model_name(name) in configured_normalized_candidates
                ),
                None,
            )
            if exact_match:
                resolved = exact_match
            else:
                preferred_order = ["models/text-embedding-004", "models/embedding-001", "text-embedding-004", "embedding-001"]
                preferred_match = next(
                    (
                        name
                        for preferred in preferred_order
                        for name in embed_models
                        if self._normalize_model_name(name) == self._normalize_model_name(preferred)
                    ),
                    None,
                )
                resolved = preferred_match or embed_models[0]

        self._resolved_model = resolved
        if self._normalize_model_name(resolved) != configured_normalized:
            LOGGER.info("Using Gemini embedding model '%s' instead of '%s'.", resolved, configured)
        return resolved

    def _to_vector(self, value: object) -> List[float]:
        if isinstance(value, dict):
            if "values" in value and isinstance(value["values"], (list, tuple)):
                return [float(item) for item in value["values"]]
            if "embedding" in value:
                return self._to_vector(value["embedding"])
        if hasattr(value, "values"):
            values = getattr(value, "values")
            if isinstance(values, (list, tuple)):
                return [float(item) for item in values]
        if hasattr(value, "embedding"):
            return self._to_vector(getattr(value, "embedding"))
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        raise RuntimeError("Unexpected embedding payload.")

    def _extract_embeddings(self, result: object, expected: int) -> List[List[float]]:
        embeddings = None
        if isinstance(result, dict):
            embeddings = result.get("embeddings") or result.get("embedding")
        else:
            embeddings = getattr(result, "embeddings", None) or getattr(result, "embedding", None)

        if embeddings is None:
            embeddings = result

        if isinstance(embeddings, list):
            if not embeddings:
                vectors: List[List[float]] = []
            elif isinstance(embeddings[0], (int, float)):
                vectors = [self._to_vector(embeddings)]
            else:
                vectors = [self._to_vector(item) for item in embeddings]
        else:
            vectors = [self._to_vector(embeddings)]

        if len(vectors) != expected:
            raise RuntimeError(f"Unexpected embedding count returned. expected={expected} got={len(vectors)}")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self._client.models.embed_content(
            model=self._resolve_model_name(),
            contents=list(texts),
        )
        return self._extract_embeddings(result, len(texts))

    def embed_query(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self._resolve_model_name(),
            contents=[text],
        )
        return self._extract_embeddings(result, 1)[0]


@lru_cache(maxsize=1)
def _get_embedder() -> _GeminiEmbedder:
    model = os.getenv("GEMINI_EMBEDDING_MODEL") or "models/text-embedding-004"
    return _GeminiEmbedder(model=model, api_key=_resolve_api_key())


@lru_cache(maxsize=1)
def _get_chroma_client():
    try:
        import chromadb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing chromadb dependency.") from exc
    path = _vectorstore_path()
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


@lru_cache(maxsize=2)
def _get_collection(name: str):
    client = _get_chroma_client()
    return client.get_or_create_collection(name=name)


def _upsert_batch(collection, embedder: _GeminiEmbedder, documents: List[str], ids: List[str], metadatas: List[Dict[str, object]]) -> None:
    embeddings = embedder.embed_documents(documents)
    if hasattr(collection, "upsert"):
        collection.upsert(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)
    else:
        collection.add(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)


def _ingest_source(path: Path, source: str, source_label: str, collection_name: str) -> None:
    collection = _get_collection(collection_name)
    if collection.count() > 0:
        LOGGER.info("Collection %s already populated (%s entries)", collection_name, collection.count())
        return
    if not path.exists():
        LOGGER.warning("Source data not found for %s: %s", source, path)
        return

    embedder = _get_embedder()
    batch_docs: List[str] = []
    batch_ids: List[str] = []
    batch_metadatas: List[Dict[str, object]] = []
    indexed = 0
    skipped = 0

    for record in _read_jsonl(path):
        question, answer = _extract_question_answer(record)
        if not question or not answer:
            skipped += 1
            continue
        chunks = list(_chunk_answer(question, answer))
        for chunk_index, chunk in enumerate(chunks):
            metadata: Dict[str, object] = {
                "source": source,
                "source_label": source_label,
                "question": question,
                "answer": _normalize_text(chunk.split("Answer:", 1)[-1]),
            }
            if source == "stackoverflow":
                for key in ("question_id", "answer_id", "score"):
                    if key in record and record[key] is not None:
                        metadata[key] = record[key]
            doc_id = _build_chunk_id(source, record, chunk_index, question, chunk)
            batch_docs.append(chunk)
            batch_ids.append(doc_id)
            batch_metadatas.append(metadata)
            if len(batch_docs) >= DEFAULT_BATCH_SIZE:
                _upsert_batch(collection, embedder, batch_docs, batch_ids, batch_metadatas)
                indexed += len(batch_docs)
                batch_docs, batch_ids, batch_metadatas = [], [], []

    if batch_docs:
        _upsert_batch(collection, embedder, batch_docs, batch_ids, batch_metadatas)
        indexed += len(batch_docs)

    LOGGER.info(
        "Indexed %s chunks for %s (skipped=%s)",
        indexed,
        source,
        skipped,
    )


def _ensure_collections() -> None:
    _ingest_source(_data_path("glaive_cleaned.jsonl"), "glaive", "Glaive", GLAIVE_COLLECTION)
    _ingest_source(_data_path("stackoverflow_docs.jsonl"), "stackoverflow", "StackOverflow", STACKOVERFLOW_COLLECTION)


def _query_collection(collection_name: str, query_embedding: List[float], top_k: int) -> List[Dict[str, object]]:
    if top_k <= 0:
        return []
    collection = _get_collection(collection_name)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    documents = result.get("documents") or [[]]
    document_rows = documents[0] if documents else []
    distance_rows = distances[0] if distances else []

    results: List[Dict[str, object]] = []
    for idx, metadata in enumerate(metadatas[0]):
        metadata = metadata or {}
        question = str(metadata.get("question") or "")
        doc_text = document_rows[idx] if idx < len(document_rows) else ""
        answer = str(metadata.get("answer") or doc_text or "")
        source = str(metadata.get("source") or "")
        if not question and not answer:
            continue
        item: Dict[str, object] = {
            "question": question,
            "answer": answer,
            "source": source,
        }
        if "score" in metadata:
            item["score"] = metadata["score"]
        if idx < len(distance_rows):
            distance = distance_rows[idx]
            if isinstance(distance, (float, int)):
                item["similarity"] = 1.0 - float(distance)
        for key in ("question_id", "answer_id", "source_label"):
            if key in metadata:
                item[key] = metadata[key]
        results.append(item)
    return results


def retrieve_vector(
    query: str,
    glaive_top_k: int = DEFAULT_GLAIVE_TOP_K,
    stackoverflow_top_k: int = DEFAULT_STACKOVERFLOW_TOP_K,
) -> List[Dict[str, object]]:
    """Retrieve fallback results from Gemini+Chroma vector stores."""
    normalized = _normalize_text(query)
    if not normalized:
        return []

    try:
        _ensure_collections()
        embedder = _get_embedder()
        query_embedding = embedder.embed_query(normalized)
        glaive_results = _query_collection(GLAIVE_COLLECTION, query_embedding, glaive_top_k)
        stack_results = _query_collection(STACKOVERFLOW_COLLECTION, query_embedding, stackoverflow_top_k)
    except Exception as exc:
        LOGGER.error("Vector retrieval failed for query '%s': %s", normalized, exc)
        return []

    seen: set[Tuple[str, str]] = set()
    merged: List[Dict[str, object]] = []
    for item in glaive_results + stack_results:
        key = (str(item.get("question") or ""), str(item.get("answer") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged
