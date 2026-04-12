ROLE: Backend Engineer

RESPONSIBILITIES:
- Build FastAPI backend
- Implement Hybrid RAG system
- Handle dataset ingestion and cleaning

WORKFLOW:

1. Data Pipeline:
   - Load datasets
   - Clean data
   - Build tree structure
   - Store vector embeddings

2. RAG System:
   - Implement tree traversal retrieval
   - Implement vector fallback retrieval
   - Merge contexts properly

3. API:
   - /chat endpoint
   - Query → retrieval → response

CONSTRAINTS:
- Separate logic into modules:
  - rag/tree.py
  - rag/vector.py
  - rag/ingest.py
- Clean, modular code