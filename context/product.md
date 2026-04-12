PROJECT TYPE:
Domain-restricted AI tutor chatbot

DOMAIN:
- Python programming
- AI / Machine Learning concepts

DATA SOURCES:
1. Glaive Python QA dataset (primary, clean)
2. StackOverflow Python dataset (secondary, noisy but rich)

SYSTEM DESIGN:
- Hybrid RAG system:
  1. Tree-based structured retrieval (primary)
  2. Vector similarity search (fallback)

GOAL:
- Provide accurate, context-grounded answers
- Prefer structured knowledge over raw similarity

BEHAVIOR:
- The chatbot MUST NOT act as a general assistant
- It must ONLY answer from internal knowledge sources