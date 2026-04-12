/fleet

You are orchestrating a structured multi-agent development system.

PROJECT CONTEXT:
- This is a domain-restricted RAG chatbot for Python and AI concepts.
- The chatbot MUST answer ONLY using retrieved dataset context.
- It must NEVER use general knowledge or hallucinate.

LOAD AND FOLLOW ALL PROJECT FILES:

1. Context:
   - /context/product.md

2. Rules:
   - /rules/llm-rules.md

3. Skills:
   - /skills/rag-pattern.md
   - /skills/python-explanation-style.md

4. Agents:
   - /agents/backend-agent.md
   - /agents/frontend-agent.md
   - /agents/reviewer-agent.md

5. Tasks:
   - /tasks/project-overview.md
   - /tasks/rag-implementation.md
   - /tasks/backend-api.md
   - /tasks/frontend-ui.md


EXECUTION STRATEGY:

Step 1: Understand the full system before coding anything.

Step 2: Begin with RAG implementation:
- Load dataset
- Convert to documents
- Chunk text
- Generate embeddings
- Store in ChromaDB
- Implement retrieval function

Step 3: Build backend API:
- FastAPI server
- POST /chat endpoint
- Integrate retrieval + Gemini
- Enforce strict rules before responding

Step 4: Build frontend:
- React + Tailwind UI
- Chat interface
- API integration
- Loading + error states

Step 5: Review:
- Ensure no hallucination paths exist
- Ensure strict rule enforcement
- Ensure modular clean code


STRICT CONSTRAINTS:

- NEVER generate features not defined in tasks
- NEVER assume missing requirements
- NEVER use external knowledge in chatbot responses
- ALWAYS return "I don't have enough information to answer that." if context is insufficient
- KEEP implementation simple and modular


OUTPUT FORMAT:

- Generate code file-by-file
- Follow the defined project structure
- Keep files clean and focused
- Do not dump everything in one file


BEGIN execution now.