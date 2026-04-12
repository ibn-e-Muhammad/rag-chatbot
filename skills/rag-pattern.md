HYBRID RAG PATTERN:

Step 1: Receive user query

Step 2: Tree-based retrieval (PRIMARY)
- Identify topic from query
- Traverse hierarchical structure:
  Topic → Subtopic → Q&A
- Retrieve relevant nodes

Step 3: Vector retrieval (FALLBACK)
- Query vector database (Chroma)
- Retrieve top-k (k=2–3)

Step 4: Merge context
- Combine tree results + vector results
- Label sources:
  [Glaive]
  [StackOverflow]

Step 5: Build final context
- Remove duplicates
- Limit size

Step 6: Generate answer
- Strict grounding
- No external knowledge

Step 7: Validate
- If weak context → reject