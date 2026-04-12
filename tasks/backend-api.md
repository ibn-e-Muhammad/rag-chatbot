TASK: Backend API with Hybrid RAG

ENDPOINT:
POST /chat

FLOW:
1. Receive query
2. Run hybrid retrieval:
   - Tree-based
   - Vector-based
3. Merge context
4. Validate context
5. Send to Gemini
6. Return response

CONSTRAINTS:
- Reject weak context
- Apply strict rules
- Maintain fast response time