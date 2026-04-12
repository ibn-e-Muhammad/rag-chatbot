TASK: Implement Hybrid RAG System

PART 1: DATA INGESTION
- Load both datasets
- Clean data properly
- Normalize text

PART 2: TREE STRUCTURE
- Extract topics from questions
- Build hierarchical JSON tree
- Store locally

PART 3: VECTOR DATABASE
- Generate embeddings
- Store in ChromaDB
- Keep datasets separated

PART 4: RETRIEVAL SYSTEM

1. Tree Retrieval:
   - Match query to topic
   - Traverse tree
   - Retrieve relevant nodes

2. Vector Retrieval:
   - Similarity search
   - Top-k results

3. Context Merge:
   - Combine both sources
   - Add labels

OUTPUT:
- retrieve(query) function returning structured context