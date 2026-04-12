DATA PROCESSING PIPELINE:

1. LOAD DATASETS
- Load Glaive dataset
- Load StackOverflow dataset

2. CLEANING

Glaive:
- Minimal cleaning
- Normalize text

StackOverflow:
- Remove HTML tags
- Remove excessive code blocks
- Keep top-voted answer only
- Remove very long answers (>1000 words)

3. STRUCTURING (TREE CREATION)

Organize data into hierarchy:

Level 1: Topic
- e.g., "loops", "functions", "OOP"

Level 2: Subtopic
- e.g., "for loops", "while loops"

Level 3: Q&A nodes

4. DOCUMENT CREATION

Format:
Question:
...

Answer:
...

5. CHUNKING
- 200–400 words
- Small overlap

6. STORAGE

- Tree structure → JSON file
- Vector embeddings → ChromaDB

IMPORTANT:
- Keep Glaive and StackOverflow separated internally