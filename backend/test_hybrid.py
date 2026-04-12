import sys
from pathlib import Path

import pytest

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from rag.retrieve import retrieve

QUERIES = [
    "How do I use for loops in Python?",
    "FastAPI middleware for custom headers",
    "How do I bake a chocolate cake?",
]

@pytest.mark.parametrize("query", QUERIES)
def test_query(query):
    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print(f"{'='*50}")
    context = retrieve(query)
    print(context if context else "No context retrieved.")

if __name__ == "__main__":
    for query in QUERIES:
        test_query(query)
