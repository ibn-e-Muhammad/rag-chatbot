import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from rag.retrieve import retrieve

def test_query(query):
    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print(f"{'='*50}")
    context = retrieve(query)
    print(context if context else "No context retrieved.")

if __name__ == "__main__":
    # Test 1: Specific topic (Should hit Tree first)
    test_query("How do I use for loops in Python?")

    # Test 2: Niche/Complex query (Should trigger Vector fallback)
    test_query("FastAPI middleware for custom headers")

    # Test 3: Out-of-domain (Should return minimal/no context)
    test_query("How do I bake a chocolate cake?")