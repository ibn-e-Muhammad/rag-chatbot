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


def test_source_labels_present_when_context_exists():
    context = retrieve("How do I use for loops in Python?")
    if not context:
        pytest.skip("No context returned")
    assert "Source: [Glaive]" in context or "Source: [StackOverflow]" in context


def test_no_exact_duplicate_blocks():
    context = retrieve("How do I use for loops in Python?")
    if not context:
        pytest.skip("No context returned")
    blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
    assert len(blocks) == len(set(blocks))


def test_context_formatting_pattern():
    context = retrieve("FastAPI middleware for custom headers")
    if not context:
        pytest.skip("No context returned")
    for block in [b for b in context.split("\n\n") if b.strip()]:
        assert "Source: [" in block
        assert "Question:" in block
        assert "Answer:" in block

if __name__ == "__main__":
    for query in QUERIES:
        test_query(query)
