import os
import sys
from pathlib import Path

import pytest

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from llm.gemini import FALLBACK_MESSAGE, generate_response, generate_response_with_meta

def run_llm_test(test_name: str, query: str, context: str):
    print(f"\n{'='*50}")
    print(f"🚀 {test_name}")
    print(f"QUERY: {query}")
    print(f"CONTEXT SIZE: {len(context)} characters")
    print(f"{'-'*50}")
    
    try:
        response = generate_response(query, context)
        print(f"RESPONSE:\n{response}")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_short_circuit_empty_context():
    response, meta = generate_response_with_meta("How do I reverse a string in Python?", "")
    assert response == FALLBACK_MESSAGE
    assert meta.mode == "offline_fallback"


def test_structured_response_from_relevant_context():
    context = (
        "Source: [Glaive]\n"
        "Question: What is a list comprehension in Python?\n"
        "Answer: A list comprehension offers a shorter syntax when you want to create a new list based on the values of an existing list. "
        "Example: newlist = [x for x in fruits if 'a' in x]"
    )
    response = generate_response("Explain list comprehensions to me.", context)
    assert "Short definition:" in response
    assert "Simple explanation:" in response
    assert "Example:" in response
    assert "Source:" not in response or "Sources:" in response


def test_hallucination_trap_refusal():
    context = (
        "Source: [StackOverflow]\n"
        "Question: How to print to console?\n"
        "Answer: Use the print() function."
    )
    response = generate_response("Who won the World Cup in 2022?", context)
    assert response == FALLBACK_MESSAGE


@pytest.mark.parametrize(
    "query,context",
    [
        ("How do I reverse a string in Python?", ""),
        ("Who won the World Cup in 2022?", "Source: [StackOverflow]\nQuestion: How to print to console?\nAnswer: Use the print() function."),
    ],
)
def test_no_context_dump_patterns(query: str, context: str):
    response = generate_response(query, context)
    assert response.count("Source:") <= 1
    assert response.count("Question:") <= 1
    assert response.count("Answer:") <= 1

if __name__ == "__main__":
    # Ensure your API key is loaded in your environment before running this!
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ WARNING: API key not found in environment variables.")

    # ---------------------------------------------------------
    # TEST 1: The Short-Circuit (Empty Context)
    # ---------------------------------------------------------
    # This should NOT call the API. It should instantly return your fallback string.
    run_llm_test(
        test_name="TEST 1: The Short-Circuit",
        query="How do I reverse a string in Python?",
        context=""
    )

    # ---------------------------------------------------------
    # TEST 2: The Perfect Scenario
    # ---------------------------------------------------------
    # This should follow your python-explanation-style.md format.
    good_context = """
    Source: [Glaive]
    Question: What is a list comprehension in Python?
    Answer: A list comprehension offers a shorter syntax when you want to create a new list based on the values of an existing list. Example: newlist = [x for x in fruits if "a" in x]
    """
    run_llm_test(
        test_name="TEST 2: Standard Explanation",
        query="Explain list comprehensions to me.",
        context=good_context.strip()
    )

    # ---------------------------------------------------------
    # TEST 3: The Hallucination Trap (Irrelevant Query)
    # ---------------------------------------------------------
    # The LLM knows the answer from its pre-training, but the context doesn't have it. 
    # It MUST refuse to answer.
    trap_context = """
    Source: [StackOverflow]
    Question: How to print to console?
    Answer: Use the print() function.
    """
    run_llm_test(
        test_name="TEST 3: Hallucination Trap",
        query="Who won the World Cup in 2022?",
        context=trap_context.strip()
    )