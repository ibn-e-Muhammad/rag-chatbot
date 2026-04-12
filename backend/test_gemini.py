import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from llm.gemini import generate_response

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