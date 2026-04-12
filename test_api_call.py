#!/usr/bin/env python3
"""Test if Gemini API is actually being called."""

import os
import sys

# Set API key for this test
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

sys.path.insert(0, ".")

from backend.llm.gemini import generate_response, _context_is_empty, _offline_response

# Test 1: Empty context (should NOT call API)
print("=" * 50)
print("TEST 1: Empty Context (should short-circuit)")
print("=" * 50)
response1 = generate_response("How do I reverse a string?", "")
print(f"Response: {response1}")
print(f"Offline only: {_context_is_empty('')}")
print()

# Test 2: Good context (should try API, then fallback if needed)
good_context = """Source: [Glaive]
Question: What is a list comprehension in Python?
Answer: A list comprehension offers a shorter syntax when you want to create a new list based on the values of an existing list. newlist = [x for x in fruits if "a" in x]"""

print("=" * 50)
print("TEST 2: Valid Context (should use API if available)")
print("=" * 50)
print(f"Context empty: {_context_is_empty(good_context)}")
response2 = generate_response("Explain list comprehensions to me.", good_context)
print(f"Response:\n{response2}")
print()

# Test 3: Check if offline fallback is being used
print("=" * 50)
print("TEST 3: Offline Fallback Response")
print("=" * 50)
offline_resp = _offline_response("Explain list comprehensions.", good_context)
print(f"Offline response:\n{offline_resp}")
