"""Shared recall scoring for the memory demos and evals.

These scripts all check the same thing — did the model's output contain the
expected needle? — but each used to define its own one-line ``hit`` (with
inconsistent argument order). Use this one instead.
"""

from __future__ import annotations


def hit(output: str, expected: str) -> bool:
    """True if ``expected`` appears (case-insensitive) in the model ``output``."""
    return expected.lower() in output.lower()
