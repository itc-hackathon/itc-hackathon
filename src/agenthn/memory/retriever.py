"""Dependency-free TF-IDF retriever over napped memory segments.

The base 2B LLM makes a poor sentence encoder (mean/last-token pooling routes
only 4-5/6 of NIAH queries; cosines all cluster ~0.8). A plain TF-IDF cosine
routes 6/6 with wide margins on the same probes, because NIAH queries reuse the
stored segment's keywords. It is also fast and fully inspectable, which matters
for a live demo. Swap in / add a dense encoder later for paraphrase robustness;
the LoRAStore API does not change.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9_]+")


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class TfidfRetriever:
    """Incremental TF-IDF index: add(text) on each nap, score(query) at recall."""

    def __init__(self) -> None:
        self._docs: list[list[str]] = []
        self._df: Counter[str] = Counter()

    def add(self, text: str) -> int:
        """Index a new segment; returns its integer id (insertion order)."""
        toks = _tok(text)
        self._docs.append(toks)
        self._df.update(set(toks))
        return len(self._docs) - 1

    def _idf(self, word: str) -> float:
        n = len(self._docs)
        return math.log((n + 1) / (self._df.get(word, 0) + 1)) + 1

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {w: tf[w] * self._idf(w) for w in tf}

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(w, 0.0) for w, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb + 1e-9)

    def scores(self, query: str) -> list[float]:
        """Cosine of the query against every indexed segment (by id order)."""
        q = self._vec(_tok(query))
        return [self._cos(q, self._vec(d)) for d in self._docs]

    def topk(self, query: str, k: int = 1) -> list[tuple[int, float]]:
        """Top-k (segment_id, score), highest first."""
        scored = list(enumerate(self.scores(query)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
