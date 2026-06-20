"""Text-memory baselines to compare against NapLoRA weight memory.

All expose the same surface as NapLoRAMemory (observe / ask -> dict) so the demo
and ablation harnesses can drive them identically.

  VanillaContextMemory  — keep the raw transcript, but the prompt can only hold
                          the most-recent turns that fit a fixed context budget.
                          Facts that scrolled out of the window are simply gone.
  MarkdownMemory        — the agent maintains a running markdown notes file: every
                          K turns it summarizes the evicted block into bullets (one
                          2B forward pass) and keeps the whole .md in the prompt.
                          Lossy summaries can drop facts, and the prompt grows with
                          history (the "wastes the context window" failure).
  TextRAGMemory         — the *ablation* of NapLoRA's weight-injection step. SAME
                          segmentation (nap every K) and SAME TF-IDF retriever, but
                          the retrieved segment is delivered as TEXT in the prompt
                          instead of as a LoRA adapter. Isolates the one thing the
                          hypernetwork adds: the retrieved memory's tokens never
                          enter the context (O(1) query context vs O(chunk) for RAG).
"""

from __future__ import annotations

from ..core.model import D2LModel
from .retriever import TfidfRetriever


class VanillaContextMemory:
    """Full transcript, but only the most-recent turns that fit the budget."""

    def __init__(self, model: D2LModel, context_budget_tokens: int = 256):
        self.model = model
        self.budget = context_budget_tokens
        self.history: list[tuple[str, str]] = []

    def observe(self, role: str, text: str) -> None:
        self.history.append((role, text))

    def _window(self) -> tuple[list[tuple[str, str]], int]:
        """Most-recent turns whose cumulative tokens fit the budget."""
        kept: list[tuple[str, str]] = []
        used = 0
        for role, text in reversed(self.history):
            t = self.model.count_tokens(text) + 2
            if used + t > self.budget:
                break
            kept.append((role, text))
            used += t
        kept.reverse()
        return kept, used

    def ask(self, query: str, max_new_tokens: int = 40) -> dict:
        window, used = self._window()
        transcript = "\n".join(f"{r}: {t}" for r, t in window)
        user = f"Conversation so far:\n{transcript}\n\nQuestion: {query}"
        messages = [{"role": "user", "content": user}]
        answer = self.model.respond(messages, max_new_tokens=max_new_tokens)
        return {
            "query": query,
            "answer": answer,
            "prompt_tokens": self.model.count_tokens(user),
            "turns_in_window": len(window),
            "turns_dropped": len(self.history) - len(window),
        }


class MarkdownMemory:
    """Running markdown notes maintained by the model; kept fully in the prompt."""

    _SUMMARIZE = (
        "You maintain a project memory. Extract only the durable facts from these "
        "conversation turns as terse markdown bullets (one fact per line, keep names, "
        "codes, numbers, dates verbatim). If there are no durable facts, output NONE.\n\n"
        "Turns:\n{block}\n\nFacts:"
    )

    def __init__(self, model: D2LModel, summarize_every_k: int = 4):
        self.model = model
        self.k = summarize_every_k
        self.pending: list[tuple[str, str]] = []
        self.notes: list[str] = []

    def observe(self, role: str, text: str) -> None:
        self.pending.append((role, text))
        if len(self.pending) >= self.k:
            self._summarize()

    def _summarize(self) -> None:
        if not self.pending:
            return
        block = "\n".join(f"{r}: {t}" for r, t in self.pending)
        self.pending = []
        out = self.model.respond(
            [{"role": "user", "content": self._SUMMARIZE.format(block=block)}],
            max_new_tokens=120,
        )
        for line in out.splitlines():
            line = line.strip().lstrip("-*").strip()
            if line and line.upper() != "NONE":
                self.notes.append(line)

    def _md(self) -> str:
        return "# Project memory\n" + "\n".join(f"- {n}" for n in self.notes)

    def ask(self, query: str, max_new_tokens: int = 40) -> dict:
        self._summarize()  # flush any pending turns into notes first
        md = self._md()
        user = f"{md}\n\nUsing the project memory above, answer: {query}"
        messages = [{"role": "user", "content": user}]
        answer = self.model.respond(messages, max_new_tokens=max_new_tokens)
        return {
            "query": query,
            "answer": answer,
            "prompt_tokens": self.model.count_tokens(user),
            "notes_lines": len(self.notes),
            "notes_tokens": self.model.count_tokens(md),
        }


class TextRAGMemory:
    """RAG ablation: NapLoRA's pipeline with the weight-injection step removed.

    Identical segmentation (nap every K turns) and identical TF-IDF retriever as
    NapLoRAMemory — the ONLY change is that the retrieved segment is pasted into
    the prompt as text rather than loaded as a LoRA adapter. Comparing this to
    NapLoRA isolates the contribution of the hypernetwork / weight compaction:
    same retrieval, same recall, but here the chunk's tokens re-enter context.
    """

    def __init__(self, model: D2LModel, nap_every_k: int = 4, top_k: int = 1):
        self.model = model
        self.k = nap_every_k
        self.top_k = top_k
        self.pending: list[tuple[str, str]] = []
        self.segments: list[str] = []        # stored as TEXT (the only difference)
        self.retriever = TfidfRetriever()

    def observe(self, role: str, text: str) -> None:
        self.pending.append((role, text))
        while len(self.pending) >= self.k:
            self._nap()

    def _nap(self) -> None:
        chunk, self.pending = self.pending[: self.k], self.pending[self.k :]
        doc = "\n".join(f"{r}: {t}" for r, t in chunk)
        self.retriever.add(doc)
        self.segments.append(doc)

    def flush(self) -> None:
        while self.pending:
            n = min(self.k, len(self.pending))
            chunk, self.pending = self.pending[:n], self.pending[n:]
            doc = "\n".join(f"{r}: {t}" for r, t in chunk)
            self.retriever.add(doc)
            self.segments.append(doc)

    def ask(self, query: str, max_new_tokens: int = 40) -> dict:
        if self.segments:
            hits = self.retriever.topk(query, k=min(self.top_k, len(self.segments)))
            retrieved = [self.segments[i] for i, _ in hits]
        else:
            hits, retrieved = [], []
        ctx = "\n".join(retrieved)
        user = f"Relevant memory:\n{ctx}\n\nQuestion: {query}" if ctx else query
        messages = [{"role": "user", "content": user}]
        answer = self.model.respond(messages, max_new_tokens=max_new_tokens)
        return {
            "query": query,
            "answer": answer,
            "prompt_tokens": self.model.count_tokens(user),
            "retrieved": [i for i, _ in hits],
            "retrieved_tokens": self.model.count_tokens(ctx),
        }
