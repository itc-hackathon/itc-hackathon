"""Text-memory baselines to compare against NapLoRA weight memory.

Both expose the same surface as NapLoRAMemory (observe / ask -> dict) so the demo
harness can drive all three identically.

  VanillaContextMemory  — keep the raw transcript, but the prompt can only hold
                          the most-recent turns that fit a fixed context budget.
                          Facts that scrolled out of the window are simply gone.
  MarkdownMemory        — the agent maintains a running markdown notes file: every
                          K turns it summarizes the evicted block into bullets (one
                          2B forward pass) and keeps the whole .md in the prompt.
                          Lossy summaries can drop facts, and the prompt grows with
                          history (the "wastes the context window" failure).
"""

from __future__ import annotations

from ..core.model import D2LModel


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
