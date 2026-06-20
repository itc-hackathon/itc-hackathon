"""MemoryArena: drive NapLoRA + baselines together for a LIVE side-by-side UI.

Eric's web app can call this to stream a trajectory into all three memory
strategies at once and render, per step:
  * context-window fill (NapLoRA stays flat as it evicts to weights; vanilla caps
    at the budget; markdown grows) -> the "context window zooming out" panel
  * evicted-to-weights tokens + number of nap adapters (NapLoRA)
And per query:
  * each method's answer, hit/miss, and query-time prompt tokens (the KV-cache /
    inference-cost proxy) for the side-by-side trace.

Everything is plain dicts (JSON-serializable) so it can go straight over a socket.
"""

from __future__ import annotations

from typing import Iterator

from ..core.model import D2LModel
from .baselines import MarkdownMemory, VanillaContextMemory
from .nap_memory import NapLoRAMemory


class MemoryArena:
    def __init__(self, model: D2LModel, nap_k: int = 4, baseline_budget: int = 256,
                 md_k: int | None = None):
        # md_k decouples the markdown baseline's summarization interval from the
        # NapLoRA nap interval: a larger md_k keeps long runs tractable (fewer 2B
        # summarization passes) without changing how NapLoRA naps.
        self.model = model
        self.napora = NapLoRAMemory(model, nap_every_k=nap_k, keep_recent_turns=0)
        self.vanilla = VanillaContextMemory(model, context_budget_tokens=baseline_budget)
        self.markdown = MarkdownMemory(model, summarize_every_k=md_k or nap_k)
        self.baseline_budget = baseline_budget
        self.step = 0

    def observe(self, role: str, text: str) -> dict:
        """Stream one turn into all three; return a per-step frame for the UI."""
        self.step += 1
        st = self.napora.observe(role, text)
        self.vanilla.observe(role, text)
        self.markdown.observe(role, text)
        window, used = self.vanilla._window()
        return {
            "type": "turn",
            "step": self.step,
            "role": role,
            "text": text,
            "napora": {
                "context_tokens": st.context_tokens,
                "evicted_tokens": st.evicted_tokens,
                "segments": st.num_segments,
                "adapter_rank": st.adapter_rank,
                "event": st.last_event,
            },
            "vanilla": {"context_tokens": used, "budget": self.baseline_budget,
                        "turns_dropped": len(self.vanilla.history) - len(window)},
            "markdown": {"context_tokens": self.model.count_tokens(self.markdown._md()),
                         "notes_lines": len(self.markdown.notes)},
        }

    def ask(self, query: str, needle: str | None = None, max_new_tokens: int = 96) -> dict:
        """Ask all three the same question; return a side-by-side answer frame."""
        self.napora.flush()  # make sure all streamed turns are in weights
        frames = {}
        for name, mem in (("napora", self.napora), ("vanilla", self.vanilla),
                          ("markdown", self.markdown)):
            r = mem.ask(query, max_new_tokens=max_new_tokens) if name != "napora" \
                else mem.ask(query, top_k=1, max_new_tokens=max_new_tokens)
            hit = needle is not None and needle.lower() in r["answer"].lower()
            frames[name] = {"answer": r["answer"], "prompt_tokens": r["prompt_tokens"],
                            "hit": hit}
        return {"type": "query", "query": query, "needle": needle, "methods": frames}

    def run(self, turns, probes) -> Iterator[dict]:
        """Convenience generator: yield a frame per turn, then per probe."""
        for role, text in turns:
            yield self.observe(role, text)
        for q, needle in probes:
            yield self.ask(q, needle=needle)
