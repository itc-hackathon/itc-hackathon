"""NapLoRAMemory: long-horizon agent memory in weight space.

The agent streams turns into a small rolling context. Every K evictable turns it
"naps": the oldest segment is internalized into a LoRA adapter (one D2L forward
pass), indexed in a TF-IDF store, and evicted from the prompt. At query time it
retrieves the top-k relevant segment adapters, composes them (rank-concatenation),
and answers from a tiny prompt + weights.

Why retrieve instead of summing ALL segments: rank-concatenating every segment
sums their weight deltas, which interfere destructively as memory grows
(measured: 3 similar segments -> 0/6 recall). Each segment adapter ALONE recalls
its facts (6/6), so we route each query to the few segments it needs. This is the
"retrieve top-k nap adapters" step of the MVP and what makes memory scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.model import D2LModel
from .retriever import TfidfRetriever


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class Segment:
    id: int
    text: str
    adapter: object          # {module: {A,B: [1, n_layers, r, dim]}}
    n_turns: int
    tokens: int


@dataclass
class MemoryStats:
    step: int = 0
    context_turns: int = 0
    context_tokens: int = 0          # tokens currently in the prompt window
    evicted_tokens: int = 0          # cumulative tokens moved into weights
    num_segments: int = 0            # napped adapters stored
    adapter_rank: int = 0            # rank of the last composed adapter served
    context_budget: int = 0
    last_event: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class NapLoRAMemory:
    """Streaming agent memory: context buffer + nap/evict + retrieve/compose."""

    def __init__(
        self,
        model: D2LModel,
        nap_every_k: int = 4,
        keep_recent_turns: int = 0,
        context_budget_tokens: int = 8192,
    ):
        self.model = model
        self.nap_every_k = nap_every_k
        self.keep_recent_turns = keep_recent_turns
        self.context_budget = context_budget_tokens

        self.context: list[Turn] = []
        self.segments: list[Segment] = []
        self.retriever = TfidfRetriever()
        self.events: list[dict] = []
        self.rank_per_segment = 0     # base LoRA rank r (filled after first nap)
        self._step = 0

    # --- streaming --------------------------------------------------------
    def observe(self, role: str, text: str) -> MemoryStats:
        """Append a turn; nap if enough turns have accumulated past the recent window."""
        self._step += 1
        self.context.append(Turn(role, text))
        evicted = ""
        while len(self.context) - self.keep_recent_turns >= self.nap_every_k:
            evicted = self._nap()
        return self._stats(evicted or f"observed {role} turn")

    def _nap(self) -> str:
        """Move the oldest nap_every_k evictable turns into one weight segment."""
        n = self.nap_every_k
        chunk, self.context = self.context[:n], self.context[n:]
        doc = self._segment_doc(chunk)
        adapter = self.model.internalize_segment(doc)
        if not self.rank_per_segment:
            a = adapter[next(iter(adapter))]["A"]
            self.rank_per_segment = a.shape[2]            # [1, layers, r, dim]
        sid = self.retriever.add(doc)
        tokens = self.model.count_tokens(doc)
        self.segments.append(
            Segment(id=sid, text=doc, adapter=adapter, n_turns=len(chunk), tokens=tokens)
        )
        msg = f"napped {len(chunk)} turns -> segment #{sid} ({tokens} tok -> weights)"
        self.events.append({"step": self._step, "type": "nap", "segment": sid, "tokens": tokens})
        return msg

    @staticmethod
    def _segment_doc(turns: list[Turn]) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in turns)

    def flush(self) -> MemoryStats:
        """Nap any remaining evictable context (e.g. before a cold query)."""
        evicted = ""
        while len(self.context) > self.keep_recent_turns:
            # nap whatever remains, in chunks of up to nap_every_k
            n = min(self.nap_every_k, len(self.context))
            chunk, self.context = self.context[:n], self.context[n:]
            doc = self._segment_doc(chunk)
            adapter = self.model.internalize_segment(doc)
            if not self.rank_per_segment:
                a = adapter[next(iter(adapter))]["A"]
                self.rank_per_segment = a.shape[2]
            sid = self.retriever.add(doc)
            tokens = self.model.count_tokens(doc)
            self.segments.append(
                Segment(id=sid, text=doc, adapter=adapter, n_turns=len(chunk), tokens=tokens)
            )
            evicted = f"flushed segment #{sid}"
            if not self.context:
                break
        return self._stats(evicted or "flush (nothing to evict)")

    # --- querying ---------------------------------------------------------
    def ask(self, query: str, top_k: int = 1, max_new_tokens: int = 64) -> dict:
        """Retrieve top-k segment adapters, compose, and answer.

        Returns a dict with the answer, retrieval trace, and the exact prompt
        string sent to the model (for logging / the UI side-by-side view).
        """
        recent_text = self._segment_doc(self.context) if self.context else ""
        if self.segments:
            hits = self.retriever.topk(query, k=min(top_k, len(self.segments)))
            picked = [self.segments[i] for i, _ in hits]
            composed = D2LModel.compose([s.adapter for s in picked])
            n_seg = len(picked)
        else:
            hits, picked, composed, n_seg = [], [], None, 1

        if recent_text:
            user = f"Context (recent turns):\n{recent_text}\n\nQuestion: {query}"
        else:
            user = query
        messages = [{"role": "user", "content": user}]
        raw_prompt = self.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        answer = self.model.respond(
            messages, composed=composed, n_segments=n_seg, max_new_tokens=max_new_tokens
        )
        rank = n_seg * self.rank_per_segment if composed is not None else 0
        return {
            "query": query,
            "answer": answer,
            "raw_prompt": raw_prompt,
            "retrieved": [{"segment": self.segments[i].id, "score": round(sc, 4)} for i, sc in hits],
            "n_segments_active": n_seg if composed is not None else 0,
            "adapter_rank": rank,
            "prompt_tokens": self.model.count_tokens(user),
        }

    # --- stats ------------------------------------------------------------
    def _stats(self, event: str) -> MemoryStats:
        ctx_tokens = sum(self.model.count_tokens(t.text) for t in self.context)
        return MemoryStats(
            step=self._step,
            context_turns=len(self.context),
            context_tokens=ctx_tokens,
            evicted_tokens=sum(s.tokens for s in self.segments),
            num_segments=len(self.segments),
            adapter_rank=len(self.segments) * self.rank_per_segment,
            context_budget=self.context_budget,
            last_event=event,
        )
