"""Ablation study: isolate where long-horizon NIAH recall actually comes from.

We hold the task fixed (6 facts planted early, conversation grown until it
overflows the 8k window) and remove ONE component of NapLoRA at a time. Each row
reports recall and the query-time context the model must attend to (≈ KV cost).

  vanilla        no memory at all                         -> control (unguessable)
  naive          internalize each segment, REPLACE        -> no store / no retrieval
                 (only the newest adapter is active)          (isolates: need a store)
  compose-all    store all adapters, rank-concat ALL      -> no retrieval
                 at query (no routing)                        (isolates: need retrieval)
  text-RAG       retrieve top-1 segment, inject as TEXT   -> no weight-injection
                                                              (isolates: LoRA compaction)
  NapLoRA        retrieve top-1 segment, inject as LoRA   -> full system

The text-RAG vs NapLoRA row is the key one: same retriever, same recall — the
only difference is whether the retrieved memory's tokens enter the context. That
difference is exactly what the hypernetwork (Doc-to-LoRA) buys.

  D2L_REPO=... AGENTHN_CHECKPOINT=... PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/doc-to-lora/.venv/bin/python scripts/ablation_study.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from agenthn.core.model import D2LModel
from agenthn.memory import NapLoRAMemory, TextRAGMemory, VanillaContextMemory
from agenthn.memory.scenarios import GEMMA_WINDOW, _FILLER, _NEEDLE_SETS

NAP_K = 4
TARGET_TOKENS = 12000           # overflow the 8k window so vanilla must fail
RESULTS = Path(__file__).resolve().parents[1] / "results" / "ablations.json"


def hit(ans: str, needle: str) -> bool:
    return needle.lower() in ans.lower()


def build_stream(model, target):
    needles, probes = _NEEDLE_SETS["apollo_migration"]
    turns, fi = [], 0
    for role, text in needles:                       # each needle + K-1 filler
        turns.append((role, text))
        for _ in range(NAP_K - 1):
            turns.append(("user", _FILLER[fi % len(_FILLER)])); fi += 1
    while sum(model.count_tokens(t) + 4 for _, t in turns) < target:
        turns.append(("user", _FILLER[fi % len(_FILLER)])); fi += 1
    return turns, probes


@torch.inference_mode()
def respond_with_adapters(model, query, adapters):
    """Generate from a query-only prompt with a composed set of adapters active."""
    composed = D2LModel.compose(adapters)
    msgs = [{"role": "user", "content": query}]
    return model.respond(msgs, composed=composed, n_segments=len(adapters), max_new_tokens=40)


def main():
    t0 = time.time()
    model = D2LModel.load()
    turns, probes = build_stream(model, TARGET_TOKENS)
    raw = sum(model.count_tokens(t) + 4 for _, t in turns)
    print(f"haystack: {len(turns)} turns, {raw} tokens (window={GEMMA_WINDOW}), "
          f"6 needles in first {6*NAP_K} turns")

    nap = NapLoRAMemory(model, nap_every_k=NAP_K, keep_recent_turns=0)
    rag = TextRAGMemory(model, nap_every_k=NAP_K)
    van = VanillaContextMemory(model, context_budget_tokens=GEMMA_WINDOW)
    for r, t in turns:
        nap.observe(r, t); rag.observe(r, t); van.observe(r, t)
    nap.flush(); rag.flush()
    adapters = [s.adapter for s in nap.segments]
    print(f"{len(adapters)} segments stored")

    def score(fn):
        """fn(query) -> (answer, query_context_tokens). Called once per probe."""
        ctx, h = [], 0
        for q, nd in probes:
            ans, pt = fn(q)
            h += hit(ans, nd); ctx.append(pt)
        return h, round(sum(ctx) / len(ctx), 1)

    qtok = model.count_tokens  # query-only prompt size proxy
    conds = []

    def van_fn(q):
        r = van.ask(q); return r["answer"], r["prompt_tokens"]

    def rag_fn(q):
        r = rag.ask(q); return r["answer"], r["prompt_tokens"]

    def nap_fn(q):
        r = nap.ask(q, top_k=1); return r["answer"], r["prompt_tokens"]

    last = [adapters[-1]]
    for name, variant, fn, iso in [
        ("vanilla", "no memory", van_fn, "memory is needed"),
        ("naive-internalize", "internalize, replace (no store)",
         lambda q: (respond_with_adapters(model, q, last), qtok(q)), "a store is needed"),
        ("compose-all", "rank-concat ALL (no retrieval)",
         lambda q: (respond_with_adapters(model, q, adapters), qtok(q)), "retrieval is needed"),
        ("text-RAG", "retrieve top-1 as TEXT", rag_fn, "(weight-injection removed)"),
        ("NapLoRA", "retrieve top-1 as LoRA (full)", nap_fn, "full system"),
    ]:
        h, c = score(fn)
        conds.append((name, variant, h, c, iso))
        print(f"  {name:18s} {h}/6  @{c}tok"); torch.cuda.empty_cache()

    out = {
        "window": GEMMA_WINDOW, "nap_k": NAP_K, "raw_tokens": raw,
        "n_probes": len(probes), "segments": len(adapters),
        "rows": [{"name": n, "variant": v, "recall": h, "recall_pct": round(100*h/len(probes)),
                  "query_ctx_tokens": c, "isolates": iso} for n, v, h, c, iso in conds],
        "runtime_sec": round(time.time() - t0, 1),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
