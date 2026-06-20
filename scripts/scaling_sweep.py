"""Scaling sweep: how far past the context window can weight-memory go?

The Doc-to-LoRA paper encodes a *bounded* document in ONE pass (chunk →
rank-concatenate, capped at ~MAX_CHUNKS=8 before going out-of-distribution),
reaching ~4× the native window. We instead nap REPEATEDLY over a stream: 6 facts
are planted early, then the conversation grows to ~48k tokens. Each nap is an
independent in-distribution single-chunk adapter added to a retrievable store, so
there is no rank-concat ceiling — at query time we retrieve the relevant nap.

We stream one long haystack once and, at token checkpoints, measure recall +
query-time prompt tokens (≈ KV-cache cost) for NapLoRA vs a vanilla 8k-window
model. Emits results/scaling.json for the homepage chart.

  D2L_REPO=/root/doc-to-lora \
  AGENTHN_CHECKPOINT=.../gemma_demo/checkpoint-80000/pytorch_model.bin \
  /root/doc-to-lora/.venv/bin/python scripts/scaling_sweep.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from agenthn.core.model import D2LModel
from agenthn.memory import NapLoRAMemory, VanillaContextMemory
from agenthn.memory.scenarios import GEMMA_WINDOW, _FILLER, _NEEDLE_SETS

CHECKPOINTS = [1000, 2000, 4000, 8000, 16000, 32000, 48000]
NAP_K = 4
RESULTS = Path(__file__).resolve().parents[1] / "results" / "scaling.json"


def hit(ans: str, needle: str) -> bool:
    return needle.lower() in ans.lower()


def build_stream(model: D2LModel, target_tokens: int):
    """6 needles planted early (each in its own K=4 nap segment), then filler."""
    needles, probes = _NEEDLE_SETS["apollo_migration"]
    turns: list[tuple[str, str]] = []
    # needle, then K-1 filler -> each needle lands in its own nap segment
    fi = 0
    for role, text in needles:
        turns.append((role, text))
        for _ in range(NAP_K - 1):
            turns.append(("assistant" if len(turns) % 2 else "user",
                          _FILLER[fi % len(_FILLER)]))
            fi += 1
    # pad with filler until we exceed the largest checkpoint
    tok = sum(model.count_tokens(t) + 4 for _, t in turns)
    while tok < target_tokens:
        t = _FILLER[fi % len(_FILLER)]
        turns.append(("assistant" if len(turns) % 2 else "user", t))
        tok += model.count_tokens(t) + 4
        fi += 1
    return turns, probes


def main() -> None:
    t0 = time.time()
    print("loading model…")
    model = D2LModel.load()
    print(f"loaded in {time.time() - t0:.1f}s")

    turns, probes = build_stream(model, CHECKPOINTS[-1])
    print(f"haystack: {len(turns)} turns, 6 needles planted in the first {6 * NAP_K} turns")

    nap = NapLoRAMemory(model, nap_every_k=NAP_K, keep_recent_turns=0)
    van = VanillaContextMemory(model, context_budget_tokens=GEMMA_WINDOW)

    rows = []
    pending = list(CHECKPOINTS)
    raw = 0
    for i, (role, text) in enumerate(turns):
        nap.observe(role, text)
        van.observe(role, text)
        raw += model.count_tokens(text) + 4
        if pending and raw >= pending[0]:
            ckpt = pending.pop(0)
            nap.flush()
            nres = [nap.ask(q, top_k=1, max_new_tokens=40) for q, _ in probes]
            n_hit = sum(hit(r["answer"], nd) for r, (_, nd) in zip(nres, probes))
            n_tok = sum(r["prompt_tokens"] for r in nres) / len(nres)
            # Vanilla: only actually generate while the transcript fits the window.
            # Once raw > window the early needles are provably outside the prompt,
            # so recall is 0 and the prompt is pinned at the window size — no need
            # to run (and a full-window generation is what crashed the sweep).
            if raw <= GEMMA_WINDOW:
                vres = [van.ask(q, max_new_tokens=40) for q, _ in probes]
                v_hit = sum(hit(r["answer"], nd) for r, (_, nd) in zip(vres, probes))
                v_tok = sum(r["prompt_tokens"] for r in vres) / len(vres)
            else:
                v_hit = 0
                v_tok = float(GEMMA_WINDOW)
            adapter_mb = 0.0
            if nap.segments:
                b = sum(t.numel() * 2 for mats in nap.segments[0].adapter.values()
                        for t in mats.values())
                adapter_mb = b * len(nap.segments) / 1024 / 1024
            row = {
                "checkpoint": ckpt, "raw_tokens": raw, "segments": len(nap.segments),
                "napora_recall": n_hit, "vanilla_recall": v_hit,
                "napora_prompt_tokens": round(n_tok, 1), "vanilla_prompt_tokens": round(v_tok, 1),
                "napora_adapter_mb": round(adapter_mb, 1), "n_probes": len(probes),
            }
            rows.append(row)
            torch.cuda.empty_cache()  # release the big full-window KV cache
            print(f"  ~{ckpt:>5} tok ({raw} raw, {len(nap.segments)} naps): "
                  f"NapLoRA {n_hit}/6 @{n_tok:.0f}tok  |  vanilla {v_hit}/6 @{v_tok:.0f}tok  "
                  f"[{time.time() - t0:.0f}s]")

    out = {
        "window": GEMMA_WINDOW,
        "nap_k": NAP_K,
        "paper_limit_tokens": 4 * GEMMA_WINDOW,  # D2L: ~4x native window, single encode
        "rows": rows,
        "runtime_sec": round(time.time() - t0, 1),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
