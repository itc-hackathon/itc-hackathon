"""Large-scale statistical evaluation of NapLoRA long-horizon memory.

Goes beyond the 6-needle demo: synthesize MANY distinct facts, multiple random
seeds, and haystacks that grow from 2k to 48k tokens. Reports recall as
mean ± std across seeds with Wilson 95% CIs, so the findings are defensible.

Conditions (all share the same retriever + segmentation):
  NapLoRA   retrieve top-1 nap, inject as LoRA adapter   (~constant query context)
  text-RAG  retrieve top-1 nap, inject as text           (ablation: weights -> tokens)
  vanilla   no memory, 8k context window                 (control)

Headline tests:
  H1  NapLoRA recall stays high as the haystack grows; vanilla collapses past 8k.
  H2  NapLoRA == text-RAG on recall (no significant difference) at ~10x less context
      -> the hypernetwork's gain is context cost, not recall (honest, isolated).

  D2L_REPO=... AGENTHN_CHECKPOINT=... PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/doc-to-lora/.venv/bin/python scripts/large_scale_eval.py
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from agenthn.core.model import D2LModel
from agenthn.memory import NapLoRAMemory, TextRAGMemory, VanillaContextMemory
from agenthn.memory.scenarios import GEMMA_WINDOW, _FILLER

NAP_K = 4
SEEDS = [0, 1, 2, 3, 4]
N_FACTS = 20
SIZES = [2000, 8000, 16000, 32000, 48000]
RESULTS = Path(__file__).resolve().parents[1] / "results" / "large_scale.json"

# Distinct subject words (none appear in _FILLER) so TF-IDF routes each query to
# its own segment; paired with a varied attribute and a distinctive code value.
_SUBJECTS = [
    "checkout", "onboarding", "webhook", "scheduler", "gateway", "billing",
    "telemetry", "ingest", "renderer", "indexer", "registry", "failover",
    "throttle", "beacon", "ledger", "rollup", "snapshot", "replica",
    "dispatcher", "sandbox", "provisioner", "collector", "router", "compactor",
]
_ATTRS = ["region code", "service ticket", "access token", "shard id",
          "revision tag", "endpoint id", "lease id", "batch label"]


def gen_facts(rng: random.Random, n: int):
    """n distinct (statement, question, needle) triples with distinctive codes."""
    subjects = rng.sample(_SUBJECTS, n)
    facts = []
    for i, subj in enumerate(subjects):
        attr = _ATTRS[i % len(_ATTRS)]
        code = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randint(1000, 9999)}"
        stmt = f"Note for the record: the {subj} {attr} is {code}."
        q = f"What is the {subj} {attr}?"
        facts.append((stmt, q, code.lower()))
    return facts


def build_stream(model, rng, facts, target_tokens):
    turns, fi = [], 0
    for stmt, _, _ in facts:                 # each fact + K-1 filler -> own segment
        turns.append(("user", stmt))
        for _ in range(NAP_K - 1):
            turns.append(("user", _FILLER[fi % len(_FILLER)])); fi += 1
    fi = rng.randint(0, len(_FILLER) - 1)  # vary filler ordering by seed
    while sum(model.count_tokens(t) + 4 for _, t in turns) < target_tokens:
        turns.append(("user", _FILLER[fi % len(_FILLER)])); fi += 1
    return turns


def hit(ans, needle):
    return needle.lower() in ans.lower()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def main():
    t0 = time.time()
    model = D2LModel.load()
    print(f"loaded ({time.time()-t0:.0f}s). seeds={SEEDS} facts/seed={N_FACTS} sizes={SIZES}")

    # per size -> per method -> list of per-seed recall fractions, and pooled hits/total
    agg = {s: {m: {"frac": [], "hits": 0, "total": 0, "ctx": []}
               for m in ("napora", "rag", "vanilla")} for s in SIZES}

    for seed in SEEDS:
        rng = random.Random(seed)
        facts = gen_facts(rng, N_FACTS)
        turns = build_stream(model, rng, facts, SIZES[-1])
        nap = NapLoRAMemory(model, nap_every_k=NAP_K, keep_recent_turns=0)
        rag = TextRAGMemory(model, nap_every_k=NAP_K)
        van = VanillaContextMemory(model, context_budget_tokens=GEMMA_WINDOW)
        pending = list(SIZES)
        raw = 0
        for role, text in turns:
            nap.observe(role, text); rag.observe(role, text); van.observe(role, text)
            raw += model.count_tokens(text) + 4
            if pending and raw >= pending[0]:
                size = pending.pop(0)
                nap.flush(); rag.flush()
                for label, fn in (("napora", lambda q: nap.ask(q, top_k=1, max_new_tokens=24)),
                                  ("rag", lambda q: rag.ask(q, max_new_tokens=24))):
                    h = 0; ctx = []
                    for _, q, nd in facts:
                        r = fn(q); h += hit(r["answer"], nd); ctx.append(r["prompt_tokens"])
                    cell = agg[size][label]
                    cell["frac"].append(h / len(facts)); cell["hits"] += h
                    cell["total"] += len(facts); cell["ctx"].append(sum(ctx) / len(ctx))
                # vanilla: only generate while it fits the window; else structural 0
                vcell = agg[size]["vanilla"]
                if raw <= GEMMA_WINDOW:
                    h = 0; ctx = []
                    for _, q, nd in facts:
                        r = van.ask(q, max_new_tokens=24); h += hit(r["answer"], nd); ctx.append(r["prompt_tokens"])
                    vcell["frac"].append(h / len(facts)); vcell["hits"] += h
                    vcell["total"] += len(facts); vcell["ctx"].append(sum(ctx) / len(ctx))
                else:
                    vcell["frac"].append(0.0); vcell["hits"] += 0
                    vcell["total"] += len(facts); vcell["ctx"].append(float(GEMMA_WINDOW))
                torch.cuda.empty_cache()
        print(f"  seed {seed} done [{time.time()-t0:.0f}s]  "
              + "  ".join(f"{s//1000}k:N={agg[s]['napora']['frac'][-1]:.2f}/R={agg[s]['rag']['frac'][-1]:.2f}/V={agg[s]['vanilla']['frac'][-1]:.2f}" for s in SIZES))

    def stats(cell):
        fr = cell["frac"]
        mean = sum(fr) / len(fr)
        std = (sum((x - mean) ** 2 for x in fr) / len(fr)) ** 0.5
        lo, hi = wilson(cell["hits"], cell["total"])
        return {"mean": round(100 * mean, 1), "std": round(100 * std, 1),
                "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
                "pooled": round(100 * cell["hits"] / cell["total"], 1),
                "n": cell["total"], "ctx_tokens": round(sum(cell["ctx"]) / len(cell["ctx"]), 1)}

    rows = []
    for s in SIZES:
        rows.append({"size": s,
                     "napora": stats(agg[s]["napora"]),
                     "rag": stats(agg[s]["rag"]),
                     "vanilla": stats(agg[s]["vanilla"])})

    out = {"window": GEMMA_WINDOW, "nap_k": NAP_K, "seeds": SEEDS, "n_facts": N_FACTS,
           "sizes": SIZES, "rows": rows, "runtime_sec": round(time.time() - t0, 1)}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}  ({time.time()-t0:.0f}s)")
    big = rows[-1]
    print(f"\nAt {SIZES[-1]} tok ({big['napora']['n']} trials/method):")
    print(f"  NapLoRA  {big['napora']['pooled']}% CI{big['napora']['ci95']} @ {big['napora']['ctx_tokens']} ctx")
    print(f"  text-RAG {big['rag']['pooled']}% CI{big['rag']['ci95']} @ {big['rag']['ctx_tokens']} ctx")
    print(f"  vanilla  {big['vanilla']['pooled']}% CI{big['vanilla']['ci95']} @ {big['vanilla']['ctx_tokens']} ctx")


if __name__ == "__main__":
    main()
