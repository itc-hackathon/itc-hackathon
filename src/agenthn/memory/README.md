# Memory track — NapLoRA long-horizon memory

> Agents that write their history into their own **weights** instead of holding it
> in the context window. Built on Doc-to-LoRA (D2L) over `gemma-2-2b-it`.

## The idea

A long-running agent streams turns into a small rolling context. Every **K** turns
it **naps**: the oldest segment is internalized into a LoRA adapter via one D2L
forward pass, indexed, and **evicted from the prompt**. At query time we retrieve
the relevant segment adapter(s), compose them, and answer from a *tiny* prompt +
weights.

```
turns ──▶ [rolling context] ──nap every K──▶ internalize(segment) ──▶ LoRA adapter ──▶ store + evict
query ──▶ retrieve top-k adapters ──▶ compose (rank-concat) ──▶ generate (≈8-token prompt)
```

## What we found (measured, not assumed)

1. **A single internalized segment recalls its facts perfectly** — base gemma can't
   recall a planted fact; after `internalize()` it answers correctly (6/6 per
   segment). See [logs/memory_diag.log](../../../logs/memory_diag.log).
2. **Blindly summing ALL segments interferes destructively.** Rank-concatenating
   every segment sums their weight deltas; with 3 similar segments recall
   collapsed to **0/6** (cross-talk like `blue_sailing` = "Blue Falcon" +
   "smooth_sailing"). Scaling (1/n, 1/√n) didn't fix it.
3. **Fix = retrieve, then compose only what's needed.** Because each segment alone
   is perfect, we route each query to its segment(s). A dependency-free **TF-IDF**
   retriever routes **6/6** here (vs 4/6 mean-pool / 5/6 last-token LLM
   embeddings — see [logs/retrieval_bakeoff.log](../../../logs/retrieval_bakeoff.log)).
   This is the "retrieve top-k nap adapters" step of the MVP and what makes memory
   scale. The rank-concat composition is still used when a query needs >1 segment.

## Results — NIAH over trajectories (3 scenarios × 6 needles = 18)

Needles planted early in a 48-turn haystack, queried after they've scrolled out of
a fixed window. From [results/memory_demo.json](../../../results/memory_demo.json):

| method | recall | avg query-prompt tokens |
|---|---|---|
| **NapLoRA (weights)** | **17/18 (94%)** | **8.4** |
| markdown notes (text memory) | 16/18 (89%) | 343 |
| vanilla context (truncate to window) | 0/18 (0%) | 286 |

- **Vanilla fails entirely** — the early needles are no longer in the window.
- **NapLoRA matches/beats the `.md` baseline on recall while using ~40× fewer
  context tokens per query**, and that prompt stays *flat* (~8 tok) as history
  grows, whereas the markdown prompt grows unbounded (286→364 tok and rising).
  Query-prompt tokens ≈ KV-cache size ≈ inference cost.
- NapLoRA's one miss is a 2B generation slip ("Ayemi" vs "Adeyemi"), not a memory
  miss; markdown's misses are lossy-summary drops (a flight code, a room number).

## API

```python
from agenthn.core.model import D2LModel
from agenthn.memory import NapLoRAMemory

m = D2LModel.load()
mem = NapLoRAMemory(m, nap_every_k=4, keep_recent_turns=0)
for role, text in trajectory:
    stats = mem.observe(role, text)      # streams + naps; stats for the UI
mem.flush()                              # evict any remainder to weights
out = mem.ask("What is the release codename?", top_k=1)
#   -> {answer, raw_prompt, retrieved, adapter_rank, prompt_tokens}
```

### Core ops added to `D2LModel` (`core/model.py`)
- `internalize_segment(doc)` → per-segment adapter `{module: {A,B: [1,L,r,dim]}}`
- `compose(segments)` → rank-concat to `[n_seg,L,r,dim]` (used with `n_ctx_chunks=[n_seg]`)
- `respond(messages, composed=None, n_segments=1)` → one greedy path for the agent and baselines
- `count_tokens(text)`

### Live UI driver — `MemoryArena` (`live.py`)
Drives NapLoRA + both baselines together and yields JSON-serializable frames for
the side-by-side LIVE view (per-turn context-fill + evicted-to-weights; per-query
answers + hit/miss + prompt tokens):

```python
from agenthn.memory import MemoryArena
arena = MemoryArena(m, nap_k=4, baseline_budget=256)
for frame in arena.run(scenario.turns, scenario.probes):
    socket.send(frame)   # {"type":"turn",...} or {"type":"query",...}
```

## Scripts
- `scripts/memory_proof.py` — end-to-end NapLoRA on one trajectory (6/6), logs raw requests.
- `scripts/memory_demo.py` — full 3-way benchmark → `results/memory_demo.json` + log.
- `scripts/memory_diag.py`, `scripts/retrieval_bakeoff.py` — the interference / retrieval evidence above.

Run with the shared venv and env:
```bash
D2L_REPO=/root/doc-to-lora \
AGENTHN_CHECKPOINT=/root/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
/root/doc-to-lora/.venv/bin/python scripts/memory_demo.py | tee logs/memory_demo.log
```
