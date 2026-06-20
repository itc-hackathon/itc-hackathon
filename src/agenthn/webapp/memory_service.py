"""Service behind the live long-horizon memory demo.

Streams a NIAH-over-trajectory through three memory strategies at once
(NapLoRA weight memory vs a markdown-notes `.md` baseline vs raw vanilla context)
and yields JSON frames for the UI: per-turn context-window fill + inference memory
(KV cache + adapter weights), then per-query answers with hit/miss and prompt cost.

Everything runs on the shared, lock-protected D2L model (see runtime.py).
"""

from __future__ import annotations

from typing import Iterator

from ..memory.live import MemoryArena
from ..memory.scenarios import (
    GEMMA_WINDOW,
    MD_K,
    NAP_K,
    SCENARIO_NAMES,
    SIZES,
    make_scenario,
)
from .runtime import MODEL_LOCK, get_model


def _kv_bytes_per_token(model) -> int:
    """KV-cache bytes per token for the base model (2 = K and V, bf16 = 2 bytes)."""
    cfg = model.model.base_model.config
    n_layers = getattr(cfg, "num_hidden_layers", 26)
    n_kv = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 8)
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // getattr(cfg, "num_attention_heads", 8)
    )
    return 2 * n_layers * n_kv * head_dim * 2


def _adapter_bytes(arena: MemoryArena) -> int:
    """Bytes of ONE per-segment LoRA adapter (A + B across modules/layers, bf16)."""
    if not arena.napora.segments:
        return 0
    total = 0
    for mats in arena.napora.segments[0].adapter.values():
        for t in mats.values():
            total += t.numel() * 2
    return total


class MemoryService:
    def __init__(self) -> None:
        self._lock = MODEL_LOCK

    def meta(self) -> dict:
        return {"scenarios": SCENARIO_NAMES, "sizes": list(SIZES),
                "window": GEMMA_WINDOW, "turns_per_size": SIZES}

    def run(self, scenario: str, size: str) -> Iterator[dict]:
        """Yield frames for the whole run (held under the model lock)."""
        size = size if size in SIZES else "medium"
        scenario = scenario if scenario in SCENARIO_NAMES else SCENARIO_NAMES[0]
        sc = make_scenario(scenario, size)
        with self._lock:
            model = get_model()
            kv_per_tok = _kv_bytes_per_token(model)
            arena = MemoryArena(model, nap_k=NAP_K[size], baseline_budget=GEMMA_WINDOW,
                                md_k=MD_K[size])

            yield {
                "type": "meta", "scenario": sc.name, "size": size,
                "total_turns": len(sc.turns), "nap_k": NAP_K[size],
                "window": GEMMA_WINDOW, "needle_positions": sc.needle_positions,
                "probes": [{"q": q, "needle": n} for q, n in sc.probes],
                "kv_mb_per_1k": round(kv_per_tok * 1000 / 1024 / 1024, 2),
            }

            raw_tokens = 0  # cumulative tokens if you naively kept the whole transcript
            prev_segments = 0
            prev_notes = 0
            for role, text in sc.turns:
                frame = arena.observe(role, text)
                raw_tokens += model.count_tokens(text) + 4
                adapter_mb = _adapter_bytes(arena) * len(arena.napora.segments) / 1024 / 1024
                # enrich with memory + window-fill fields
                frame["window"] = GEMMA_WINDOW
                frame["raw_tokens"] = raw_tokens
                frame["text_tokens"] = model.count_tokens(text)
                nap = frame["napora"]
                nap["kv_mb"] = round(nap["context_tokens"] * kv_per_tok / 1024 / 1024, 2)
                nap["adapter_mb"] = round(adapter_mb, 1)
                nap["fill_pct"] = round(100 * nap["context_tokens"] / GEMMA_WINDOW, 1)
                # a nap happened this step iff the adapter count grew
                nap["napped"] = nap["segments"] > prev_segments
                prev_segments = nap["segments"]
                van = frame["vanilla"]
                van["kv_mb"] = round(van["context_tokens"] * kv_per_tok / 1024 / 1024, 2)
                van["raw_tokens"] = raw_tokens
                van["overflow"] = raw_tokens > GEMMA_WINDOW
                van["fill_pct"] = round(100 * min(raw_tokens, GEMMA_WINDOW) / GEMMA_WINDOW, 1)
                md = frame["markdown"]
                md["kv_mb"] = round(md["context_tokens"] * kv_per_tok / 1024 / 1024, 2)
                md["fill_pct"] = round(100 * md["context_tokens"] / GEMMA_WINDOW, 1)
                # send the FULL .md notes only when they changed (bounds payload); the
                # UI keeps the last set otherwise. Emit a summarization event too.
                n_notes = len(arena.markdown.notes)
                if n_notes != prev_notes:
                    md["notes"] = list(arena.markdown.notes)
                    md["event"] = (
                        f"summarized turns → +{n_notes - prev_notes} note(s) "
                        f"(now {n_notes} lines, {md['context_tokens']} tok in prompt)"
                    )
                    prev_notes = n_notes
                else:
                    md["notes"] = None
                yield frame

            for q, needle in sc.probes:
                frame = arena.ask(q, needle=needle)
                for name in ("napora", "vanilla", "markdown"):
                    m = frame["methods"][name]
                    m["kv_mb"] = round(m["prompt_tokens"] * kv_per_tok / 1024 / 1024, 2)
                yield frame

            yield {"type": "done", "scenario": sc.name, "size": size}


def build_memory_service() -> MemoryService:
    return MemoryService()


__all__ = ["MemoryService", "build_memory_service"]
