"""Long-horizon memory demo: NapLoRA vs vanilla-context vs markdown-notes.

For each NIAH-over-trajectory scenario we stream the same long conversation into
all three memory strategies, then ask the same probe questions and score recall.
We also record context cost (query-time prompt tokens) and a per-step timeline of
context fill, so Eric's UI can render the "context window zooming out" + cost
panels. Writes a machine-readable results/memory_demo.json and prints a summary.

Run:
  D2L_REPO=/root/doc-to-lora \
  AGENTHN_CHECKPOINT=/root/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  /root/doc-to-lora/.venv/bin/python scripts/memory_demo.py | tee logs/memory_demo.log
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from agenthn.core.model import D2LModel
from agenthn.memory import MarkdownMemory, NapLoRAMemory, VanillaContextMemory
from agenthn.memory.scenarios import all_scenarios

NAP_K = 4
BASELINE_BUDGET = 256          # the fixed "context window" the baselines share
RESULTS = Path(__file__).resolve().parents[1] / "results" / "memory_demo.json"


def hit(answer: str, needle: str) -> bool:
    return needle.lower() in answer.lower()


def run_napora(model, scenario):
    """Stream into weight memory; capture a per-step context-fill timeline."""
    mem = NapLoRAMemory(model, nap_every_k=NAP_K, keep_recent_turns=0)
    timeline = []
    for role, text in scenario.turns:
        st = mem.observe(role, text)
        timeline.append({"step": st.step, "context_tokens": st.context_tokens,
                         "evicted_tokens": st.evicted_tokens, "segments": st.num_segments})
    mem.flush()
    results = []
    for q, needle in scenario.probes:
        r = mem.ask(q, top_k=1, max_new_tokens=40)
        r["hit"] = hit(r["answer"], needle)
        r["needle"] = needle
        results.append(r)
    return {
        "method": "napora",
        "timeline": timeline,
        "results": results,
        "segments": len(mem.segments),
        "evicted_tokens": sum(s.tokens for s in mem.segments),
    }


def run_textmem(model, scenario, kind):
    """kind in {'vanilla','markdown'} — same streaming, text-based memory."""
    if kind == "vanilla":
        mem = VanillaContextMemory(model, context_budget_tokens=BASELINE_BUDGET)
    else:
        mem = MarkdownMemory(model, summarize_every_k=NAP_K)
    timeline = []
    for i, (role, text) in enumerate(scenario.turns, 1):
        mem.observe(role, text)
        if kind == "vanilla":
            window, used = mem._window()
            timeline.append({"step": i, "context_tokens": used, "turns_in_window": len(window)})
        else:
            timeline.append({"step": i, "context_tokens": mem.model.count_tokens(mem._md()),
                             "notes_lines": len(mem.notes)})
    results = []
    for q, needle in scenario.probes:
        r = mem.ask(q, max_new_tokens=40)
        r["hit"] = hit(r["answer"], needle)
        r["needle"] = needle
        results.append(r)
    return {"method": kind, "timeline": timeline, "results": results}


def main():
    t0 = time.time()
    print("Loading model...")
    model = D2LModel.load()
    print(f"loaded in {time.time() - t0:.1f}s")

    scenarios = all_scenarios()
    out = {
        "config": {"nap_k": NAP_K, "baseline_budget": BASELINE_BUDGET,
                   "base_model": "google/gemma-2-2b-it"},
        "scenarios": [],
    }
    totals = {"napora": [0, 0], "vanilla": [0, 0], "markdown": [0, 0]}  # [hits, total]

    for sc in scenarios:
        print("\n" + "=" * 80)
        print(f"SCENARIO: {sc.name}  ({len(sc.turns)} turns, {len(sc.probes)} probes, "
              f"needles at turns {sc.needle_positions})")
        print("=" * 80)
        methods = {
            "napora": run_napora(model, sc),
            "vanilla": run_textmem(model, sc, "vanilla"),
            "markdown": run_textmem(model, sc, "markdown"),
        }
        # side-by-side probe table
        for i, (q, needle) in enumerate(sc.probes):
            print(f"\nQ{i+1}: {q}   (needle: {needle!r})")
            for name in ("napora", "vanilla", "markdown"):
                r = methods[name]["results"][i]
                mark = "HIT " if r["hit"] else "miss"
                print(f"   {name:9s} [{mark}] ({r['prompt_tokens']:>4} tok) -> {r['answer'][:64]!r}")

        sc_out = {"name": sc.name, "needle_positions": sc.needle_positions, "methods": {}}
        for name, m in methods.items():
            hits = sum(r["hit"] for r in m["results"])
            tot = len(m["results"])
            avg_tok = sum(r["prompt_tokens"] for r in m["results"]) / tot
            totals[name][0] += hits
            totals[name][1] += tot
            m["recall"] = f"{hits}/{tot}"
            m["avg_prompt_tokens"] = round(avg_tok, 1)
            sc_out["methods"][name] = m
            print(f"   -> {name:9s} recall {hits}/{tot}, avg prompt {avg_tok:.0f} tok")
        out["scenarios"].append(sc_out)

    print("\n" + "=" * 80)
    print("OVERALL")
    print("=" * 80)
    summary = {}
    for name in ("napora", "vanilla", "markdown"):
        hits, tot = totals[name]
        # average prompt tokens across all probes/scenarios
        toks = [r["prompt_tokens"] for sc in out["scenarios"]
                for r in sc["methods"][name]["results"]]
        avg = sum(toks) / len(toks)
        summary[name] = {"recall": f"{hits}/{tot}", "recall_pct": round(100 * hits / tot, 1),
                         "avg_prompt_tokens": round(avg, 1)}
        print(f"  {name:9s}: recall {hits}/{tot} ({100*hits/tot:.0f}%), "
              f"avg query prompt {avg:.0f} tokens")
    out["summary"] = summary
    out["runtime_sec"] = round(time.time() - t0, 1)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")
    print(f"total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
