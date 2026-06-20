# skill.md → D2L adapter → base-vs-adapter eval

Turns a **skill.md** document into a **LoRA adapter** with Doc-to-LoRA (D2L), then
measures its effect: solve a problem with the **base model**, then with **base +
adapter**, and — when generation is deterministic — score correctness of each.

## Pipeline (`skill_to_adapter.py`)
1. **Base model** answers the problem (adapter reset).
2. **skill.md → adapter**: `model.internalize(skill_text)` generates a LoRA, which is
   **exported to disk** (`adapters/<skill>.lora.pt` = generated LoRA weights + metadata).
3. **Base + adapter** answers the same problem.
4. **Determinism check**: greedy decode is run twice per model; `deterministic` is true
   only if both reproduce themselves.
5. **Correctness** (only if deterministic + `--expected` given): normalized
   exact-or-contains match (`' | '` requires all key phrases). Reuses
   `ctx_to_lora.eval_utils.normalize_answer`.
6. **JSON** with both outputs + scoring → `outputs/<skill>.result.json`.

## Run
Requires the **doc2lora** conda env (torch 2.7.1+cu128, flash-attn for sm_90) with the
D2L package installed editable from `/home/nikash/projects/doc-to-lora`.

```bash
conda run -n doc2lora env PYTHONNOUSERSITE=1 python skill_to_adapter.py
```

Useful flags: `--skill <path.md>` `--problem "<q>"|@file` `--expected "a | b"`
`--checkpoint <bin>` `--max-new-tokens N` `--out <json>` `--adapter-out <pt>`
`--reload-adapter <pt>` (load a saved adapter and regenerate, skipping internalization).

## Notes
- Base model uses the ungated mirror **`unsloth/gemma-2-2b-it`** (identical weights to the
  gated `google/gemma-2-2b-it`), so no HF license acceptance is needed. The script chdirs
  into the D2L repo root so the custom chat template (`chat_templates/unsloth/...`) resolves.
- Default checkpoint: `trained_d2l/gemma_demo/checkpoint-80000` (SakanaAI/doc-to-lora).

## Example result (`outputs/systematic-debugging.result.json`)
Skill: `skills/systematic-debugging.md` (from github.com/obra/superpowers). Problem asks for
the skill's "Iron Law". Correct = expresses *find the root cause before fixing* (`root cause | before`).

| Model | Answer | Correct |
|-------|--------|---------|
| **base** | "…break down the problem into smaller, manageable pieces." | ❌ |
| **base + adapter** | "…always find the root cause before attempting to fix it." | ✅ |

`deterministic: true`. The adapter recovers the skill's core principle (the Iron Law:
"NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST") where the base model gives a generic,
wrong answer. The exported adapter reloads via `--reload-adapter` and reproduces the
identical output without re-internalizing.

> Note on scoring: D2L on a 2B model reliably internalizes the skill's *meaning* but not
> always verbatim terminology — e.g. asking for the four phase names by exact wording
> scores false for both models even though the adapter shifts the answer. Correctness here
> targets the principle (concept-level key phrases), which is the faithful signal.
