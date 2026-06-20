"""Measured Cartridges baseline (faithful reimplementation, prefix-tuning).

A "cartridge" (Cartridges, HazyResearch, arXiv:2506.06266) is a small TRAINED KV
cache that stands in for a corpus, built OFFLINE by gradient descent. We reimplement
it with prefix-tuning: a learnable prefix (virtual tokens -> per-layer K,V) over a
frozen gemma-2-2b-it, trained per corpus. This is a lightweight stand-in for their
context-distillation "self-study" (we use supervised self-study questions instead
of full logit distillation), so treat the numbers as a faithful-but-conservative
lower bound on their method.

FAIR PROTOCOL (no test leakage):
  corpus      = the 20 fact statements (the information to remember)
  train on    = PARAPHRASED questions about each fact (4 templates)
  evaluate on = a HELD-OUT question phrasing ("What is the {subj} {attr}?")
So the cartridge must generalize from training phrasings to the eval phrasing,
exactly like the other methods are asked the eval phrasing cold.

Recall is independent of haystack filler (the cartridge is trained on the fact
corpus), so it is reported flat across sizes — an honest property of an offline,
whole-corpus cache vs the retrieval methods.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/doc-to-lora/.venv/bin/python scripts/cartridge_eval.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/gemma-2-2b-it"
N_FACTS = 20
SEEDS = [0, 1, 2, 3, 4]
NUM_VIRTUAL_TOKENS = 96      # the cartridge size (prefix length = per-query context)
EPOCHS = 60
LR = 5e-3
RESULTS = Path(__file__).resolve().parents[1] / "results" / "cartridge.json"

# Same fact generator as scripts/large_scale_eval.py (identical facts per seed).
_SUBJECTS = [
    "checkout", "onboarding", "webhook", "scheduler", "gateway", "billing",
    "telemetry", "ingest", "renderer", "indexer", "registry", "failover",
    "throttle", "beacon", "ledger", "rollup", "snapshot", "replica",
    "dispatcher", "sandbox", "provisioner", "collector", "router", "compactor",
]
_ATTRS = ["region code", "service ticket", "access token", "shard id",
          "revision tag", "endpoint id", "lease id", "batch label"]

# training question phrasings (NOT the eval phrasing)
_TRAIN_TEMPLATES = [
    "Tell me the {s} {a}.",
    "Remind me, what's the {s} {a}?",
    "Give me the {s} {a}, please.",
    "The {s} {a} — what was it again?",
]
_EVAL_TEMPLATE = "What is the {s} {a}?"   # held out


def gen_facts(rng, n):
    subjects = rng.sample(_SUBJECTS, n)
    facts = []
    for i, subj in enumerate(subjects):
        attr = _ATTRS[i % len(_ATTRS)]
        code = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randint(1000, 9999)}"
        facts.append((subj, attr, code))
    return facts


def build_examples(tok, facts):
    """Chat-formatted (paraphrase question -> code) pairs; loss only on the code."""
    ex = []
    for subj, attr, code in facts:
        for tmpl in _TRAIN_TEMPLATES:
            q = tmpl.format(s=subj, a=attr)
            prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                             tokenize=False, add_generation_prompt=True)
            full = prompt + code + tok.eos_token
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            ex.append((f_ids, labels))
    return ex


def train_cartridge(model, tok, facts):
    model_p = get_peft_model(model, PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM, num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        prompt_tuning_init=PromptTuningInit.RANDOM))
    model_p.train()
    opt = torch.optim.AdamW([p for p in model_p.parameters() if p.requires_grad], lr=LR)
    ex = build_examples(tok, facts)
    dev = model.device
    for ep in range(EPOCHS):
        random.shuffle(ex)
        for i in range(0, len(ex), 8):
            batch = ex[i:i + 8]
            mx = max(len(f) for f, _ in batch)
            ids = torch.full((len(batch), mx), tok.pad_token_id, device=dev)
            lab = torch.full((len(batch), mx), -100, device=dev)
            att = torch.zeros((len(batch), mx), device=dev, dtype=torch.long)
            for j, (f, l) in enumerate(batch):
                ids[j, :len(f)] = torch.tensor(f, device=dev)
                lab[j, :len(l)] = torch.tensor(l, device=dev)
                att[j, :len(f)] = 1
            opt.zero_grad()
            out = model_p(input_ids=ids, attention_mask=att, labels=lab)
            out.loss.backward()
            opt.step()
    model_p.eval()
    return model_p


@torch.inference_mode()
def evaluate(model_p, base, tok, facts):
    """Manual generation: prepend the trained soft-prompt (the cartridge) to the
    token embeddings and let the frozen base generate (bypasses a peft+gemma-2
    4D-mask bug in PeftModel.generate)."""
    dev = base.device
    emb_layer = base.get_input_embeddings()
    prompt = model_p.get_prompt(batch_size=1).to(emb_layer.weight.dtype)  # [1, V, H]
    hits = 0
    for subj, attr, code in facts:
        q = _EVAL_TEMPLATE.format(s=subj, a=attr)
        ids = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, return_tensors="pt").to(dev)
        emb = emb_layer(ids)                                   # [1, L, H]
        inputs_embeds = torch.cat([prompt, emb], dim=1)        # cartridge ++ question
        attn = torch.ones(1, inputs_embeds.shape[1], device=dev, dtype=torch.long)
        out = base.generate(inputs_embeds=inputs_embeds, attention_mask=attn,
                            max_new_tokens=24, do_sample=False)
        ans = tok.decode(out[0], skip_special_tokens=True)     # only new tokens w/ inputs_embeds
        hits += code.lower() in ans.lower()
    return hits


def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"loading {BASE}…")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda()
    base.requires_grad_(False)
    print(f"loaded ({time.time()-t0:.0f}s). cartridge={NUM_VIRTUAL_TOKENS} vtok, "
          f"epochs={EPOCHS}, seeds={SEEDS}")

    fracs, total_hits, total = [], 0, 0
    for seed in SEEDS:
        facts = gen_facts(random.Random(seed), N_FACTS)
        mp = train_cartridge(base, tok, facts)
        h = evaluate(mp, base, tok, facts)
        # prefix-tuning doesn't modify base weights (it injects past_key_values),
        # so dropping the wrapper restores a clean frozen base for the next seed.
        del mp
        base.requires_grad_(False)
        torch.cuda.empty_cache()
        fracs.append(h / N_FACTS); total_hits += h; total += N_FACTS
        print(f"  seed {seed}: {h}/{N_FACTS} recall  [{time.time()-t0:.0f}s]")

    mean = 100 * sum(fracs) / len(fracs)
    std = (sum((100 * x - mean) ** 2 for x in fracs) / len(fracs)) ** 0.5
    lo, hi = wilson(total_hits, total)
    out = {"base": BASE, "num_virtual_tokens": NUM_VIRTUAL_TOKENS, "epochs": EPOCHS,
           "seeds": SEEDS, "n_facts": N_FACTS, "n": total,
           "recall_mean": round(mean, 1), "recall_std": round(std, 1),
           "recall_pooled": round(100 * total_hits / total, 1),
           "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
           "ctx_tokens": NUM_VIRTUAL_TOKENS, "method": "prefix-tuning cartridge (reimpl)",
           "runtime_sec": round(time.time() - t0, 1)}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nCartridge recall: {out['recall_pooled']}% CI{out['ci95']} "
          f"(mean {out['recall_mean']}±{out['recall_std']}) @ {NUM_VIRTUAL_TOKENS} ctx tok")
    print(f"wrote {RESULTS}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
