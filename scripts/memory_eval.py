"""Measure weight-memory vs the alternatives as the agent remembers more.

For each memory size N we build a synthetic agent session of N observations and
probe recall of every one of them under four conditions:

  vanilla     base model, no memory                 (control: values are unguessable)
  naive       D2L re-internalize, replaces          (only the LAST memory survives)
  concat      D2L rank-concat memory ( OURS)        (all memories, ~0 context cost)
  in-context  dump the whole memory log into prompt (oracle, but pays tokens/query)

Reports recall accuracy and the per-query context-token cost (the thing that
makes text-memory unscalable). Averaged over several seeds.

  AGENTHN_CHECKPOINT=<chunk ckpt> \
  /home/ubuntu/doc-to-lora/.venv/bin/python scripts/memory_eval.py
"""

import argparse

import torch

from agenthn.core import config
from agenthn.core.model import D2LModel
from agenthn.memory.memory_store import WeightMemory
from agenthn.memory.tasks import generate_session


def hit(answer: str, output: str) -> bool:
    return answer.lower() in output.lower()


def n_prompt_tokens(d2l: D2LModel, message: str) -> int:
    ids = d2l.tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        add_special_tokens=False, add_generation_prompt=True, return_tensors="pt",
    )
    return ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    print(f"Loading chunk-trained model ({config.CHUNK_CHECKPOINT.parent.parent.name})...")
    d2l = D2LModel.load(config.CHUNK_CHECKPOINT)
    mnt = args.max_new_tokens

    rows = []
    for N in args.sizes:
        agg = {k: 0 for k in ("vanilla", "naive", "concat_doc", "concat_inc", "incontext")}
        total = 0
        ctx_concat = ctx_incontext = 0
        nchunks_doc = 0
        for seed in args.seeds:
            entries, probes = generate_session(N, seed=seed)

            mem_doc = WeightMemory(d2l, mode="doc")          # packs log into chunks
            mem_inc = WeightMemory(d2l, mode="incremental")  # one adapter per memory
            for e in entries:
                mem_doc.remember(e.text)
                mem_inc.remember(e.text)
            nchunks_doc += len(mem_doc.chunks)
            naive_chunks = [mem_inc.chunks[-1]]  # replace semantics: last memory only

            for p in probes:
                total += 1
                d2l.reset()
                if hit(p.answer, d2l.chat(p.question, max_new_tokens=mnt)):
                    agg["vanilla"] += 1
                if hit(p.answer, d2l.chat_memory(p.question, naive_chunks, max_new_tokens=mnt)):
                    agg["naive"] += 1
                if hit(p.answer, mem_doc.recall(p.question, max_new_tokens=mnt)):
                    agg["concat_doc"] += 1
                if hit(p.answer, mem_inc.recall(p.question, max_new_tokens=mnt)):
                    agg["concat_inc"] += 1
                d2l.reset()
                ctx_prompt = mem_doc.context_prompt(p.question)
                if hit(p.answer, d2l.chat(ctx_prompt, max_new_tokens=mnt)):
                    agg["incontext"] += 1
                ctx_concat += n_prompt_tokens(d2l, p.question)
                ctx_incontext += n_prompt_tokens(d2l, ctx_prompt)

        row = {
            "N": N,
            **{k: 100.0 * agg[k] / total for k in agg},
            "chunks": nchunks_doc / len(args.seeds),
            "ctx_concat": ctx_concat / total,
            "ctx_incontext": ctx_incontext / total,
        }
        rows.append(row)
        print(f"  N={N:2d} [{row['chunks']:.1f} chunks]  vanilla {row['vanilla']:5.1f}  "
              f"naive {row['naive']:5.1f}  concat-doc {row['concat_doc']:5.1f}  "
              f"concat-inc {row['concat_inc']:5.1f}  in-ctx {row['incontext']:5.1f}  "
              f"| tok/q  concat {row['ctx_concat']:4.0f}  in-ctx {row['ctx_incontext']:4.0f}")

    print("\n=== SUMMARY (recall accuracy %, then context tokens/query) ===")
    print(f"{'N':>3} {'chunks':>6} | {'vanilla':>7} {'naive':>7} {'cat-doc':>7} {'cat-inc':>7} "
          f"{'in-ctx':>7} | {'cat tok':>7} {'ic tok':>7}")
    for r in rows:
        print(f"{r['N']:>3} {r['chunks']:>6.1f} | {r['vanilla']:>6.1f}% {r['naive']:>6.1f}% "
              f"{r['concat_doc']:>6.1f}% {r['concat_inc']:>6.1f}% {r['incontext']:>6.1f}% | "
              f"{r['ctx_concat']:>7.0f} {r['ctx_incontext']:>7.0f}")


if __name__ == "__main__":
    main()
