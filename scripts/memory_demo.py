"""Agentic weight-memory demo (CLI).

Story:
  1. An agent works a long session, journaling discrete observations (tool
     outputs, config it set, records it saw). Each observation is internalized
     into a LoRA block and rank-concatenated onto its running memory adapter —
     then evicted from the context window.
  2. Later, with an EMPTY context (just the question), the agent recalls those
     facts from its weights. Per-query context cost stays ~constant no matter how
     much it has remembered.
  3. We contrast four memory strategies on the same probes:
       vanilla     no memory                  -> can't answer (values are unguessable)
       naive       D2L internalize (replaces) -> only the LAST observation survives
       weight-mem  D2L rank-concat (ours)     -> recalls across observations, ~0 ctx
       in-context  whole log in the prompt     -> recalls all, but pays tokens/query

Needs the chunk-trained checkpoint (config.CHUNK_CHECKPOINT).

  CUDA_HOME=$HOME/cuda-shim \
  /home/ubuntu/doc-to-lora/.venv/bin/python scripts/memory_demo.py
"""

from agenthn.core import config
from agenthn.core.model import D2LModel
from agenthn.memory.memory_store import WeightMemory
from agenthn.memory.tasks import generate_session

N = 8
SEED = 1


def n_tokens(d2l, message):
    ids = d2l.tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        add_special_tokens=False, add_generation_prompt=True, return_tensors="pt")
    return ids.shape[1]


def main():
    print("Loading chunk-trained model...")
    d2l = D2LModel.load(config.CHUNK_CHECKPOINT)
    entries, probes = generate_session(N, seed=SEED)

    print(f"\n=== 1. AGENT SESSION — journaling {N} observations into weight memory ===")
    mem = WeightMemory(d2l, mode="doc")
    for e in entries:
        mem.remember(e.text)
        print(f"  [remember] {e.text}")
    print(f"\n  weight memory now holds {len(mem)} observations in "
          f"{len(mem.chunks)} concatenated LoRA block(s); context is empty.")

    naive_chunks = [d2l.encode_chunk(entries[-1].text)]  # replace: last memory only

    print("\n=== 2. RECALL with an EMPTY context (just the question) ===")
    score = {"vanilla": 0, "naive": 0, "weight": 0, "incontext": 0}
    for p in probes:
        d2l.reset()
        vanilla = d2l.chat(p.question, max_new_tokens=24)
        naive = d2l.chat_memory(p.question, naive_chunks, max_new_tokens=24)
        weight = mem.recall(p.question, max_new_tokens=24)
        d2l.reset()
        incontext = d2l.chat(mem.context_prompt(p.question), max_new_tokens=24)

        def mark(out):
            return "OK " if p.answer.lower() in out.lower() else "x  "

        score["vanilla"] += p.answer.lower() in vanilla.lower()
        score["naive"] += p.answer.lower() in naive.lower()
        score["weight"] += p.answer.lower() in weight.lower()
        score["incontext"] += p.answer.lower() in incontext.lower()

        print(f"\n  Q: {p.question}   (answer: {p.answer})")
        print(f"     vanilla    {mark(vanilla)} {vanilla!r}")
        print(f"     naive      {mark(naive)} {naive!r}")
        print(f"     weight-mem {mark(weight)} {weight!r}")
        print(f"     in-context {mark(incontext)} {incontext!r}")

    m = len(probes)
    q_tokens = sum(n_tokens(d2l, p.question) for p in probes) / m
    ic_tokens = sum(n_tokens(d2l, mem.context_prompt(p.question)) for p in probes) / m
    print(f"\n=== 3. SCORECARD (recall over {m} probes) ===")
    print(f"  vanilla     {score['vanilla']}/{m}")
    print(f"  naive       {score['naive']}/{m}   (only the last memory survives 'internalize')")
    print(f"  weight-mem  {score['weight']}/{m}   <- rank-concatenated LoRA blocks, OURS")
    print(f"  in-context  {score['incontext']}/{m}   (oracle)")
    print(f"\n  context tokens / query:  weight-mem ~{q_tokens:.0f}   "
          f"in-context ~{ic_tokens:.0f}  ({ic_tokens / max(q_tokens,1):.1f}x more, and growing)")


if __name__ == "__main__":
    main()
