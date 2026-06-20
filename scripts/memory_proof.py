"""End-to-end proof for long-horizon NapLoRA memory.

Streams an agent trajectory into NapLoRAMemory: every K turns the oldest segment
is internalized into a LoRA adapter and EVICTED from context. At query time we
retrieve the top-k relevant segment adapters, compose them (rank-concat), and
answer from a tiny prompt + weights. Logs the RAW request sent for every recall.

Run:
  D2L_REPO=/root/doc-to-lora \
  AGENTHN_CHECKPOINT=/root/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  /root/doc-to-lora/.venv/bin/python scripts/memory_proof.py | tee logs/memory_proof.log
"""

from __future__ import annotations

import time

from agenthn.core.model import D2LModel
from agenthn.memory import NapLoRAMemory

# A streamed agent trajectory. Each entry is one turn; needles are planted facts
# that get evicted into weights long before they are queried.
TRAJECTORY = [
    ("user", "Kicking off the Apollo migration project. Let's track decisions here."),
    ("assistant", "Sounds good. I'll keep a running log of every decision."),
    ("user", "Decision: the deployment region is set to eu-west-2."),
    ("assistant", "Logged. Deployment region = eu-west-2."),
    ("user", "Also, the on-call engineer for launch week is Priya."),
    ("assistant", "Noted: on-call for launch week is Priya."),
    ("user", "Update: the release codename is Blue Falcon."),
    ("assistant", "Got it. Release codename = Blue Falcon."),
    ("user", "The launch date was moved to October 19."),
    ("assistant", "Updated the launch date to October 19."),
    ("user", "New: the feature flag for the new checkout is called smooth_sailing."),
    ("assistant", "Recorded feature flag smooth_sailing for the new checkout."),
    ("user", "And QA sign-off is owned by Marcus."),
    ("assistant", "Understood, Marcus owns QA sign-off."),
]

# Probes asked AFTER the trajectory, once those turns live only in weights.
PROBES = [
    ("Which region was the deployment set to?", "eu-west-2"),
    ("Who is the on-call engineer for launch week?", "priya"),
    ("What is the release codename?", "blue falcon"),
    ("What date was the launch moved to?", "october 19"),
    ("What is the name of the feature flag for the new checkout?", "smooth_sailing"),
    ("Who owns QA sign-off?", "marcus"),
]


def hit(answer: str, needle: str) -> bool:
    return needle.lower() in answer.lower()


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("LOADING D2L MODEL (gemma-2-2b-it + hypernetwork checkpoint-80000)")
    print("=" * 80)
    m = D2LModel.load()
    print(f"loaded in {time.time() - t0:.1f}s\n")

    # keep_recent_turns=0 -> EVERYTHING ends up in weights (pure weight memory).
    mem = NapLoRAMemory(m, nap_every_k=2, keep_recent_turns=0)

    print("=" * 80)
    print("STREAMING TRAJECTORY (nap every 2 turns, evict to weights)")
    print("=" * 80)
    for role, text in TRAJECTORY:
        st = mem.observe(role, text)
        print(
            f"  step {st.step:2d} | ctx={st.context_turns} turns/{st.context_tokens:>3} tok "
            f"| segments={st.num_segments} rank={st.adapter_rank:>2} | {st.last_event}"
        )
    mem.flush()
    print(f"\n  -> {len(mem.segments)} segments in weights, "
          f"{sum(s.tokens for s in mem.segments)} tokens evicted, "
          f"{len(mem.context)} turns left in context\n")

    print("=" * 80)
    print("RECALL (retrieve top-1 adapter -> compose -> answer)")
    print("=" * 80)
    n_hit = 0
    for q, needle in PROBES:
        r = mem.ask(q, top_k=1, max_new_tokens=40)
        ok = hit(r["answer"], needle)
        n_hit += ok
        print(f"\nQ: {q}")
        print(f"   RAW REQUEST: {r['raw_prompt']!r}")
        print(f"   retrieved: {r['retrieved']}  (rank {r['adapter_rank']}, "
              f"prompt {r['prompt_tokens']} tok)")
        print(f"   answer -> {r['answer']!r}  [{'HIT' if ok else 'MISS'}] "
              f"(needle {needle!r})")

    print("\n" + "=" * 80)
    print(f"RESULT: NapLoRA recovered {n_hit}/{len(PROBES)} needles from weight memory "
          f"(0 trajectory turns in context)")
    print(f"total runtime {time.time() - t0:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
