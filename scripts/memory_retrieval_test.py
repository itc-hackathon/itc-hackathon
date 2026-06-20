"""Retrieval-routed NapLoRA: embed each segment with the frozen base LLM,
route each query to the top-k segment adapter(s), compose only those, recall.

Logs the RAW request (exact prompt string) sent to the model for every call.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agenthn.core.model import D2LModel
from agenthn.memory import hit

SEGMENTS = [
    "Project log, week 1: the deployment region was set to eu-west-2. "
    "The on-call engineer for launch week is Priya.",
    "Project log, week 2: the release codename is Blue Falcon. "
    "The launch date was moved to October 19.",
    "Project log, week 3: the feature flag for the new checkout is called "
    "smooth_sailing. QA sign-off is owned by Marcus.",
]
PROBES = [
    ("Which region was the deployment set to?", "eu-west-2", 1),
    ("Who is the on-call engineer for launch week?", "priya", 1),
    ("What is the release codename?", "blue falcon", 2),
    ("What date was the launch moved to?", "october 19", 2),
    ("What is the name of the feature flag for the new checkout?", "smooth_sailing", 3),
    ("Who owns QA sign-off?", "marcus", 3),
]


@torch.inference_mode()
def embed(m: D2LModel, text: str) -> torch.Tensor:
    """Mean-pooled last-hidden-state embedding from the frozen base model."""
    m.model.reset()
    ids = m.tokenizer(text, return_tensors="pt").to(m.model.device)
    out = m.model.base_model(**ids, output_hidden_states=True)
    h = out.hidden_states[-1][0]                      # [seq, dim]
    mask = ids["attention_mask"][0].unsqueeze(-1)     # [seq, 1]
    pooled = (h * mask).sum(0) / mask.sum()
    return F.normalize(pooled.float(), dim=0)


@torch.inference_mode()
def recall(m: D2LModel, query: str, composed, n_seg: int, max_new_tokens=40) -> tuple[str, str]:
    m.model.reset()
    msgs = [{"role": "user", "content": query}]
    raw = m.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = m.tokenizer.apply_chat_template(
        msgs, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt"
    ).to(m.model.device)
    m.model.patch_lora_forward()
    m.model.generated_loras = composed
    out = m.model.generate(
        input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        n_ctx_chunks=torch.tensor([n_seg], device=m.model.device),
    )
    ans = m.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return ans, raw


def main():
    TOP_K = 1
    m = D2LModel.load()

    print("=" * 78 + "\nINDEXING: internalize + embed each segment\n" + "=" * 78)
    segs = []
    for i, text in enumerate(SEGMENTS, 1):
        adapter = m.internalize_segment(text)
        emb = embed(m, text)
        segs.append({"id": i, "text": text, "adapter": adapter, "emb": emb})
        print(f"  segment {i}: indexed ({len(text)} chars)")

    seg_embs = torch.stack([s["emb"] for s in segs])  # [n_seg, dim]

    print("\n" + "=" * 78 + f"\nQUERY (retrieve top-{TOP_K} -> compose -> recall)\n" + "=" * 78)
    n_hit = 0
    n_route = 0
    for q, needle, gold_seg in PROBES:
        qemb = embed(m, q)
        scores = seg_embs @ qemb                       # cosine (all normalized)
        order = scores.argsort(descending=True).tolist()
        picked = order[:TOP_K]
        routed_ids = [segs[j]["id"] for j in picked]
        route_ok = gold_seg in routed_ids
        n_route += route_ok

        composed = D2LModel.compose([segs[j]["adapter"] for j in picked])
        ans, raw = recall(m, q, composed, n_seg=len(picked))
        ok = hit(ans, needle)
        n_hit += ok

        print(f"\nQ: {q}")
        print(f"   RAW REQUEST: {raw!r}")
        print(f"   retrieval scores: " +
              ", ".join(f"seg{segs[j]['id']}={scores[j]:.3f}" for j in order))
        print(f"   routed -> seg{routed_ids}  (gold seg{gold_seg}) "
              f"[{'ROUTE OK' if route_ok else 'MISROUTE'}]")
        print(f"   answer -> {ans!r}  [{'HIT' if ok else 'MISS'}] (needle {needle!r})")

    print("\n" + "=" * 78)
    print(f"RESULT: routing {n_route}/{len(PROBES)} correct, "
          f"recall {n_hit}/{len(PROBES)} needles  (top_k={TOP_K})")
    print("=" * 78)


if __name__ == "__main__":
    main()
