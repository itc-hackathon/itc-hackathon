"""Diagnose NapLoRA segment interference: per-segment recall vs composed,
and whether per-segment down-scaling (scalers) fixes the cross-talk."""

from __future__ import annotations

import torch

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
def gen(m, q, composed, n_seg, scalers=None):
    m.model.reset()
    ids = m.tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        add_special_tokens=False, add_generation_prompt=True, return_tensors="pt",
    ).to(m.model.device)
    kw = dict(input_ids=ids, max_new_tokens=40, do_sample=False, num_beams=1)
    if composed is not None:
        m.model.patch_lora_forward()
        m.model.generated_loras = composed
        kw["n_ctx_chunks"] = torch.tensor([n_seg], device=m.model.device)
        if scalers is not None:
            kw["scalers"] = scalers.to(m.model.device)
    out = m.model.generate(**kw)
    return m.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def score(m, segs, label, scalers=None):
    composed = D2LModel.compose(segs)
    n = len(segs)
    nh = 0
    print(f"\n--- {label} (n_seg={n}, scalers={None if scalers is None else scalers.tolist()}) ---")
    for q, needle, sid in PROBES[: 2 * n]:
        a = gen(m, q, composed, n, scalers)
        ok = hit(a, needle)
        nh += ok
        print(f"  seg{sid} [{'HIT' if ok else 'miss'}] {needle!r}: {a[:70]!r}")
    print(f"  => {nh}/{2*n}")
    return nh


def main():
    m = D2LModel.load()
    segs = [m.internalize_segment(s) for s in SEGMENTS]

    # 1) each segment ALONE
    print("=" * 70 + "\nPER-SEGMENT (1 adapter active at a time)\n" + "=" * 70)
    for i, s in enumerate(segs, 1):
        composed = D2LModel.compose([s])
        for q, needle, sid in PROBES:
            if sid != i:
                continue
            a = gen(m, q, composed, 1)
            print(f"  seg{i} [{'HIT' if hit(a,needle) else 'miss'}] {needle!r}: {a[:70]!r}")

    # 2) pairwise + triple, unscaled vs scaled
    print("\n" + "=" * 70 + "\nCOMPOSED, scaler sweep\n" + "=" * 70)
    for n in (2, 3):
        sub = segs[:n]
        score(m, sub, f"unscaled x{n}")
        score(m, sub, f"1/n x{n}", scalers=torch.full((n,), 1.0 / n))
        score(m, sub, f"1/sqrt(n) x{n}", scalers=torch.full((n,), n ** -0.5))


if __name__ == "__main__":
    main()
