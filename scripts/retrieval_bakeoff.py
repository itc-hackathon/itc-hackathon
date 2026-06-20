"""Retrieval bakeoff: which scorer routes each NIAH query to the right segment?
No generation — just routing accuracy. Compares mean-pool / last-token LLM
embeddings vs a dependency-free lexical TF-IDF cosine."""

from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn.functional as F

from agenthn.core.model import D2LModel

SEGMENTS = [
    "Project log, week 1: the deployment region was set to eu-west-2. "
    "The on-call engineer for launch week is Priya.",
    "Project log, week 2: the release codename is Blue Falcon. "
    "The launch date was moved to October 19.",
    "Project log, week 3: the feature flag for the new checkout is called "
    "smooth_sailing. QA sign-off is owned by Marcus.",
]
PROBES = [
    ("Which region was the deployment set to?", 1),
    ("Who is the on-call engineer for launch week?", 1),
    ("What is the release codename?", 2),
    ("What date was the launch moved to?", 2),
    ("What is the name of the feature flag for the new checkout?", 3),
    ("Who owns QA sign-off?", 3),
]


def tok(s):
    return re.findall(r"[a-z0-9_]+", s.lower())


def tfidf_index(docs):
    toks = [tok(d) for d in docs]
    df = Counter()
    for t in toks:
        df.update(set(t))
    n = len(docs)
    idf = {w: math.log((n + 1) / (df[w] + 1)) + 1 for w in df}
    vecs = []
    for t in toks:
        tf = Counter(t)
        vecs.append({w: tf[w] * idf.get(w, 0.0) for w in tf})
    return idf, vecs


def tfidf_vec(text, idf):
    tf = Counter(tok(text))
    return {w: tf[w] * idf.get(w, math.log(1) + 1) for w in tf}


def cos_sparse(a, b):
    if not a or not b:
        return 0.0
    dot = sum(a[w] * b.get(w, 0.0) for w in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb + 1e-9)


@torch.inference_mode()
def llm_embed(m, text, mode):
    m.model.reset()
    ids = m.tokenizer(text, return_tensors="pt").to(m.model.device)
    out = m.model.base_model(**ids, output_hidden_states=True)
    h = out.hidden_states[-1][0]
    mask = ids["attention_mask"][0]
    if mode == "mean":
        pooled = (h * mask.unsqueeze(-1)).sum(0) / mask.sum()
    else:  # last token
        pooled = h[mask.bool()][-1]
    return F.normalize(pooled.float(), dim=0)


def routing_acc(name, score_seg):
    correct = 0
    rows = []
    for q, gold in PROBES:
        scores = [score_seg(q, i) for i in range(len(SEGMENTS))]
        pick = int(max(range(len(scores)), key=lambda i: scores[i]))
        ok = (pick + 1) == gold
        correct += ok
        rows.append((q, gold, pick + 1, ok, scores))
    print(f"\n### {name}: {correct}/{len(PROBES)} routed correctly")
    for q, gold, pick, ok, scores in rows:
        sc = ", ".join(f"seg{i+1}={s:.3f}" for i, s in enumerate(scores))
        print(f"  [{'OK ' if ok else 'XX '}] gold{gold} pick{pick}  {sc}  | {q}")
    return correct


def main():
    m = D2LModel.load()

    # lexical
    idf, seg_vecs = tfidf_index(SEGMENTS)
    routing_acc("TF-IDF lexical", lambda q, i: cos_sparse(tfidf_vec(q, idf), seg_vecs[i]))

    # llm mean-pool
    seg_mean = [llm_embed(m, s, "mean") for s in SEGMENTS]
    routing_acc("LLM mean-pool", lambda q, i: float(llm_embed(m, q, "mean") @ seg_mean[i]))

    # llm last-token
    seg_last = [llm_embed(m, s, "last") for s in SEGMENTS]
    routing_acc("LLM last-token", lambda q, i: float(llm_embed(m, q, "last") @ seg_last[i]))


if __name__ == "__main__":
    main()
