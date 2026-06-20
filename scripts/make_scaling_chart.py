"""Render results/scaling.json -> webapp static/scaling.png (two panels).

Left:  recall vs conversation length (AgentHN holds past the paper's ~4x-window
       single-encode limit; vanilla collapses at the 8k window).
Right: query-time prompt tokens (≈ KV-cache cost) — AgentHN flat ~8 regardless of
       horizon; vanilla pinned to the full window.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "scaling.json"
OUT = ROOT / "src" / "agenthn" / "webapp" / "static" / "scaling.png"

BLUE = "#2f6ae0"
GREY = "#94908a"
GOLD = "#c79a3a"
INK = "#26241f"
DIM = "#8a877f"


def kfmt(x, _pos):
    if x >= 1000:
        return f"{x/1000:.0f}k"
    return f"{x:.0f}"


def main() -> None:
    d = json.loads(DATA.read_text())
    rows = d["rows"]
    window = d["window"]
    paper = d["paper_limit_tokens"]
    x = [r["raw_tokens"] for r in rows]
    n_probes = rows[0]["n_probes"]
    nap_rec = [100 * r["napora_recall"] / n_probes for r in rows]
    van_rec = [100 * r["vanilla_recall"] / n_probes for r in rows]
    rag_rec = [100 * r.get("rag_recall", 0) / n_probes for r in rows]
    nap_tok = [r["napora_prompt_tokens"] for r in rows]
    van_tok = [r["vanilla_prompt_tokens"] for r in rows]
    rag_tok = [r.get("rag_prompt_tokens", 0) for r in rows]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.edgecolor": "#d8d4cb", "axes.linewidth": 1,
        "axes.titlesize": 13, "axes.titleweight": "bold",
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    fig.patch.set_facecolor("white")

    for ax in (axL, axR):
        ax.set_facecolor("#faf9f6")
        ax.set_xscale("log")
        ax.grid(True, which="both", color="#ece9e2", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.set_xlabel("conversation length (tokens)", color=DIM)
        # reference lines
        ax.axvline(window, color=GREY, ls=":", lw=1.3)
        ax.axvline(paper, color=GOLD, ls=":", lw=1.3)

    # --- left: recall ---  (AgentHN and text-RAG overlap at 100% -> offset RAG for legibility)
    axL.plot(x, nap_rec, "-o", color=BLUE, lw=2.4, ms=6, label="AgentHN (weights)")
    axL.plot(x, [r - 2 for r in rag_rec], "--s", color=GOLD, lw=2.0, ms=5,
             label="text-RAG ablation (retrieve as text)")
    axL.plot(x, van_rec, "-o", color=GREY, lw=2.0, ms=5, label="Vanilla (8k window, no memory)")
    axL.set_ylim(-5, 108)
    axL.set_ylabel("needle recall (%)", color=DIM)
    axL.set_title("Recall holds as the conversation grows", color=INK)
    axL.text(window, 10, " 8k window", color=GREY, fontsize=9, rotation=90, va="bottom", ha="right")
    axL.text(paper, 10, " D2L single-encode ~4×window", color="#9a7a30", fontsize=9,
             rotation=90, va="bottom", ha="right")
    axL.text(x[1], 92, "AgentHN = text-RAG (both 6/6)", color="#6b6862", fontsize=9)
    axL.legend(loc="center left", frameon=True, fontsize=9.5, facecolor="white", edgecolor="#e7e4dd")

    # --- right: cost ---  (the ablation gap: RAG re-enters the chunk's tokens, AgentHN doesn't)
    axR.plot(x, van_tok, "-o", color=GREY, lw=2.0, ms=5, label="Vanilla prompt")
    axR.plot(x, rag_tok, "--s", color=GOLD, lw=2.2, ms=5, label="text-RAG prompt")
    axR.plot(x, nap_tok, "-o", color=BLUE, lw=2.6, ms=6, label="AgentHN prompt (weights)")
    axR.set_yscale("log")
    axR.set_ylim(5, 15000)
    axR.set_ylabel("query-time prompt tokens  (≈ KV-cache cost, log)", color=DIM)
    axR.set_title("Inference cost: AgentHN vs the real baseline (RAG)", color=INK)
    xr, ny, gy = x[-1], nap_tok[-1], rag_tok[-1]
    if ny > 0 and gy > 0:
        axR.annotate(f"{gy/ny:.0f}× less\nthan text-RAG", xy=(xr, ny * 1.05),
                     xytext=(xr * 0.42, (ny * gy) ** 0.5 * 0.9),
                     color=BLUE, fontsize=10, ha="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    axR.legend(loc="lower left", frameon=True, fontsize=9.5, facecolor="white", edgecolor="#e7e4dd")

    fig.suptitle("Repeated compaction scales long-horizon memory past the single-encode limit",
                 fontsize=14, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
