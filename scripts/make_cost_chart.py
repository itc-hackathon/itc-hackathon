"""Render results/cost.json -> static/cost.png (two panels).

Left:  $ per query, cold (uncached) vs warm (prefix-cache hit) — shows caching
       helps vanilla a lot but AgentHN is still cheapest.
Right: $ per 1,000 queries incl. one-time creation — shows Cartridges' offline
       training cost dominating; AgentHN lowest total.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cost.json"
OUT = ROOT / "src" / "agenthn" / "webapp" / "static" / "cost.png"

ORDER = ["vanilla", "rag", "cartridges", "napora"]
LABELS = {"vanilla": "vanilla\n(full ctx)", "rag": "RAG", "cartridges": "Cartridges\n(trained KV)",
          "napora": "AgentHN"}
COLOR = {"vanilla": "#94908a", "rag": "#c79a3a", "cartridges": "#7b61b8", "napora": "#2f6ae0"}
INK, DIM = "#26241f", "#8a877f"


def main():
    d = json.loads(DATA.read_text())
    m = d["methods"]
    x = np.arange(len(ORDER))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.edgecolor": "#d8d4cb", "axes.titlesize": 13, "axes.titleweight": "bold"})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    for ax in (axL, axR):
        ax.set_facecolor("#faf9f6"); ax.grid(True, axis="y", which="both", color="#ece9e2", lw=0.8)
        ax.set_axisbelow(True); ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([LABELS[k] for k in ORDER], fontsize=10)

    # left: $/query cold vs warm (milli-dollars)
    cold = [m[k]["query_cold"] * 1000 for k in ORDER]
    warm = [m[k]["query_warm"] * 1000 for k in ORDER]
    w = 0.38
    axL.bar(x - w / 2, cold, w, color=[COLOR[k] for k in ORDER], alpha=0.45, label="cold (uncached)")
    axL.bar(x + w / 2, warm, w, color=[COLOR[k] for k in ORDER], label="warm (prefix-cache hit)")
    for xi, k in enumerate(ORDER):
        axL.text(xi + w / 2, m[k]["query_warm"] * 1000 * 1.15, f"{m[k]['query_warm']*1000:.2f}",
                 ha="center", fontsize=8.5, color=INK)
    axL.set_ylabel("milli-$ per query (log)", color=DIM)
    axL.set_title("Cost per query — caching helps, AgentHN still cheapest", color=INK)
    axL.annotate("prefix caching\n10× cheaper", xy=(0 + w / 2, warm[0]), xytext=(0.55, cold[0] * 0.7),
                 fontsize=9, color="#6b6862", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#6b6862", lw=1.2))
    axL.legend(loc="upper right", fontsize=9, facecolor="white", edgecolor="#e7e4dd")

    # right: total $ per 1000 queries incl creation
    tot = [m[k]["total_warm"] for k in ORDER]
    bars = axR.bar(x, tot, 0.6, color=[COLOR[k] for k in ORDER])
    for xi, k in enumerate(ORDER):
        axR.text(xi, tot[xi] * 1.15, f"${m[k]['total_warm']:.2f}", ha="center", fontsize=9, color=INK)
    axR.set_ylabel("$ per 1,000 queries (log)", color=DIM)
    axR.set_title("Total cost incl. one-time creation", color=INK)
    axR.text(2, tot[2] * 0.35, "offline\ntraining\ndominates", ha="center", fontsize=8.5, color="#5b4a8a")
    axR.text(3, tot[3] * 0.35, "one fwd\npass", ha="center", fontsize=8.5, color="#1e50c0")

    sc = d["scenario"]
    fig.suptitle(f"Cost model — {sc['history_tokens']:,}-tok history, {sc['n_queries']:,} queries "
                 f"· frontier pricing w/ prompt caching", fontsize=12.5, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
