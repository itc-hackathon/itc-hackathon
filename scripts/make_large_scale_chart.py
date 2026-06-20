"""Render results/large_scale.json -> static/large_scale.png.

Left:  recall vs haystack size, mean ± std across seeds (the statistical claim).
Right: query-time context tokens (log) — the NapLoRA vs text-RAG vs vanilla gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "large_scale.json"
OUT = ROOT / "src" / "agenthn" / "webapp" / "static" / "large_scale.png"
BLUE, GOLD, GREY, INK, DIM = "#2f6ae0", "#c79a3a", "#94908a", "#26241f", "#8a877f"


def kfmt(x, _):
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"


def main():
    d = json.loads(DATA.read_text())
    rows = d["rows"]
    window = d["window"]
    x = [r["size"] for r in rows]
    seeds, nf = len(d["seeds"]), d["n_facts"]

    def series(method, key):
        return [r[method][key] for r in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.edgecolor": "#d8d4cb", "axes.titlesize": 13, "axes.titleweight": "bold"})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    for ax in (axL, axR):
        ax.set_facecolor("#faf9f6"); ax.set_xscale("log")
        ax.grid(True, which="both", color="#ece9e2", lw=0.8); ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.set_xlabel("haystack size (tokens)", color=DIM)
        ax.axvline(window, color=GREY, ls=":", lw=1.3)

    # left: recall with ±std bands
    for method, color, ls, lab in (("napora", BLUE, "-", "NapLoRA (weights)"),
                                    ("rag", GOLD, "--", "text-RAG ablation"),
                                    ("vanilla", GREY, "-", "vanilla (no memory)")):
        m = series(method, "mean"); s = series(method, "std")
        axL.plot(x, m, ls, color=color, lw=2.3, marker="o", ms=5, label=lab)
        axL.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                         color=color, alpha=0.15)
    axL.set_ylim(-5, 108); axL.set_ylabel("needle recall (%)", color=DIM)
    axL.set_title(f"Recall: mean ± std over {seeds} seeds × {nf} facts", color=INK)
    axL.text(window, 8, " 8k window", color=GREY, fontsize=9, rotation=90, va="bottom", ha="right")
    axL.legend(loc="center left", frameon=True, fontsize=9.5, facecolor="white", edgecolor="#e7e4dd")

    # right: context tokens (log)
    for method, color, ls, lab in (("vanilla", GREY, "-", "vanilla"),
                                    ("rag", GOLD, "--", "text-RAG"),
                                    ("napora", BLUE, "-", "NapLoRA")):
        axR.plot(x, series(method, "ctx_tokens"), ls, color=color, lw=2.4, marker="o", ms=5, label=lab)
    axR.set_yscale("log"); axR.set_ylim(5, 15000)
    axR.set_ylabel("query context tokens (≈ KV cost, log)", color=DIM)
    axR.set_title("Query-time context", color=INK)
    ny, gy = series("napora", "ctx_tokens")[-1], series("rag", "ctx_tokens")[-1]
    if ny > 0 and gy > 0:
        axR.annotate(f"{gy/ny:.0f}× less\nthan text-RAG", xy=(x[-1], ny * 1.1),
                     xytext=(x[-1] * 0.4, (ny * gy) ** 0.5),
                     color=BLUE, fontsize=10, ha="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    axR.legend(loc="lower left", frameon=True, fontsize=9.5, facecolor="white", edgecolor="#e7e4dd")

    big = rows[-1]
    fig.suptitle(f"Statistical validation — {seeds} seeds × {nf} facts × {len(x)} sizes "
                 f"(n={big['napora']['n']} trials/cell)", fontsize=13.5, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
