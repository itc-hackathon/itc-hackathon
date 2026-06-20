"""Render physics_benchmark_results.json -> accuracy_by_condition.png and
context_tokens.png, styled to match scripts/make_scaling_chart.py's theme.

Reads the existing benchmark JSON only -- does not re-run the model or
recompute any results.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "physics_benchmark_results.json"

YELLOW = "#c98a13"
PURPLE = "#7c3fa1"
BLUE = "#2f5fd6"
INK = "#26241f"
DIM = "#8a877f"


def main():
    d = json.loads(DATA.read_text())
    rows = d["rows"]
    n = len(rows)
    conditions = ["base", "in_context", "adapter"]
    labels = ["Base\n(cold)", "In-Context\n(doc in prompt)", "Adapter\n(D2L internalized)"]
    colors = [YELLOW, PURPLE, BLUE]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.edgecolor": "#d8d4cb", "axes.linewidth": 1,
        "axes.titlesize": 13, "axes.titleweight": "bold",
    })

    # --- accuracy_by_condition.png: all three columns, same order ---
    pass_rates = [100 * sum(r[f"{c}_pass"] for r in rows) / n for c in conditions]
    passed_n = [sum(r[f"{c}_pass"] for r in rows) for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#faf9f6")
    ax.grid(True, axis="y", color="#ece9e2", linewidth=0.8)
    ax.set_axisbelow(True)
    bars = ax.bar(labels, pass_rates, color=colors, width=0.6, zorder=3)
    ax.set_ylabel("pass rate (%)", color=DIM)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_title("Physics word-problem accuracy", color=INK)
    for bar, rate, np_ in zip(bars, pass_rates, passed_n):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2.5, f"{rate:.0f}%\n({np_}/{n})",
                ha="center", va="bottom", fontsize=10, color=INK)

    # trend line: origin -> base -> in-context -> adapter
    line_x = [-0.5, 0, 1, 2]
    line_y = [0] + pass_rates
    ax.plot(line_x, line_y, color=INK, linewidth=1.8, marker="o",
             markersize=5, markerfacecolor=INK, zorder=4)
    fig.suptitle("Physics skill acquisition: base vs. in-context vs. D2L adapter",
                 fontsize=14, fontweight="bold", color=INK, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.savefig(HERE / "accuracy_by_condition.png", bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)

    # --- context_tokens.png: only in_context and adapter columns ---
    ctx_conditions = ["in_context", "adapter"]
    ctx_labels = ["In-Context\n(doc in prompt)", "Adapter\n(D2L internalized)"]
    ctx_colors = [PURPLE, BLUE]
    avg_input_tokens = [np.mean([r[f"{c}_input_tokens"] for r in rows]) for c in ctx_conditions]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#faf9f6")
    ax.grid(True, axis="y", color="#ece9e2", linewidth=0.8)
    ax.set_axisbelow(True)
    bars = ax.bar(ctx_labels, avg_input_tokens, color=ctx_colors, width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("avg. input (context) tokens per question  (log)", color=DIM)
    ax.set_title("Context size consumed per question", color=INK)
    for bar, val in zip(bars, avg_input_tokens):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.0f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    ratio = avg_input_tokens[0] / avg_input_tokens[1]
    ax.annotate(f"{ratio:.0f}× less\nthan in-context", xy=(1, avg_input_tokens[1] * 1.3),
                xytext=(0.97, 0.42), textcoords="axes fraction",
                color=BLUE, fontsize=10, ha="right", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    fig.suptitle("D2L adapter avoids re-paying the doc's tokens every query",
                 fontsize=13, fontweight="bold", color=INK, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.savefig(HERE / "context_tokens.png", bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)

    print("wrote accuracy_by_condition.png, context_tokens.png")


if __name__ == "__main__":
    main()
