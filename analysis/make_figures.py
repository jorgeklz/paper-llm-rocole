"""
make_figures.py
===============

Regenerates every figure reported in the paper, from raw_results.csv and
the consolidated summary.

Inputs:
    results/raw_results.csv
    results/summary_by_config.csv

Outputs (under docs/figures/):
    fig1_tier_evolution.png    Trajectories acc/F1 vs tier (both tasks)
    fig2_boxplots.png          Per-seed distributions per (LLM, tier)
    fig3_perclass_recall.png   Per-class recall on Task B
    fig4_perfgain.png          Performance Gain Delta relative to T1 baseline
    fig5_compute_time.png      Wall-clock training time per cell

Usage:
    python3 analysis/make_figures.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Per-LLM colour scheme used consistently across all figures
COLORS = {
    "chatgpt":  "#10A37F",
    "claude":   "#D97757",
    "deepseek": "#4D6BFE",
    "gemini":   "#4285F4",
    "qwen":     "#A020F0",
}

TIER_ORDER = ["tier1", "tier2", "tier3"]
TIER_LABELS = ["T1\n(Basic)", "T2\n(Structured)", "T3\n(Detailed)"]


# ---------------------------------------------------------------------------
# Figure 1 — trajectories acc/F1 by tier
# ---------------------------------------------------------------------------
def fig1_tier_evolution(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    pairs = [("task_a_binary", "Task A (binary)"),
             ("task_b_3class", "Task B (3-class, imbalanced)")]

    for col, (task, label) in enumerate(pairs):
        for row, metric in enumerate(["accuracy", "f1"]):
            ax = axes[row][col]
            sub = summary[summary.task == task]
            for llm in COLORS:
                sl = sub[sub.llm == llm].set_index("tier").reindex(TIER_ORDER)
                y = sl[f"{metric}_mean"].values
                yerr = sl[f"{metric}_std"].values
                ax.errorbar(range(3), y, yerr=yerr, marker="o", capsize=4,
                            label=llm, color=COLORS[llm], linewidth=2,
                            markersize=7)
            ax.set_xticks(range(3))
            ax.set_xticklabels(TIER_LABELS)
            ax.set_ylabel("Macro-F1" if metric == "f1" else "Accuracy")
            ax.set_title(f"{label} — "
                         f"{'Macro-F1' if metric == 'f1' else 'Accuracy'}")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Effect of prompt tier on classification performance "
                 "(mean ± SD over 10 seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figure 2 — boxplots of macro-F1 per cell
# ---------------------------------------------------------------------------
def fig2_boxplots(raw: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, task in zip(axes,
                        ["task_a_binary", "task_b_3class"]):
        sub = raw[(raw.task == task) & (raw.syntactic_ok == True)].copy()  # noqa
        data, labels, colors, positions = [], [], [], []
        pos = 0
        for tier in TIER_ORDER:
            for llm in COLORS:
                g = sub[(sub.tier == tier) & (sub.llm == llm)]
                if len(g):
                    data.append(g["f1"].values)
                    labels.append(f"{llm[:4]}\nT{tier[-1]}")
                    colors.append(COLORS[llm])
                    positions.append(pos)
                pos += 1
            pos += 0.7

        bp = ax.boxplot(data, positions=positions, widths=0.7,
                        patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": 6})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Macro-F1 (n=10 seeds)")
        ax.set_title("Task A (binary)" if task == "task_a_binary"
                     else "Task B (3-class, imbalanced)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Distribution of Macro-F1 over 10 seeds per (LLM, tier) "
                 "configuration",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figure 3 — per-class recall on Task B
# ---------------------------------------------------------------------------
def fig3_perclass(raw: pd.DataFrame, out: Path) -> None:
    needed = {"recall_healthy", "recall_red_spider_mite", "recall_coffee_leaf_rust"}
    if not needed.issubset(raw.columns):
        print("[SKIP] fig3: per-class recall columns missing.")
        return

    b = raw[(raw.task == "task_b_3class") & (raw.syntactic_ok == True) &  # noqa
            (raw["recall_red_spider_mite"].notna())].copy()

    agg = b.groupby(["tier", "llm"]).agg({
        "recall_healthy":         ["mean", "std"],
        "recall_red_spider_mite": ["mean", "std"],
        "recall_coffee_leaf_rust": ["mean", "std"],
    }).reset_index()
    agg.columns = ["tier", "llm",
                   "rh_mean", "rh_std",
                   "rm_mean", "rm_std",
                   "rr_mean", "rr_std"]
    agg["tier_n"] = agg["tier"].map({"tier1": 1, "tier2": 2, "tier3": 3})
    agg = agg.sort_values(["tier_n", "llm"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(agg))
    w = 0.27
    ax.bar(x - w, agg["rh_mean"], w, yerr=agg["rh_std"],
           label="healthy (50.7%)", color="#2ca02c", capsize=2)
    ax.bar(x,     agg["rm_mean"], w, yerr=agg["rm_std"],
           label="red_spider_mite (10.7%, minority)",
           color="#d62728", capsize=2)
    ax.bar(x + w, agg["rr_mean"], w, yerr=agg["rr_std"],
           label="coffee_leaf_rust (38.6%)", color="#ff7f0e", capsize=2)

    labels = [f"{r.llm}\n{r.tier[-1]}" for _, r in agg.iterrows()]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Per-class recall (mean ± SD, n=10)")
    ax.set_xlabel("LLM and prompt tier")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_title("Per-class recall on Task B (n=10): minority class behavior "
                 "across LLMs and prompt tiers",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figure 4 — Performance Gain Delta vs T1 baseline
# ---------------------------------------------------------------------------
def fig4_perfgain(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, task in zip(axes, ["task_a_binary", "task_b_3class"]):
        sub = summary[summary.task == task].copy()
        baseline = sub[sub.tier == "tier1"]["f1_mean"].mean()
        sub["delta"] = sub["f1_mean"] - baseline
        sub["tier_n"] = sub["tier"].map({"tier1": 1, "tier2": 2, "tier3": 3})
        sub = sub.sort_values(["tier_n", "llm"]).reset_index(drop=True)

        x = np.arange(len(sub))
        colors = [COLORS[l] for l in sub["llm"]]
        ax.bar(x, sub["delta"], yerr=sub["f1_std"], capsize=3,
               color=colors, alpha=0.75, edgecolor="black")
        labels = [f"{r.llm[:4]}\nT{r.tier[-1]}" for _, r in sub.iterrows()]
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel(f"Δ Macro-F1 vs T1 baseline ({baseline:.3f})")
        ax.set_title("Task A (binary)" if task == "task_a_binary"
                     else "Task B (3-class, imbalanced)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Performance Gain (Δ) relative to mean Tier-1 macro-F1 baseline",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figure 5 — Wall-clock training time per cell
# ---------------------------------------------------------------------------
def fig5_compute_time(summary: pd.DataFrame, out: Path) -> None:
    if "wallclock_seconds_mean" not in summary.columns:
        print("[SKIP] fig5: wallclock not aggregated.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, task in zip(axes, ["task_a_binary", "task_b_3class"]):
        sub = summary[summary.task == task].copy()
        sub["tier_n"] = sub["tier"].map({"tier1": 1, "tier2": 2, "tier3": 3})
        sub["time_min"] = sub["wallclock_seconds_mean"] / 60
        sub = sub.sort_values(["tier_n", "llm"]).reset_index(drop=True)

        x = np.arange(len(sub))
        colors = [COLORS[l] for l in sub["llm"]]
        ax.bar(x, sub["time_min"], color=colors, alpha=0.75, edgecolor="black")
        labels = [f"{r.llm[:4]}\nT{r.tier[-1]}" for _, r in sub.iterrows()]
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Wall-clock training time (min/seed)")
        ax.set_title("Task A (binary)" if task == "task_a_binary"
                     else "Task B (3-class, imbalanced)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Wall-clock training time per seed (CPU-only)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("results/raw_results.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("results/summary_by_config.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    if not args.raw.is_file() or not args.summary.is_file():
        raise SystemExit("Raw and summary CSVs not found; "
                         "run consolidate_results.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw)
    summary = pd.read_csv(args.summary)

    fig1_tier_evolution(summary, args.out_dir / "fig1_tier_evolution.png")
    fig2_boxplots(raw,           args.out_dir / "fig2_boxplots.png")
    fig3_perclass(raw,           args.out_dir / "fig3_perclass_recall.png")
    fig4_perfgain(summary,       args.out_dir / "fig4_perfgain.png")
    fig5_compute_time(summary,   args.out_dir / "fig5_compute_time.png")


if __name__ == "__main__":
    main()
