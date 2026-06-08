"""
compute_statistics.py
=====================

Within-LLM paired statistical comparisons across prompt tiers:
    - paired t-tests on (T1 vs T2, T1 vs T3, T2 vs T3),
    - Bonferroni correction over the four primary metrics
      (accuracy, macro-F1, balanced accuracy, AUC):
      alpha_adj = 0.05 / 4 = 0.0125,
    - Cohen's d on paired differences with the conventional
      small/medium/large labels.

Also computes a minority-recall summary for Task B (per-LLM, per-tier mean,
std, min, and the count of seeds for which the minority class collapses).

Inputs:
    results/raw_results.csv   (produced by consolidate_results.py)

Outputs:
    results/tier_comparison_ttests.csv
    results/minority_recall_summary.csv

Usage:
    python3 analysis/compute_statistics.py
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = ["accuracy", "f1", "balanced_accuracy", "auc"]
ALPHA = 0.05
ALPHA_ADJ = ALPHA / len(PRIMARY_METRICS)   # = 0.0125 with Bonferroni over 4


def cohens_d_paired(x, y):
    """Cohen's d on paired differences."""
    diff = np.asarray(x) - np.asarray(y)
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return diff.mean() / sd


def effect_label(d):
    a = abs(d)
    if np.isnan(a):
        return "n/a"
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def pairwise_tier_tests(df, out_csv):
    rows = []
    valid = df[df.syntactic_ok == True].copy()  # noqa: E712

    for task in sorted(valid.task.dropna().unique()):
        for llm in sorted(valid.llm.dropna().unique()):
            sub = valid[(valid.task == task) & (valid.llm == llm)]
            tiers_present = sorted(sub.tier.unique())
            for t_a, t_b in combinations(tiers_present, 2):
                ga = sub[sub.tier == t_a].sort_values("seed")
                gb = sub[sub.tier == t_b].sort_values("seed")
                common = set(ga.seed) & set(gb.seed)
                if len(common) < 3:
                    continue
                ga = ga[ga.seed.isin(common)].sort_values("seed")
                gb = gb[gb.seed.isin(common)].sort_values("seed")
                n = len(common)

                for metric in PRIMARY_METRICS:
                    if metric not in ga.columns:
                        continue
                    x = ga[metric].dropna().to_numpy()
                    y = gb[metric].dropna().to_numpy()
                    if len(x) != len(y) or len(x) < 3:
                        continue
                    if np.var(x - y, ddof=1) == 0:
                        # constant difference: t-test undefined but effect known
                        p = 0.0 if (x.mean() != y.mean()) else 1.0
                        t = np.inf if (x.mean() != y.mean()) else 0.0
                    else:
                        t, p = stats.ttest_rel(x, y)
                    d = cohens_d_paired(x, y)
                    rows.append({
                        "task": task,
                        "llm": llm,
                        "comparison": f"{t_a}_vs_{t_b}",
                        "metric": metric,
                        "n_seeds": n,
                        "mean_1": float(np.mean(x)),
                        "mean_2": float(np.mean(y)),
                        "diff": float(np.mean(x) - np.mean(y)),
                        "t_stat": float(t),
                        "p_value": float(p),
                        "sig_uncorrected": bool(p < ALPHA),
                        "sig_bonferroni": bool(p < ALPHA_ADJ),
                        "cohens_d": float(d) if not np.isnan(d) else None,
                        "effect": effect_label(d),
                    })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"[OK] {out_csv}  ({len(out)} comparisons; alpha_adj={ALPHA_ADJ})")
    return out


def minority_summary(df, out_csv):
    """Per-(tier, llm) summary of minority-class recall on Task B."""
    valid = df[(df.task == "task_b_3class") & (df.syntactic_ok == True)].copy()  # noqa
    if "recall_red_spider_mite" not in valid.columns:
        print("[WARN] recall_red_spider_mite not in raw_results; "
              "minority summary skipped.")
        return pd.DataFrame()

    rows = []
    for (tier, llm), g in valid.groupby(["tier", "llm"]):
        vals = g["recall_red_spider_mite"].dropna()
        if vals.empty:
            continue
        rows.append({
            "tier": tier,
            "llm": llm,
            "n": int(len(vals)),
            "minority_recall_mean": float(vals.mean()),
            "minority_recall_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "minority_recall_min": float(vals.min()),
            "collapse_count": int((vals < 0.05).sum()),
        })
    out = pd.DataFrame(rows).sort_values(["tier", "llm"])
    out.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"[OK] {out_csv}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("results/raw_results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"{args.csv} not found. Run consolidate_results.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    pairwise_tier_tests(df, args.out_dir / "tier_comparison_ttests.csv")
    minority_summary(df, args.out_dir / "minority_recall_summary.csv")


if __name__ == "__main__":
    main()
