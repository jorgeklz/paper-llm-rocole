"""
consolidate_results.py
======================

Reads all per-run JSON files from logs/ and produces:
    - results/raw_results.csv:          one row per (task, tier, llm, seed)
                                        with metrics and resource usage.
    - results/summary_by_config.csv:    one row per (task, tier, llm)
                                        with mean and std over seeds.

Usage:
    python3 analysis/consolidate_results.py
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PER_CLASS_KEYS = ("recall_healthy", "recall_unhealthy",
                  "recall_red_spider_mite", "recall_coffee_leaf_rust")


def filename_meta(json_path: Path) -> dict:
    """Parse task / tier / llm / seed from filenames like
       task_a_binary_tier1_chatgpt_seed00.json."""
    m = re.match(
        r"(?P<task>task_[ab]_(?:binary|3class))_"
        r"(?P<tier>tier[123])_"
        r"(?P<llm>[a-z]+)_seed(?P<seed>\d+)\.json$",
        json_path.name,
    )
    return m.groupdict() if m else {}


def load_one(json_path: Path) -> dict:
    """Flatten one JSON record into a single dict suitable for a CSV row."""
    try:
        d = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    meta = filename_meta(json_path)
    row = {
        "task": meta.get("task"),
        "tier": meta.get("tier"),
        "llm": meta.get("llm"),
        "seed": int(meta["seed"]) if meta.get("seed") is not None else None,
        "syntactic_ok": d.get("syntactic_ok", False),
        "return_code": d.get("return_code"),
        "wallclock_seconds": d.get("wallclock_seconds"),
        "timed_out": d.get("timed_out", False),
    }
    # Metrics
    for k, v in (d.get("metrics") or {}).items():
        row[k] = v
    # Resources
    for k, v in (d.get("resources") or {}).items():
        row[f"resource_{k}"] = v
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(args.logs_dir.glob("*.json")):
        row = load_one(f)
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No JSON files in {args.logs_dir}")

    df = pd.DataFrame(rows)
    raw_csv = args.out_dir / "raw_results.csv"
    df.to_csv(raw_csv, index=False)
    print(f"[OK] {raw_csv}  ({len(df)} rows)")

    # Summary: mean and std per (task, tier, llm)
    valid = df[df.syntactic_ok == True].copy()  # noqa: E712
    if valid.empty:
        print("[WARN] No valid runs to summarize.")
        return

    metric_cols = ["accuracy", "f1", "balanced_accuracy", "auc", "loss"]
    metric_cols += [c for c in PER_CLASS_KEYS if c in valid.columns]
    metric_cols = [c for c in metric_cols if c in valid.columns]

    agg = {c: ["mean", "std", "count"] for c in metric_cols}
    agg["wallclock_seconds"] = ["mean", "std"]
    summary = valid.groupby(["task", "tier", "llm"]).agg(agg)
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()

    summary_csv = args.out_dir / "summary_by_config.csv"
    summary.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"[OK] {summary_csv}  ({len(summary)} configurations)")


if __name__ == "__main__":
    main()
