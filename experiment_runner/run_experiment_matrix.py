"""
run_experiment_matrix.py
========================

Orchestrates the full experimental matrix:
    5 LLMs x 3 tiers x 2 tasks x N seeds = up to 300 training runs.

Skips:
    - cells already completed (the corresponding JSON exists and is valid),
    - known-bug cells (documented in the paper, no point re-running),
    - missing scripts (e.g., the bug cells where no script was generated).

Each cell is delegated to experiment_runner/run_one_experiment.py.

Usage:
    python3 experiment_runner/run_experiment_matrix.py --seeds 10

    # Restrict scope
    python3 experiment_runner/run_experiment_matrix.py --seeds 10 \
        --tasks task_a_binary --tiers tier1 tier2 --llms chatgpt claude
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


LLMS = ["chatgpt", "claude", "deepseek", "gemini", "qwen"]
TIERS = ["tier1", "tier2", "tier3"]
TASKS = ["task_a_binary", "task_b_3class"]

# Cells known to fail deterministically at the code level. These failures are
# themselves reported findings of the paper (Validity Rate section).
KNOWN_BUGS = {
    ("task_a_binary", "tier1", "gemini"),
    ("task_a_binary", "tier2", "gemini"),
    ("task_a_binary", "tier1", "qwen"),
    ("task_b_3class", "tier3", "qwen"),
}


def is_already_completed(out_path: Path) -> bool:
    """A run is considered complete if the JSON exists and reports a valid run."""
    if not out_path.exists():
        return False
    try:
        d = json.loads(out_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(d.get("syntactic_ok")) and d.get("metrics", {}).get("accuracy") is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds per cell (default 10).")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--tiers", nargs="+", default=TIERS, choices=TIERS)
    parser.add_argument("--llms", nargs="+", default=LLMS, choices=LLMS)
    parser.add_argument("--scripts-root", type=Path, default=Path("scripts"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--timeout-min", type=int, default=240)
    parser.add_argument("--skip-known-bugs", action="store_true", default=True)
    args = parser.parse_args()

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    runner = Path(__file__).parent / "run_one_experiment.py"

    cells = list(product(args.tasks, args.tiers, args.llms))
    seeds = list(range(args.seeds))
    total = len(cells) * len(seeds)

    print(f"[INFO] Configurations: {len(cells)} x {len(seeds)} seeds = {total} runs")
    print(f"[INFO] Timeout per run: {args.timeout_min} min ({args.timeout_min/60:.1f} h)")

    t0 = time.time()
    ran = skipped = failed = 0
    for task, tier, llm in cells:
        if args.skip_known_bugs and (task, tier, llm) in KNOWN_BUGS:
            print(f"[SKIP] {task}/{tier}/{llm}  (known bug, see paper)")
            skipped += len(seeds)
            continue

        script = args.scripts_root / task / tier / f"{llm}.py"
        if not script.is_file():
            print(f"[SKIP] {task}/{tier}/{llm}  (script not found: {script})")
            skipped += len(seeds)
            continue

        for seed in seeds:
            out = args.logs_dir / f"{task}_{tier}_{llm}_seed{seed:02d}.json"
            if is_already_completed(out):
                continue

            cmd = [sys.executable, str(runner),
                   "--script", str(script),
                   "--task", task,
                   "--seed", str(seed),
                   "--out", str(out),
                   "--timeout-min", str(args.timeout_min)]
            result = subprocess.run(cmd)
            ran += 1
            if result.returncode != 0:
                failed += 1

    elapsed = (time.time() - t0) / 60
    print(f"\n[SUMMARY] ran={ran}  skipped={skipped}  failed={failed}  "
          f"elapsed={elapsed:.1f} min")

    # Quick consolidation of what is currently on disk
    rows = []
    for f in sorted(args.logs_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        row = {
            "task": d.get("task"),
            "seed": d.get("seed"),
            "script": d.get("script"),
            "syntactic_ok": d.get("syntactic_ok"),
            "return_code": d.get("return_code"),
            "wallclock_seconds": d.get("wallclock_seconds"),
        }
        row.update(d.get("metrics", {}))
        # Resolve tier/llm from filename
        parts = f.stem.split("_")
        # Filenames look like: task_a_binary_tier1_chatgpt_seed00
        try:
            seed_idx = next(i for i, p in enumerate(parts) if p.startswith("seed"))
            row["tier"] = parts[seed_idx - 2]
            row["llm"] = parts[seed_idx - 1]
        except (StopIteration, IndexError):
            pass
        rows.append(row)

    if rows:
        all_keys = sorted({k for r in rows for k in r.keys()})
        csv_path = args.results_dir / "raw_results.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
