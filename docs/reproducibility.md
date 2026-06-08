# Reproducibility Guide

This document walks through reproducing the full experimental campaign reported in the paper.

## System requirements

- Linux x86_64 (the paper used Ubuntu Linux on a laptop with 16 GB RAM, no GPU).
- Python 3.12.
- Approximately 5 GB of free disk space.
- Approximately 12 days of wall-clock CPU time for the full campaign of 300 runs; partial runs are also supported.

## Step 1. Clone the repository and set up the environment

```bash
git clone <repository-url>
cd <repository-name>

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the installation:

```bash
python3 -c "import torch, torchvision, sklearn, pandas, scipy, tqdm, psutil; print('OK')"
```

## Step 2. Download the RoCoLe dataset

The RoCoLe dataset is publicly available from Mendeley Data: <https://data.mendeley.com/datasets/c5yvn32dzg/2>.

Extract it under `./rocole/` so that the structure matches:

```
rocole/
├── Photos/
│   ├── C1P1H1.jpg
│   ├── C1P1H2.jpg
│   └── ... (1560 images total)
└── Annotations/
    └── RoCoLe-classes.xlsx
```

Verify the count:

```bash
ls rocole/Photos/*.jpg | wc -l    # expected: 1560
```

## Step 3. Build the data splits

```bash
python3 data_preparation/prepare_rocole.py
```

Expected output (excerpt):

```
[INFO] Images found in rocole/Photos: 1560
=== Building task_a_binary ===
Counts per class (raw):
  healthy:   791
  unhealthy: 769
  train: 1092 images
  val:    234 images
  test:   234 images
=== Building task_b_3class ===
Counts per class (raw):
  coffee_leaf_rust:  602
  healthy:           791
  red_spider_mite:   167
  train: 1090 images
  val:    234 images
  test:   236 images
[OK] Splits generated at .../data/splits
```

This creates `data/splits/task_a_binary/` and `data/splits/task_b_3class/`, each with `train/`, `val/` and `test/` subdirectories per class. The random split uses seed 42 by default.

## Step 4. Run the experimental matrix

The full matrix is `2 tasks × 3 tiers × 5 LLMs × 10 seeds = 300 runs`, of which 12 are skipped (known-bug cells × 10 seeds) and 4 are expected to fail at the code level, leaving 260 successful runs.

```bash
# Recommended: prevent the laptop from suspending during the long campaign
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# Launch the full campaign in the background, surviving terminal close
nohup python3 experiment_runner/run_experiment_matrix.py --seeds 10 \
    > campaign.log 2>&1 &
```

Each run produces a JSON file under `logs/`. The orchestrator skips cells already completed, so it can be safely restarted after an interruption.

### Monitoring the campaign

```bash
# Count completed runs
ls logs/*.json | wc -l

# Tail the orchestrator log
tail -20 campaign.log

# Detailed per-configuration progress
python3 -c "
import os, json
from itertools import product
LLMS = ['chatgpt','claude','deepseek','gemini','qwen']
TIERS = ['tier1','tier2','tier3']
TASKS = ['task_a_binary','task_b_3class']
KNOWN = {('task_a_binary','tier1','gemini'),('task_a_binary','tier2','gemini'),
         ('task_a_binary','tier1','qwen'),('task_b_3class','tier3','qwen')}
for task, tier, llm in product(TASKS, TIERS, LLMS):
    if (task,tier,llm) in KNOWN: continue
    ok = 0
    for seed in range(10):
        f = f'logs/{task}_{tier}_{llm}_seed{seed:02d}.json'
        if os.path.exists(f):
            try:
                d = json.load(open(f))
                if d.get('syntactic_ok'):
                    ok += 1
            except: pass
    bar = '✓' * ok + '·' * (10 - ok)
    print(f'  {task}/{tier}/{llm}: {ok:2d}/10  [{bar}]')
"
```

### Running a subset

```bash
# Only one task
python3 experiment_runner/run_experiment_matrix.py --seeds 10 \
    --tasks task_a_binary

# Only one LLM at one tier
python3 experiment_runner/run_experiment_matrix.py --seeds 10 \
    --llms claude --tiers tier2

# Fewer seeds (e.g., to verify the pipeline before the full campaign)
python3 experiment_runner/run_experiment_matrix.py --seeds 2
```

## Step 5. Analyse the results

After the campaign:

```bash
# Consolidate per-run JSONs into a single CSV
python3 analysis/consolidate_results.py

# Compute paired t-tests, Cohen's d, minority-recall summary
python3 analysis/compute_statistics.py

# Recreate every figure in the paper
python3 analysis/make_figures.py
```

Outputs:

```
results/raw_results.csv               260 successful + ~40 failed rows
results/summary_by_config.csv         26 valid (LLM, tier, task) cells, mean ± SD
results/tier_comparison_ttests.csv    All pairwise tier contrasts with significance
results/minority_recall_summary.csv   Per-cell minority-class behavior on Task B
docs/figures/fig1_tier_evolution.png
docs/figures/fig2_boxplots.png
docs/figures/fig3_perclass_recall.png
docs/figures/fig4_perfgain.png
docs/figures/fig5_compute_time.png
```

## Known-bug cells

Four (task, tier, LLM) cells fail deterministically at the code level. These failures are reported in the paper's Validity Rate analysis and are intentionally not corrected:

| Cell                          | Cause |
|-------------------------------|-------|
| Task A, Tier 1, Gemini        | Missing `.detach()` before NumPy conversion |
| Task A, Tier 2, Gemini        | Unclosed parenthesis in a `print` statement |
| Task A, Tier 1, Qwen          | `ReduceLROnPlateau(verbose=True)` removed in PyTorch 2.2 |
| Task B, Tier 3, Qwen          | `RandomErasing` applied to PIL image instead of tensor |

The orchestrator skips these cells by default (`--skip-known-bugs`, enabled). The four bugs are visible in the generated scripts under `scripts/` and can be inspected directly.

## Notes on randomness

Each script reads the random seed from the environment variable `EXPERIMENT_SEED` set by the experiment runner. The seed is propagated through Python's `random`, NumPy, and PyTorch (including the data-loader workers). Per-seed reproducibility on the same hardware and Python/PyTorch versions is bit-for-bit; small numerical differences may appear across different hardware or library versions.

## Disk usage

Approximate disk usage after a full campaign:

```
data/                ~3 GB    (RoCoLe images replicated across the two task layouts)
logs/                ~30 MB   (300 JSON files)
results/             ~1 MB    (CSVs)
docs/figures/        ~500 KB  (PNG figures)
```

## Troubleshooting

**Some scripts fail with `ModuleNotFoundError`.** Reinstall dependencies with `pip install -r requirements.txt` and verify with `python3 -c "import torch, tqdm"`.

**A run finishes too quickly.** A run shorter than approximately 5 minutes most likely failed before training. Inspect the corresponding `logs/*.json` file for the `stdout_tail` field, which contains the captured exception or error message.

**The system suspends during the campaign.** On laptops, ensure the suspension targets are masked (see Step 4) and that the configuration in `/etc/systemd/logind.conf` does not suspend on lid close. Restarting the campaign after suspension is safe; the orchestrator resumes from where it stopped.

**Disk fills up.** The dominant consumer is `data/`. If disk is constrained, you can use symlinks instead of copies by modifying `prepare_rocole.py` to use `os.symlink` instead of `shutil.copy2`, at the cost of slower data loading.
