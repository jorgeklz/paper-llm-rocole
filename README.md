# Prompt Engineering for LLM-Generated CNNs in Coffee Leaf Disease Recognition

This repository accompanies the paper *"Prompt Engineering for LLM-Generated CNNs in Coffee Leaf Disease Recognition"*. It contains all the code, prompts and instructions needed to reproduce the experimental campaign: five LLMs (ChatGPT, Claude, DeepSeek, Gemini, Qwen) generating PyTorch CNN training pipelines under three prompt tiers, evaluated on two tasks defined over the public RoCoLe dataset, over 10 random seeds per cell.

The campaign produced **260 successful training runs out of 300 attempted**, with the remaining 4 cells failing deterministically at the code level (these failures are themselves a reported finding of the paper).

## Repository layout

```
.
├── README.md                       This file
├── LICENSE                         MIT license for code; data follows RoCoLe CC-BY-4.0
├── requirements.txt                Pinned Python dependencies (Linux CPU)
│
├── prompts/                        The three prompt templates submitted to each LLM
│   ├── tier1_basic.md
│   ├── tier2_structured.md
│   └── tier3_detailed.md
│
├── data_preparation/
│   └── prepare_rocole.py           Build train/val/test splits from RoCoLe original
│
├── scripts/                        LLM-generated training scripts (semilla parameterizada)
│   ├── task_a_binary/
│   │   ├── tier1/  {chatgpt,claude,deepseek}.py
│   │   ├── tier2/  {chatgpt,claude,deepseek,qwen}.py
│   │   └── tier3/  {chatgpt,claude,deepseek,gemini,qwen}.py
│   └── task_b_3class/
│       ├── tier1/  {chatgpt,claude,deepseek,gemini,qwen}.py
│       ├── tier2/  {chatgpt,claude,deepseek,gemini,qwen}.py
│       └── tier3/  {chatgpt,claude,deepseek,gemini}.py
│
├── experiment_runner/              Orchestrator that runs the full matrix
│   ├── run_one_experiment.py       Single-cell runner with isolation and logging
│   └── run_experiment_matrix.py    Sweep over (LLM, tier, task, seed)
│
├── analysis/                       Statistical analysis and figure generation
│   ├── consolidate_results.py      Merge all per-run JSONs into a single CSV
│   ├── compute_statistics.py       Paired t-tests, Cohen's d, summary tables
│   └── make_figures.py             Recreate every figure in the paper
│
└── docs/
    ├── reproducibility.md          Step-by-step guide to reproduce the campaign
    └── figures/                    Final paper figures (PNG)
```

## Quickstart

### 1. Get the RoCoLe dataset

The RoCoLe dataset (Parraga-Alava et al. 2019) is publicly available on Mendeley Data: <https://data.mendeley.com/datasets/c5yvn32dzg/2>. Download it and place it under `./rocole/` with the following structure:

```
rocole/
├── Photos/                         1560 .jpg images
└── Annotations/
    └── RoCoLe-classes.xlsx
```

### 2. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build the data splits

```bash
python3 data_preparation/prepare_rocole.py
```

This creates `data/splits/task_a_binary/` and `data/splits/task_b_3class/`, each with `train/`, `val/` and `test/` subdirectories under the corresponding class folders.

### 4. Run the experimental matrix

The full campaign comprises 30 (LLM, tier, task) cells × 10 random seeds = 300 runs. With CPU-only training averaging ~60 minutes per run, the full campaign takes approximately 12 days of continuous compute.

```bash
python3 experiment_runner/run_experiment_matrix.py --seeds 10
```

Each run produces a JSON file in `logs/` with the metrics, the captured stdout and resource-usage data. The four configurations that are known to fail at the code level (Gemini-T1 and Gemini-T2 on Task A, Qwen-T1 on Task A, Qwen-T3 on Task B) are documented in the paper and are intentionally not corrected.

### 5. Analyse the results

```bash
python3 analysis/consolidate_results.py
python3 analysis/compute_statistics.py
python3 analysis/make_figures.py
```

These produce `results/raw_results.csv`, `results/summary_by_config.csv`, `results/tier_comparison_ttests.csv`, `results/minority_recall_summary.csv` and all paper figures under `docs/figures/`.

## What is **not** in this repository

This repository contains only the code that was actually used to produce the published results. Specifically excluded are:

- Earlier prototypes, exploratory notebooks and ad-hoc analyses.
- Intermediate or pilot-phase results superseded by the final multi-seed campaign.
- Patches applied during debugging that were rolled into the final scripts.
- Per-machine environment artifacts (virtual environments, cache files, OS-specific scripts).

## Citation

If you use this code or build on this study, please cite:

```bibtex
@article{anonymous2026prompt,
  title={Prompt Engineering for LLM-Generated CNNs in Coffee Leaf Disease Recognition},
  author={Anonymous},
  year={2026},
  note={Submitted for double-blind peer review}
}
```

And please also cite the RoCoLe dataset:

```bibtex
@article{parraga2019rocole,
  title={{RoCoLe}: A robusta coffee leaf images dataset for evaluation of machine learning based methods in plant diseases recognition},
  author={Parraga-Alava, Jorge and Cusme, Kevin and Loor, Angel and Santander, Esneider},
  journal={Data in Brief},
  volume={25},
  pages={104414},
  year={2019}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The RoCoLe dataset is licensed separately under Creative Commons CC-BY-4.0.

## Contact

For questions about reproducing the experiments, please open an issue. Author identities are intentionally omitted from this repository while the paper is under double-blind review.
