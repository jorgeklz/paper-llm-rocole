# LLM-Generated CNN Training Scripts

This directory contains the executable PyTorch training scripts produced by each of the five LLMs (ChatGPT, Claude, DeepSeek, Gemini, Qwen) under each of the three prompt tiers (`tier1` Basic, `tier2` Structured, `tier3` Detailed) for both tasks (`task_a_binary` and `task_b_3class`).

## Layout

```
scripts/
├── task_a_binary/
│   ├── tier1/   chatgpt.py, claude.py, deepseek.py  (Gemini and Qwen failed: see Validity Rate)
│   ├── tier2/   chatgpt.py, claude.py, deepseek.py, qwen.py  (Gemini failed)
│   └── tier3/   chatgpt.py, claude.py, deepseek.py, gemini.py, qwen.py
└── task_b_3class/
    ├── tier1/   chatgpt.py, claude.py, deepseek.py, gemini.py, qwen.py
    ├── tier2/   chatgpt.py, claude.py, deepseek.py, gemini.py, qwen.py
    └── tier3/   chatgpt.py, claude.py, deepseek.py, gemini.py  (Qwen failed)
```

## Important notes

**These scripts are LLM output, reproduced verbatim** with one minimal modification: the hard-coded random seed has been parameterized to read from the environment variable `EXPERIMENT_SEED` for replication purposes. No other line of code has been changed.

The four scripts referenced in the paper as Validity Rate failures (Gemini at Task A Tier 1 and Tier 2; Qwen at Task A Tier 1; Qwen at Task B Tier 3) are intentionally **not included** in this repository because the LLMs did not produce executable code in those cases. Instead, the verbatim erroneous outputs (with their specific syntax or runtime errors) are documented in the paper, Section "Validity Rate and Code-Level Failures".

## How to inspect

Each script is self-contained. To examine the architectural and hyperparameter choices each LLM made:

```bash
head -100 scripts/task_a_binary/tier2/claude.py    # Claude's Tier 2 design on Task A
head -100 scripts/task_b_3class/tier3/deepseek.py  # DeepSeek's Tier 3 design on Task B
                                                   # (this is the configuration that
                                                   # collapses to predicting only the
                                                   # minority class)
```

The scripts vary substantially in length, structure, and architectural choices, which is itself one of the findings of the paper.

## How they are executed

Each script is launched in an isolated working directory by `experiment_runner/run_one_experiment.py`, with `EXPERIMENT_SEED` set in the environment and `data/splits/{task}/` symlinked at the expected relative path. See `docs/reproducibility.md` for the full execution protocol.

## Provenance

The exact LLM versions used are listed in the paper:

| LLM      | Platform    | Model version    |
|----------|-------------|------------------|
| ChatGPT  | OpenAI      | GPT-4o           |
| Claude   | Anthropic   | Claude Sonnet 4.6|
| DeepSeek | Open source | DeepSeek-V3      |
| Gemini   | Google      | Gemini 3         |
| Qwen     | Open source | Qwen3-Coder      |

Re-issuing the prompts to later model versions may produce different scripts.
