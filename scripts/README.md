# LLM-Generated CNN Training Scripts

This directory contains the executable PyTorch training scripts produced by each of the five LLMs (ChatGPT, Claude, DeepSeek, Gemini, Qwen) under each of the three prompt tiers (`tier1` Basic, `tier2` Structured, `tier3` Detailed) for both tasks (`task_a_binary` and `task_b_3class`), for a total of 30 scripts.

## Layout
scripts/
├── task_a_binary/
│   ├── tier1/   chatgpt.py, claude.py, deepseek.py, gemini.py [bug], qwen.py [bug]
│   ├── tier2/   chatgpt.py, claude.py, deepseek.py, gemini.py [bug], qwen.py
│   └── tier3/   chatgpt.py, claude.py, deepseek.py, gemini.py, qwen.py
└── task_b_3class/
├── tier1/   chatgpt.py, claude.py, deepseek.py, gemini.py, qwen.py
├── tier2/   chatgpt.py, claude.py, deepseek.py, gemini.py, qwen.py
└── tier3/   chatgpt.py, claude.py, deepseek.py, gemini.py, qwen.py [bug]
## Important notes

**These scripts are LLM output, reproduced verbatim.** No edits were applied except, where the LLM hard-coded a random seed, that single line was parameterized to read from the environment variable `EXPERIMENT_SEED` for replication purposes.

**Four scripts contain code-level bugs** and are included unchanged for auditability and Validity-Rate analysis:

| Script                                | Bug |
|---------------------------------------|-----|
| `task_a_binary/tier1/gemini.py`       | Missing `.detach()` before `.numpy()` conversion |
| `task_a_binary/tier2/gemini.py`       | Unclosed parenthesis in a `print` statement |
| `task_a_binary/tier1/qwen.py`         | `ReduceLROnPlateau(verbose=True)` removed in PyTorch 2.2 |
| `task_b_3class/tier3/qwen.py`         | `RandomErasing` applied before `ToTensor()` in the augmentation pipeline |

These four cells contribute to the Validity Rate metric reported in the paper and are intentionally **not corrected**, since the failures themselves are a reported finding.

## How to inspect

Each script is self-contained. To examine the architectural and hyperparameter choices each LLM made:

```bash
head -100 scripts/task_a_binary/tier2/claude.py
head -100 scripts/task_b_3class/tier3/deepseek.py
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
