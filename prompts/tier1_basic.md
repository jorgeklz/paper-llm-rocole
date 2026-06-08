# Tier 1: Basic Prompt

The Basic Prompt grants the LLM maximal autonomy. It receives only a high-level task description and independently determines every architectural and training decision, including backbone choice, learning-rate schedule, augmentation and class-imbalance handling.

## Prompt template

The same template is issued with two variants, one per task. Substitute `{N_CLASSES}` and `{IMBALANCE_NOTE}` as indicated.

### Task A (binary, balanced)

```
Act as a Deep Learning Researcher. Suggest a standard and effective way to
build and train a deep CNN model in PyTorch to classify robusta coffee leaf
images into two classes: healthy and unhealthy. The dataset is moderately
sized (around 1560 images) and approximately balanced. Use a progressive
training strategy with layer freezing and unfreezing.
```

### Task B (three-class, imbalanced)

```
Act as a Deep Learning Researcher. Suggest a standard and effective way to
build and train a deep CNN model in PyTorch to classify robusta coffee leaf
images into three classes: healthy, red spider mite presence, and coffee
leaf rust. The dataset is moderately sized (around 1560 images) and
moderately imbalanced. Use a progressive training strategy with layer
freezing and unfreezing.
```

## Protocol

- Issue at the start of a fresh conversation with no in-context examples or memory carryover.
- Save the LLM's complete response verbatim as a `.py` file under `scripts/{task}/tier1/{llm}.py`.
- Do not edit the script except for the single semilla-parameterization patch documented in the experiment runner (substitution of any hard-coded random seed with `os.environ["EXPERIMENT_SEED"]`).
