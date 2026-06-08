# Tier 2: Structured Prompt

The Structured Prompt fixes the training architecture and asks the LLM to recommend an optimal hyperparameter vector.

## Prompt template

```
Act as a Deep Learning optimizer. I have a custom deep CNN implemented in
PyTorch for the classification of robusta coffee leaf images. The
architecture uses a two-phase training strategy: Phase 1 trains only the
classification head with frozen feature extractors, and Phase 2 fine-tunes
the entire network. Your task is to recommend the optimal hyperparameter
vector theta = [d, eta_1, eta_2, b, u, w], where:
  - d is the dropout rate,
  - eta_1 is the learning rate for Phase 1,
  - eta_2 is the learning rate for Phase 2,
  - b is the batch size,
  - u is the number of units in the dense classification layer,
  - w is a class-imbalance handling strategy among
        {class weights, focal loss, oversampling, none}.

Propose specific values based on best practices, justify each choice, and
recommend a learning-rate scheduler (CosineAnnealingLR) and an early-stopping
criterion. The task is {binary | three-class} robusta coffee leaf
classification on a moderately sized, {balanced | imbalanced} dataset.
```

## Variants

| Task   | Substitution               |
|--------|----------------------------|
| Task A | binary, balanced           |
| Task B | three-class, imbalanced    |

## Protocol

- Issue at the start of a fresh conversation with no in-context examples or memory carryover.
- The LLM is expected to produce both the hyperparameter recommendations (with justifications) and the executable PyTorch training script.
- Save the LLM's complete response verbatim as `scripts/{task}/tier2/{llm}.py`.
