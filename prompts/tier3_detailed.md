# Tier 3: Detailed Prompt

The Detailed Prompt prescribes the architecture, training schedule, augmentation and class-imbalance handling in dense technical form.

## Prompt template

```
Act as a Deep Learning Architect. Design a complete, optimized PyTorch
training script for the classification of 224 x 224 x 3 RGB robusta coffee
leaf images into {2 | 3} classes ({healthy, unhealthy | healthy, red
spider mite, coffee leaf rust}) following these specifications:

1. Build a deep CNN with four convolutional stages
   (32 -> 64 -> 128 -> 256 filters), batch normalization, max pooling, and
   parallel convolutional paths with 3 x 3 and 5 x 5 kernels only in the
   third and fourth stages. Use global average pooling and a classification
   head with dropout.

2. Implement a two-phase training strategy:
   - Phase 1: freeze all convolutional layers and train only the head with
     a higher learning rate for 15 epochs, early-stopping patience = 5.
   - Phase 2: unfreeze all layers and fine-tune with a learning rate
     10 times smaller for 20 epochs, early-stopping patience = 7.

3. Select hyperparameters based on best practices. Apply CosineAnnealingLR
   during Phase 2.

4. Apply data augmentation: horizontal flip, random rotation up to 15
   degrees, color jitter for brightness and contrast, and random erasing.

5. Explicitly handle class imbalance via class weighting or focal loss, as
   appropriate for the task variant.
```

## Variants

| Task   | N_CLASSES | Class names                                          |
|--------|-----------|------------------------------------------------------|
| Task A | 2         | healthy, unhealthy                                   |
| Task B | 3         | healthy, red spider mite, coffee leaf rust           |

## Note on the architectural constraint

Tier 3 explicitly mandates a custom four-stage CNN. This is the most prescriptive specification in the study and was deliberately chosen to test whether prompt over-specification can override the LLM's otherwise sound architectural priors. The paper reports that this constraint leads every LLM to abandon transfer learning, which is the dominant cause of the Tier 3 performance degradation.

## Protocol

- Issue at the start of a fresh conversation with no in-context examples or memory carryover.
- Save the LLM's complete response verbatim as `scripts/{task}/tier3/{llm}.py`.
