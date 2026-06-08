"""
train_coffee_leaf.py
====================
Binary classification of Robusta coffee leaf images (healthy vs unhealthy).

Architecture : EfficientNet-B0 (ImageNet pretrained) with a custom classifier head.
Strategy     : Progressive training with layer freezing / unfreezing in 3 phases.
Device       : CPU only.

Expected directory layout
--------------------------
data/splits/task_a_binary/
    train/
        healthy/
        unhealthy/
    val/
        healthy/
        unhealthy/
    test/
        healthy/
        unhealthy/

Run
---
    python train_coffee_leaf.py
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Paths and hyper-parameters
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join("data", "splits", "task_a_binary")
IMG_SIZE  = 224
BATCH_SIZE = 32          # 32 is a safe choice on CPU (reduce if RAM is tight)
NUM_WORKERS = 0          # 0 avoids multiprocessing issues on Windows / macOS

# Phase-specific epochs and learning rates
PHASE_CONFIG = [
    # (label,               epochs, lr_backbone,  lr_head)
    ("Phase 1 - Head only",      8,        0.0,   1e-3),
    ("Phase 2 - Last 2 blocks",  8,        1e-4,  5e-4),
    ("Phase 3 - Full network",  10,        5e-5,  1e-4),
]

DROPOUT_RATE = 0.4
WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------------------
# 2. Data transforms
# ---------------------------------------------------------------------------
# ImageNet statistics (used because the backbone is pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),   # slight oversize for crop
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# 3. Datasets and data loaders
# ---------------------------------------------------------------------------
print("=" * 65)
print("Loading datasets ...")

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_ROOT, "train"), transform=train_transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATA_ROOT, "val"), transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    os.path.join(DATA_ROOT, "test"), transform=eval_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False
)

# Class mapping (ImageFolder sorts alphabetically: healthy=0, unhealthy=1)
class_names = train_dataset.classes
print(f"Classes  : {class_names}")
print(f"Train    : {len(train_dataset)} images")
print(f"Val      : {len(val_dataset)} images")
print(f"Test     : {len(test_dataset)} images")

# ---------------------------------------------------------------------------
# 4. Class-weighted loss (handles any residual imbalance)
# ---------------------------------------------------------------------------
labels = [s[1] for s in train_dataset.samples]
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"Class counts (train) : {dict(zip(class_names, class_counts))}")
print(f"Class weights        : {dict(zip(class_names, class_weights.round(4)))}")

# ---------------------------------------------------------------------------
# 5. Model: EfficientNet-B0 with a custom classifier head
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2, dropout: float = DROPOUT_RATE) -> nn.Module:
    """
    Load EfficientNet-B0 with pretrained ImageNet weights and replace the
    default classifier with a two-layer head suited for binary classification.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace the original classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),                         # SiLU (swish) fits EfficientNet style
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(256, num_classes),
    )
    return model


device = torch.device("cpu")
model = build_model()
model = model.to(device)

# ---------------------------------------------------------------------------
# 6. Helper: freeze / unfreeze layers
# ---------------------------------------------------------------------------

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def freeze_backbone(model: nn.Module) -> None:
    """Freeze the entire EfficientNet feature extractor."""
    set_requires_grad(model.features, False)


def unfreeze_last_n_blocks(model: nn.Module, n: int) -> None:
    """
    EfficientNet-B0 has 9 sequential blocks inside model.features (indices 0-8).
    Unfreeze only the last n blocks to allow fine-tuning with a small LR.
    """
    total = len(model.features)            # 9 for EfficientNet-B0
    for i, block in enumerate(model.features):
        flag = i >= (total - n)
        set_requires_grad(block, flag)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter in the network."""
    set_requires_grad(model, True)


# ---------------------------------------------------------------------------
# 7. Training utilities
# ---------------------------------------------------------------------------

def make_optimizer(model: nn.Module, lr_backbone: float, lr_head: float) -> optim.Optimizer:
    """
    Build an AdamW optimizer with separate parameter groups for the backbone
    and the classifier head, supporting different learning rates.
    """
    head_params = list(model.classifier.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    return optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    is_train: bool,
) -> tuple[float, float]:
    """
    Execute one epoch of training or evaluation.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0-1)
    """
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents instability when LRs are large
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_phase(
    model: nn.Module,
    label: str,
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    criterion: nn.Module,
) -> nn.Module:
    """
    Run a single training phase and return the best model checkpoint by
    validation loss (early-stopping with patience and cosine annealing).
    """
    print(f"\n{'=' * 65}")
    print(f"  {label}  |  epochs={epochs}  lr_bb={lr_backbone}  lr_head={lr_head}")
    print(f"{'=' * 65}")

    optimizer = make_optimizer(model, lr_backbone, lr_head)
    # CosineAnnealingLR gives a smooth LR decay that avoids abrupt drops
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience = 5
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, is_train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, None,      is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    model.load_state_dict(best_weights)
    print(f"  => Best val loss in this phase: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# 8. Progressive training loop
# ---------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print("\nStarting progressive training ...\n")

for phase_idx, (label, epochs, lr_bb, lr_head) in enumerate(PHASE_CONFIG):
    # --- Freezing strategy ---
    if phase_idx == 0:
        # Phase 1: Only the new head is trainable
        freeze_backbone(model)
        set_requires_grad(model.classifier, True)
    elif phase_idx == 1:
        # Phase 2: Unfreeze last 2 EfficientNet blocks + head
        unfreeze_last_n_blocks(model, n=2)
        set_requires_grad(model.classifier, True)
    else:
        # Phase 3: Full network end-to-end
        unfreeze_all(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total_params:,}")

    model = train_phase(model, label, epochs, lr_bb, lr_head, criterion)

# ---------------------------------------------------------------------------
# 9. Final evaluation on the test set
# ---------------------------------------------------------------------------

print(f"\n{'=' * 65}")
print("Evaluating on the held-out test set ...")
print(f"{'=' * 65}")

model.eval()
all_labels  = []
all_preds   = []
all_probs   = []   # Softmax probability for the positive class (unhealthy=1)

softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs  = softmax(logits)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs[:, 1].cpu().numpy())   # prob of class 1 (unhealthy)

# Compute metrics
y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
auc       = roc_auc_score(y_true, y_prob)

print(f"\n{'=' * 65}")
print("  TEST SET RESULTS")
print(f"{'=' * 65}")
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"  Precision : {precision:.4f}  (macro)")
print(f"  Recall    : {recall:.4f}  (macro)")
print(f"  F1 Score  : {f1:.4f}  (macro)")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"{'=' * 65}")

# Per-class breakdown (useful for identifying class-specific weaknesses)
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nPer-class classification report:")
print(report)