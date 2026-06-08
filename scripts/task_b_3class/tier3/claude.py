"""
train_coffee_leaf.py
=====================
Three-class coffee leaf image classifier (healthy / coffee_leaf_rust / red_spider_mite).

Architecture  : Custom four-stage CNN with parallel conv paths in stages 3-4
Training      : Two-phase strategy (frozen head -> full fine-tune)
Imbalance     : Weighted CrossEntropy loss
Device        : CPU only
Usage         : python train_coffee_leaf.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER RATIONALE
# ─────────────────────────────────────────────────────────────────────────────
# dropout_rate = 0.45
#   Medium-high dropout chosen because the model is trained from scratch on a
#   relatively small dataset (~1 560 images). 0.45 provides strong regularisation
#   without over-suppressing useful activations. Values >0.6 degrade convergence
#   on small data; values <0.3 tend to overfit.
#
# lr_phase1 = 3e-3
#   Only the classification head is trained in Phase 1. A higher LR (3e-3)
#   speeds up head convergence because the frozen conv weights provide stable
#   features. Adam with this LR is a standard starting point for lightweight
#   classification heads.
#
# lr_phase2 = 3e-4  (= lr_phase1 / 10)
#   Full fine-tuning risks catastrophic forgetting of the feature representations
#   learned in Phase 1. A 10x smaller LR preserves those representations while
#   still allowing global optimisation. CosineAnnealingLR further decays the LR
#   smoothly across Phase 2 epochs.
#
# batch_size = 32
#   Balances GPU/CPU memory usage with gradient estimate quality. Batch sizes of
#   16-64 are standard for fine-grained image classification at 224x224. Smaller
#   batches increase stochasticity (helpful for generalisation); larger batches
#   reduce it. 32 is the sweet spot for a moderately sized dataset on CPU.
#
# dense_units = 512
#   Single hidden layer with 512 units sits between the 256-channel GAP output
#   and the 3-class softmax. 512 units give enough capacity to learn non-linear
#   class boundaries while keeping the head lightweight relative to the conv
#   backbone.
#
# CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────
# Strategy: Inverse-frequency class weighting in CrossEntropyLoss.
#
# Rationale:
#   The dataset has a 4.7:3.6:1 ratio (healthy:rust:spider_mite). Focal Loss
#   is an alternative, but requires tuning the gamma exponent and is harder to
#   calibrate on three classes without a validation sweep. Inverse-frequency
#   weighting is simpler, interpretable, and well-suited to MODERATE imbalance
#   (minority class is ~10.7 %, not <1 %). It directly scales the gradient
#   contribution of each class so that the minority class red_spider_mite
#   receives proportionally larger gradient updates, preserving per-class recall
#   without distorting the decision boundary as aggressively as oversampling.
#
#   Weights are computed as:  w_c = N_total / (n_classes * N_c)
#     healthy           : 1560 / (3 * 791) = 0.657
#     coffee_leaf_rust  : 1560 / (3 * 602) = 0.864
#     red_spider_mite   : 1560 / (3 * 167) = 3.114
#
#   The spider_mite weight is ~4.7x that of healthy, meaning every misclassified
#   spider_mite sample contributes nearly 5x more to the loss update, directly
#   incentivising the model to maintain high recall on that class.
# ─────────────────────────────────────────────────────────────────────────────

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)

# ─── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT  = "data/splits/task_b_3class"
TRAIN_DIR  = os.path.join(DATA_ROOT, "train")
VAL_DIR    = os.path.join(DATA_ROOT, "val")
TEST_DIR   = os.path.join(DATA_ROOT, "test")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
DROPOUT_RATE  = 0.45
DENSE_UNITS   = 512
LR_PHASE1     = 3e-3
LR_PHASE2     = 3e-4   # LR_PHASE1 / 10
EPOCHS_P1     = 15
EPOCHS_P2     = 20
PATIENCE_P1   = 5
PATIENCE_P2   = 7
NUM_CLASSES   = 3

# Class counts (from spec) — used for loss weighting
CLASS_COUNTS  = {"healthy": 791, "coffee_leaf_rust": 602, "red_spider_mite": 167}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class ParallelConvBlock(nn.Module):
    """
    Parallel convolutional paths with 3x3 and 5x5 kernels.
    Outputs are concatenated along the channel dimension, then projected
    back to `out_channels` to keep memory footprint predictable.
    Used in stages 3 and 4.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 2  # each path contributes half the output channels

        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.path5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        # 1x1 projection after concat so spatial maps stay at out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.project(torch.cat([self.path3(x), self.path5(x)], dim=1))


class ConvStage(nn.Module):
    """Standard single-path conv stage (stages 1 and 2)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CoffeeCNN(nn.Module):
    """
    Four-stage custom CNN for coffee leaf disease classification.

    Stage 1 : 3 -> 32   (standard double-conv)
    Stage 2 : 32 -> 64  (standard double-conv)
    Stage 3 : 64 -> 128 (parallel 3x3 + 5x5 paths)
    Stage 4 : 128 -> 256 (parallel 3x3 + 5x5 paths)
    Head    : GAP -> 512 Dense -> Dropout -> 3-way softmax
    """
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.45,
                 dense_units: int = 512):
        super().__init__()

        self.stage1 = ConvStage(3, 32)
        self.pool1  = nn.MaxPool2d(2, 2)   # 224 -> 112

        self.stage2 = ConvStage(32, 64)
        self.pool2  = nn.MaxPool2d(2, 2)   # 112 -> 56

        self.stage3 = ParallelConvBlock(64, 128)
        self.pool3  = nn.MaxPool2d(2, 2)   # 56 -> 28

        self.stage4 = ParallelConvBlock(128, 256)
        self.pool4  = nn.MaxPool2d(2, 2)   # 28 -> 14

        self.gap    = nn.AdaptiveAvgPool2d(1)  # 14 -> 1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, num_classes),
        )

    # --- convenience helpers for phase-based freezing ---------------------
    @property
    def conv_layers(self):
        return [self.stage1, self.pool1,
                self.stage2, self.pool2,
                self.stage3, self.pool3,
                self.stage4, self.pool4,
                self.gap]

    def freeze_conv(self):
        for module in self.conv_layers:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.pool4(self.stage4(x))
        x = self.gap(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(class_to_idx: dict, counts: dict) -> torch.Tensor:
    """Inverse-frequency weights: w_c = N_total / (n_classes * N_c)."""
    n_classes = len(class_to_idx)
    n_total   = sum(counts.values())
    weights   = torch.zeros(n_classes)
    for cls_name, idx in class_to_idx.items():
        weights[idx] = n_total / (n_classes * counts[cls_name])
    print("\n[Imbalance] Class weights:")
    for cls_name, idx in class_to_idx.items():
        print(f"  {cls_name:<20s}: {weights[idx]:.4f}")
    return weights


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs      = model(images)
            loss         = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)
    return running_loss / total, correct / total


def train_phase(model, train_loader, val_loader, criterion,
                optimizer, scheduler, n_epochs, patience, phase_name):
    """Generic training loop with early stopping. Returns best model weights."""
    best_val_loss   = float("inf")
    best_weights    = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>3}/{n_epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch} "
                      f"(patience={patience}).")
                break

    model.load_state_dict(best_weights)
    print(f"  Best val_loss = {best_val_loss:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def final_evaluation(model, test_loader, criterion, class_names):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    running_loss = 0.0
    total        = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs      = model(images)
            loss         = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs         = torch.softmax(outputs, dim=1)
            preds         = probs.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())
            total        += labels.size(0)

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_prob  = np.array(all_probs)

    test_loss  = running_loss / total
    accuracy   = (y_true == y_pred).mean()
    macro_f1   = f1_score(y_true, y_pred, average="macro")
    bal_acc    = balanced_accuracy_score(y_true, y_pred)
    per_recall = recall_score(y_true, y_pred, average=None)
    # One-vs-rest macro AUC (requires probability scores)
    macro_auc  = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    print("\n" + "="*60)
    print("  TEST SET RESULTS")
    print("="*60)
    print(f"  Loss             : {test_loss:.4f}")
    print(f"  Accuracy         : {accuracy:.4f}")
    print(f"  Macro-F1         : {macro_f1:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro AUC (OvR)  : {macro_auc:.4f}")
    print("\n  Per-class Recall:")
    for i, cls in enumerate(class_names):
        print(f"    {cls:<22s}: {per_recall[i]:.4f}")
    print("="*60)

    return {
        "test_loss"        : test_loss,
        "accuracy"         : accuracy,
        "macro_f1"         : macro_f1,
        "balanced_accuracy": bal_acc,
        "per_class_recall" : dict(zip(class_names, per_recall)),
        "macro_auc"        : macro_auc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=eval_transform)

    print(f"[Data] Train  : {len(train_dataset)} images")
    print(f"[Data] Val    : {len(val_dataset)} images")
    print(f"[Data] Test   : {len(test_dataset)} images")
    print(f"[Data] Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    class_to_idx  = train_dataset.class_to_idx
    class_names   = [c for c, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    # ── Class weights ──────────────────────────────────────────────────────────
    weights  = compute_class_weights(class_to_idx, CLASS_COUNTS)
    criterion = nn.CrossEntropyLoss(weight=weights)   # CPU tensors only

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CoffeeCNN(num_classes=NUM_CLASSES,
                      dropout_rate=DROPOUT_RATE,
                      dense_units=DENSE_UNITS)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Total parameters    : {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 1 - Train classification head only
    # ════════════════════════════════════════════════════════════════════════════
    model.freeze_conv()
    trainable_params_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Phase 1] Frozen conv. Trainable params: {trainable_params_p1:,}")

    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1,
        weight_decay=1e-4,
    )

    model = train_phase(
        model, train_loader, val_loader, criterion,
        optimizer=optimizer_p1,
        scheduler=None,
        n_epochs=EPOCHS_P1,
        patience=PATIENCE_P1,
        phase_name="PHASE 1 | Head-only training | lr=3e-3 | epochs=15 | patience=5",
    )

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 2 - Full fine-tuning with CosineAnnealingLR
    # ════════════════════════════════════════════════════════════════════════════
    model.unfreeze_all()
    trainable_params_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Phase 2] All layers unfrozen. Trainable params: {trainable_params_p2:,}")

    optimizer_p2 = optim.Adam(
        model.parameters(),
        lr=LR_PHASE2,
        weight_decay=1e-4,
    )
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=EPOCHS_P2, eta_min=1e-6
    )

    model = train_phase(
        model, train_loader, val_loader, criterion,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        n_epochs=EPOCHS_P2,
        patience=PATIENCE_P2,
        phase_name="PHASE 2 | Full fine-tune | lr=3e-4 | CosineAnnealingLR | epochs=20 | patience=7",
    )

    # ════════════════════════════════════════════════════════════════════════════
    # FINAL TEST EVALUATION
    # ════════════════════════════════════════════════════════════════════════════
    results = final_evaluation(model, test_loader, criterion, class_names)

    # ── Save model ─────────────────────────────────────────────────────────────
    save_path = "coffee_leaf_cnn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n[Saved] Model weights -> {save_path}")

    return results


if __name__ == "__main__":
    main()