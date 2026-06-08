"""
train_coffee_leaf.py
====================
Deep Learning training script for binary classification of robusta coffee leaf
images (healthy vs unhealthy) using a custom four-stage CNN with parallel
convolutional paths in stages 3 and 4.

Run:
    python train_coffee_leaf.py

Requirements:
    pip install torch torchvision scikit-learn
"""

import os
import time
import math
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("data/splits/task_a_binary")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
TEST_DIR  = DATA_ROOT / "test"

# ---------------------------------------------------------------------------
# Hyperparameter rationale
# ---------------------------------------------------------------------------
#
# BATCH SIZE = 32
#   Provides stable gradient estimates on CPU without excessive memory use.
#   Larger batches (64+) on CPU are slow; smaller batches (8-16) introduce
#   high gradient variance. 32 is the empirical sweet spot for small-dataset
#   custom CNNs trained from scratch.
#
# LEARNING RATE Phase 1 = 3e-3
#   The head is randomly initialized, so a moderate-to-high LR converges
#   the dense layers quickly without overshooting. The convolutional backbone
#   is frozen in Phase 1, so there is no risk of destroying learned features.
#
# LEARNING RATE Phase 2 = 3e-4  (10x smaller)
#   All layers are unfrozen. A small LR prevents catastrophic forgetting of
#   the pattern representations learned in Phase 1 and allows gentle fine-
#   tuning of the convolutional filters together with the head.
#
# DROPOUT = 0.4
#   With ~1 560 training images and a moderately deep custom CNN, dropout
#   rates between 0.3 and 0.5 are typical. 0.4 gives strong regularization
#   without starving the classification head of signal. Applied after global
#   average pooling before the final dense layer.
#
# DENSE UNITS = 256
#   Global average pooling collapses the 256-channel feature map to a 256-D
#   vector. A single hidden layer of 256 units keeps the parameter count low
#   and avoids overfitting while retaining enough representational capacity
#   for a binary task.
#
# FOCAL LOSS alpha = 0.25, gamma = 2
#   Even though the dataset is approximately balanced (791 vs 769 samples),
#   Focal Loss is included so the script is robust to small imbalances and
#   to hard examples. It dynamically down-weights easy negatives, helping the
#   model focus on difficult leaf patterns near the decision boundary.

BATCH_SIZE   = 32
LR_PHASE1    = 3e-3
LR_PHASE2    = 3e-4
DROPOUT      = 0.40
DENSE_UNITS  = 256
EPOCHS_P1    = 15
PATIENCE_P1  = 5
EPOCHS_P2    = 20
PATIENCE_P2  = 7
NUM_CLASSES  = 2
IMG_SIZE     = 224

# ImageNet statistics (used for normalization; effective even for from-scratch
# training because the distribution of natural-image pixel values is similar)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ===========================================================================
# 1.  DATA TRANSFORMS
# ===========================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    # RandomErasing: erases a random rectangular patch to simulate occlusion
    # and force the model to use distributed leaf features rather than a
    # single discriminative spot.
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ===========================================================================
# 2.  DATASETS & LOADERS
# ===========================================================================

def build_loaders():
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=eval_transform)
    test_ds  = datasets.ImageFolder(str(TEST_DIR),  transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    class_names = train_ds.classes          # ['healthy', 'unhealthy']
    targets     = torch.tensor(train_ds.targets)

    return train_loader, val_loader, test_loader, class_names, targets


# ===========================================================================
# 3.  FOCAL LOSS
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Binary / multi-class Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : float or list
        Class weighting. For binary: single scalar applied to the positive
        class. For multi-class: list of per-class weights.
    gamma : float
        Focusing parameter. gamma=0 reduces to standard cross-entropy.
    """

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2,
                 class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

        if class_weights is not None:
            # Accept externally computed class weights for imbalanced data
            self.register_buffer("weight", class_weights)
        else:
            self.weight = None

        # alpha scales the focal term; ignored when class_weights is set
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets,
                                         weight=self.weight,
                                         reduction="none")
        pt  = torch.exp(-ce)                         # probability of true class
        fl  = (1.0 - pt) ** self.gamma * ce
        return fl.mean()


# ===========================================================================
# 4.  PARALLEL CONVOLUTIONAL BLOCK (used in stages 3 and 4)
# ===========================================================================

class ParallelConvBlock(nn.Module):
    """
    Two parallel conv paths with 3x3 and 5x5 kernels.
    Their outputs are concatenated along the channel dimension, then reduced
    back to `out_channels` with a 1x1 projection so the stage output width
    stays predictable.

    Architecture (per path):
        Conv -> BN -> ReLU

    The two paths are concatenated -> 1x1 Conv -> BN -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2     # each branch produces half the channels

        # 3x3 path
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # 5x5 path (larger receptive field captures broader disease patches)
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # 1x1 projection to fuse branches and keep channel count tidy
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b3, b5], dim=1)    # concat on channel axis
        out = self.fuse(out)
        return self.pool(out)


# ===========================================================================
# 5.  STANDARD CONVOLUTIONAL BLOCK (stages 1 and 2)
# ===========================================================================

class ConvBlock(nn.Module):
    """
    Single-path convolutional block: Conv -> BN -> ReLU -> MaxPool.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


# ===========================================================================
# 6.  FULL MODEL
# ===========================================================================

class CoffeeCNN(nn.Module):
    """
    Custom four-stage CNN for coffee leaf binary classification.

    Stage 1:  ConvBlock(3  -> 32)   -- standard, 3x3
    Stage 2:  ConvBlock(32 -> 64)   -- standard, 3x3
    Stage 3:  ParallelConvBlock(64  -> 128)  -- parallel 3x3 + 5x5
    Stage 4:  ParallelConvBlock(128 -> 256)  -- parallel 3x3 + 5x5

    Input 224x224 -> after 4 MaxPool(2) -> 14x14 feature maps
    Global Average Pooling -> 256-D vector
    Dropout(0.4) -> Dense(256) -> ReLU -> Dense(2)
    """

    def __init__(self, num_classes=2, dropout=DROPOUT, dense_units=DENSE_UNITS):
        super().__init__()

        # ------ Convolutional backbone ------
        self.stage1 = ConvBlock(3,   32)          # 224 -> 112
        self.stage2 = ConvBlock(32,  64)          # 112 -> 56
        self.stage3 = ParallelConvBlock(64,  128) #  56 -> 28
        self.stage4 = ParallelConvBlock(128, 256) #  28 -> 14

        # ------ Classification head ------
        self.gap = nn.AdaptiveAvgPool2d(1)        # 14x14x256 -> 1x1x256

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, dense_units),
            nn.ReLU(inplace=True),
            nn.Linear(dense_units, num_classes),
        )

    # Convenience helpers to freeze / unfreeze the backbone ----------------

    def freeze_backbone(self):
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for param in stage.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for param in stage.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return self.classifier(x)


# ===========================================================================
# 7.  TRAINING UTILITIES
# ===========================================================================

class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = math.inf
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Deep-copy state dict so the best checkpoint is preserved
            self.best_state = {k: v.clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def compute_class_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights.
    w_c = N / (num_classes * count_c)
    """
    counts = torch.bincount(targets, minlength=num_classes).float()
    total  = counts.sum()
    return total / (num_classes * counts)


def run_epoch(model, loader, criterion, optimizer=None, train=True):
    """Run one epoch; return avg loss, list of true labels, list of probs."""
    model.train(train)
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss  += loss.item() * imgs.size(0)
            probs        = torch.softmax(logits, dim=1)[:, 1]  # prob of class 1
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_probs)


def metrics_from_probs(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, probs)
    return acc, prec, rec, f1, auc


def print_metrics(split, acc, prec, rec, f1, auc):
    print(f"  {split:5s}  Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")


def train_phase(phase_num, model, train_loader, val_loader,
                criterion, optimizer, scheduler,
                num_epochs, patience, label="Phase"):
    """
    Generic training loop for one phase.
    Returns the EarlyStopping object (which holds the best checkpoint).
    """
    early_stop = EarlyStopping(patience=patience)

    print(f"\n{'='*65}")
    print(f"  {label}  ({num_epochs} max epochs, patience={patience})")
    print(f"{'='*65}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p = run_epoch(
            model, train_loader, criterion, optimizer, train=True)
        va_loss, va_y, va_p = run_epoch(
            model, val_loader, criterion, train=False)

        if scheduler is not None:
            scheduler.step()

        tr_acc, _, _, tr_f1, _ = metrics_from_probs(tr_y, tr_p)
        va_acc, _, _, va_f1, _ = metrics_from_probs(va_y, va_p)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}/{num_epochs}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} tr_f1={tr_f1:.4f}  |  "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} va_f1={va_f1:.4f}  "
              f"[{elapsed:.1f}s]")

        if early_stop.step(va_loss, model):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    early_stop.restore_best(model)
    print(f"  Best val loss: {early_stop.best_loss:.4f}")
    return early_stop


# ===========================================================================
# 8.  MAIN
# ===========================================================================

def main():
    print("\n" + "="*65)
    print("  Coffee Leaf Binary Classifier -- Custom Four-Stage CNN")
    print("="*65)

    # ------------------------------------------------------------------ data
    print("\n[1/5] Loading datasets ...")
    train_loader, val_loader, test_loader, class_names, targets = build_loaders()
    print(f"      Classes  : {class_names}")
    print(f"      Train    : {len(train_loader.dataset)} images")
    print(f"      Val      : {len(val_loader.dataset)} images")
    print(f"      Test     : {len(test_loader.dataset)} images")

    # ------------------------------------------------- class weights / loss
    class_weights = compute_class_weights(targets, num_classes=NUM_CLASSES)
    print(f"\n[2/5] Class weights: {class_weights.numpy()} "
          f"(healthy={class_weights[0]:.3f}, unhealthy={class_weights[1]:.3f})")

    # Focal Loss with inverse-frequency class weights already embedded
    criterion = FocalLoss(gamma=2.0, num_classes=NUM_CLASSES,
                          class_weights=class_weights)

    # ------------------------------------------------------- model
    print("\n[3/5] Building model ...")
    model = CoffeeCNN(num_classes=NUM_CLASSES)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"      Total parameters    : {total_params:,}")
    print(f"      Trainable (initial) : {trainable_params:,}")

    # ==========================================================
    # PHASE 1 -- train classification head only
    # ==========================================================
    print("\n[4/5] Two-phase training ...")
    model.freeze_backbone()

    head_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_p1 = optim.Adam(head_params, lr=LR_PHASE1, weight_decay=1e-4)

    train_phase(
        phase_num    = 1,
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        optimizer    = optimizer_p1,
        scheduler    = None,          # no scheduler in Phase 1
        num_epochs   = EPOCHS_P1,
        patience     = PATIENCE_P1,
        label        = "Phase 1 -- Head Only (backbone frozen)",
    )

    # ==========================================================
    # PHASE 2 -- fine-tune all layers with CosineAnnealingLR
    # ==========================================================
    model.unfreeze_backbone()

    optimizer_p2 = optim.Adam(model.parameters(), lr=LR_PHASE2,
                               weight_decay=1e-4)
    # CosineAnnealingLR decays LR from LR_PHASE2 to near-zero over EPOCHS_P2
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=EPOCHS_P2, eta_min=1e-6)

    train_phase(
        phase_num    = 2,
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        optimizer    = optimizer_p2,
        scheduler    = scheduler_p2,
        num_epochs   = EPOCHS_P2,
        patience     = PATIENCE_P2,
        label        = "Phase 2 -- Full Fine-Tuning (all layers + CosineAnnealingLR)",
    )

    # ==========================================================
    # EVALUATION on held-out test set
    # ==========================================================
    print("\n[5/5] Evaluating on test set ...")
    _, te_y, te_p = run_epoch(model, test_loader, criterion, train=False)
    acc, prec, rec, f1, auc = metrics_from_probs(te_y, te_p)

    print("\n" + "="*65)
    print("  FINAL TEST SET RESULTS")
    print("="*65)
    print_metrics("Test", acc, prec, rec, f1, auc)
    print("="*65 + "\n")

    # Also show per-split summary for completeness
    print("  Per-split summary:")
    _, tr_y, tr_p = run_epoch(model, train_loader, criterion, train=False)
    _, va_y, va_p = run_epoch(model, val_loader,   criterion, train=False)
    print_metrics("Train", *metrics_from_probs(tr_y, tr_p))
    print_metrics("Val  ", *metrics_from_probs(va_y, va_p))
    print_metrics("Test ", acc, prec, rec, f1, auc)
    print()


if __name__ == "__main__":
    main()