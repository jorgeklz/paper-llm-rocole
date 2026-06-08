"""
Three-Class Robusta Coffee Leaf Classification
Two-Phase Training with Focal Loss and CosineAnnealingLR

Optimal hyperparameter vector:
    theta = [d=0.3, eta_1=5e-4, eta_2=5e-5, b=32, u=128, w=focal_loss]

Architecture: EfficientNet-B0 backbone + custom classification head
Device: CPU only
Usage: python train_coffee_cnn.py
"""

import os
import sys
import time
import copy
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    recall_score, roc_auc_score
)
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT   = "data/splits/task_b_3class"
NUM_CLASSES = 3
IMG_SIZE    = 224
SEED        = 42

# Optimal theta vector
D       = 0.3       # dropout rate
ETA_1   = 5e-4      # Phase 1 learning rate (head only)
ETA_2   = 5e-5      # Phase 2 learning rate (full network)
BATCH   = 32        # batch size
UNITS   = 128       # dense units in classification head

# Training schedule
PHASE1_EPOCHS = 15   # head-only warm-up
PHASE2_EPOCHS = 50   # full fine-tuning
ES_PATIENCE   = 10   # early-stopping patience (macro-F1)
ES_DELTA      = 0.001

# Class names (must match folder names)
CLASS_NAMES = ["healthy", "red_spider_mite", "coffee_leaf_rust"]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Focal Loss (addresses red_spider_mite minority class at 10.7%)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Multi-class focal loss.
    Dynamically down-weights easy (well-classified) examples so that
    gradient updates concentrate on hard minority-class samples.

    gamma=2.0 is the standard value from Lin et al. (2017).
    alpha=None uses uniform class priors; set alpha to a tensor of
    inverse-frequency weights to additionally compensate for imbalance.
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha   # shape (C,) or None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt      = torch.exp(-ce_loss)                          # probability of true class
        focal   = (1.0 - pt) ** self.gamma * ce_loss           # focal modulation

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal   = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_datasets(root: str):
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    test_dir  = os.path.join(root, "test")

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            sys.exit(f"[ERROR] Directory not found: {d}")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tf)

    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Classes detected : {train_ds.classes}")
    print(f"  Train samples    : {len(train_ds)}")
    print(f"  Val   samples    : {len(val_ds)}")
    print(f"  Test  samples    : {len(test_ds)}")

    # Per-class counts in training set
    targets = np.array(train_ds.targets)
    for idx, cls in enumerate(train_ds.classes):
        n = (targets == idx).sum()
        print(f"    train/{cls:<22}: {n}")
    print()

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Compute focal-loss alpha weights from training class frequencies
# ---------------------------------------------------------------------------
def compute_alpha(train_ds) -> torch.Tensor:
    """
    Inverse-frequency weights normalized to sum to 1.
    This combines focal modulation with mild class-weight correction,
    giving extra protection to red_spider_mite (minority class).
    """
    targets  = np.array(train_ds.targets)
    n_total  = len(targets)
    n_classes = len(train_ds.classes)
    freqs    = np.array([(targets == i).sum() for i in range(n_classes)], dtype=float)
    inv_freq = n_total / (n_classes * freqs)
    alpha    = inv_freq / inv_freq.sum()
    print(f"  Focal-loss alpha (inverse-freq): {dict(zip(train_ds.classes, alpha.round(4)))}")
    return torch.tensor(alpha, dtype=torch.float)


# ---------------------------------------------------------------------------
# Model: EfficientNet-B0 backbone + custom head
# ---------------------------------------------------------------------------
def build_model(num_classes: int, units: int, dropout: float) -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet.
    The original classifier is replaced with:
        Linear(1280 -> units) -> BatchNorm -> GELU -> Dropout(d) -> Linear(units -> C)
    """
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = backbone.classifier[1].in_features  # 1280

    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, units),
        nn.BatchNorm1d(units),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(units, num_classes),
    )
    return backbone


def freeze_backbone(model: nn.Module):
    """Freeze all parameters except the classification head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.001, mode: str = "max"):
        self.patience   = patience
        self.delta      = delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0
        self.triggered  = False

    def __call__(self, score: float) -> bool:
        improved = (
            self.best_score is None or
            (self.mode == "max" and score > self.best_score + self.delta) or
            (self.mode == "min" and score < self.best_score - self.delta)
        )
        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Single epoch: training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Single epoch: evaluation
# ---------------------------------------------------------------------------
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits  = model(images)
            loss    = criterion(logits, labels)
            probs   = torch.softmax(logits, dim=1)
            preds   = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------------------
# Training loop (one phase)
# ---------------------------------------------------------------------------
def run_phase(
    phase_name, model, train_loader, val_loader,
    optimizer, criterion, scheduler, max_epochs,
    early_stopper, device, best_state
):
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")
    header = f"{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>8}  {'ValAcc':>7}  {'MacroF1':>8}  {'ES':>4}"
    print(header)
    print("-" * len(header))

    best_f1   = early_stopper.best_score if early_stopper.best_score else -1.0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _, _ = eval_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        es_stop = early_stopper(va_f1)
        marker  = "*" if early_stopper.counter == 0 else " "

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {va_loss:>8.4f}  "
              f"{va_acc:>7.4f}  {va_f1:>8.4f}  {marker:>4}  "
              f"[{epoch - early_stopper.counter}/{max_epochs}]")

        # Save best model weights (tracked by macro-F1)
        if early_stopper.counter == 0:
            best_state = copy.deepcopy(model.state_dict())
            best_f1    = va_f1

        if es_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}. "
                  f"Best val macro-F1 = {best_f1:.4f}")
            break

    return best_state, best_f1


# ---------------------------------------------------------------------------
# Final evaluation on test set
# ---------------------------------------------------------------------------
def evaluate_test(model, test_loader, criterion, device, class_names):
    _, acc, macro_f1, labels, preds, probs = eval_epoch(
        model, test_loader, criterion, device
    )
    bal_acc  = balanced_accuracy_score(labels, preds)
    recalls  = recall_score(labels, preds, average=None, zero_division=0)
    test_loss, *_ = eval_epoch(model, test_loader, criterion, device)

    # One-vs-rest macro AUC (requires probability scores)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print(f"\n{'='*60}")
    print("  FINAL TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  Macro OvR AUC     : {auc:.4f}")
    print(f"  Test Loss (Focal) : {test_loss:.4f}")
    print(f"\n  Per-class Recall:")
    for cls, rec in zip(class_names, recalls):
        flag = " <-- MINORITY" if cls == "red_spider_mite" else ""
        print(f"    {cls:<25}: {rec:.4f}{flag}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cpu")
    print(f"\nDevice : {device}")

    # ---- 1. Data ----
    train_ds, val_ds, test_ds = load_datasets(DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=0)

    # ---- 2. Loss function: Focal Loss with inverse-frequency alpha ----
    alpha     = compute_alpha(train_ds)
    criterion = FocalLoss(gamma=2.0, alpha=alpha)

    # ---- 3. Model ----
    model = build_model(NUM_CLASSES, UNITS, D).to(device)
    print(f"\n  Backbone         : EfficientNet-B0 (pretrained ImageNet)")
    print(f"  Head units       : {UNITS}")
    print(f"  Dropout          : {D}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")

    # ---- 4. PHASE 1: Warm-up (head only) ----
    freeze_backbone(model)
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Phase 1 trainable params : {trainable_p1:,} (head only)")

    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ETA_1, weight_decay=1e-4
    )
    es_p1        = EarlyStopping(patience=ES_PATIENCE, delta=ES_DELTA, mode="max")
    best_state, _ = run_phase(
        "PHASE 1 - Head warm-up (backbone frozen)",
        model, train_loader, val_loader,
        optimizer_p1, criterion,
        scheduler    = None,
        max_epochs   = PHASE1_EPOCHS,
        early_stopper= es_p1,
        device       = device,
        best_state   = None,
    )

    # Restore best phase-1 weights before phase 2
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- 5. PHASE 2: Full fine-tuning ----
    unfreeze_all(model)
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Phase 2 trainable params : {trainable_p2:,} (entire network)")

    optimizer_p2 = optim.Adam(model.parameters(), lr=ETA_2, weight_decay=1e-4)

    # CosineAnnealingLR: LR decays from eta_2 to eta_min over PHASE2_EPOCHS
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2,
        T_max   = PHASE2_EPOCHS,
        eta_min = 1e-7,
    )

    es_p2        = EarlyStopping(patience=ES_PATIENCE, delta=ES_DELTA, mode="max")
    best_state, best_val_f1 = run_phase(
        "PHASE 2 - Full fine-tuning (CosineAnnealingLR)",
        model, train_loader, val_loader,
        optimizer_p2, criterion,
        scheduler    = scheduler_p2,
        max_epochs   = PHASE2_EPOCHS,
        early_stopper= es_p2,
        device       = device,
        best_state   = None,
    )

    # Restore best phase-2 weights for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Best checkpoint restored (val macro-F1 = {best_val_f1:.4f})")

    # ---- 6. Test set evaluation ----
    evaluate_test(model, test_loader, criterion, device, CLASS_NAMES)

    # ---- 7. Save final model ----
    save_path = "coffee_leaf_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names"     : CLASS_NAMES,
        "theta"           : {
            "d"    : D,
            "eta_1": ETA_1,
            "eta_2": ETA_2,
            "b"    : BATCH,
            "u"    : UNITS,
            "w"    : "focal_loss",
        },
    }, save_path)
    print(f"  Model saved to: {save_path}\n")


if __name__ == "__main__":
    main()