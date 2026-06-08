"""
Coffee Leaf Disease Classifier
================================
3-class CNN classifier using EfficientNet-B0 with progressive layer
unfreezing to handle a moderately imbalanced dataset.

Classes:
    0 - healthy          (791 images)
    1 - coffee_leaf_rust (602 images)
    2 - red_spider_mite  (167 images)

Run:
    python train_coffee_cnn.py
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, recall_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1.  Paths and hyper-parameters
# ─────────────────────────────────────────────
DATA_ROOT   = "data/splits/task_b_3class"
NUM_CLASSES = 3
IMG_SIZE    = 224
BATCH_SIZE  = 32

# Progressive training schedule
# Each entry: (epochs, layers_to_unfreeze, lr, wd)
# "layers_to_unfreeze" is the number of EfficientNet feature blocks to open
# from the END of features (0 = only classifier, -1 = all layers)
SCHEDULE = [
    {"epochs": 10, "unfreeze": 0,  "lr": 3e-3, "wd": 1e-4},   # Phase 1: classifier only
    {"epochs": 10, "unfreeze": 3,  "lr": 5e-4, "wd": 1e-4},   # Phase 2: last 3 blocks
    {"epochs": 15, "unfreeze": -1, "lr": 1e-4, "wd": 1e-5},   # Phase 3: full fine-tune
]

DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# 2.  Data transforms
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(20),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─────────────────────────────────────────────
# 3.  Datasets and loaders
# ─────────────────────────────────────────────
train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),   transform=eval_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=eval_tf)

CLASS_NAMES = train_ds.classes          # ['coffee_leaf_rust', 'healthy', 'red_spider_mite']
print(f"Class mapping: {train_ds.class_to_idx}")
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

# Weighted sampler to counter class imbalance
targets = np.array(train_ds.targets)
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).float(),
    num_samples=len(train_ds),
    replacement=True,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)

# ─────────────────────────────────────────────
# 4.  Class-weighted loss (extra guard vs imbalance)
# ─────────────────────────────────────────────
cw = torch.tensor(class_weights / class_weights.sum() * NUM_CLASSES,
                  dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=cw)

# ─────────────────────────────────────────────
# 5.  Model: EfficientNet-B0 with custom head
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model

model = build_model(NUM_CLASSES).to(DEVICE)

# ─────────────────────────────────────────────
# 6.  Layer-freezing helpers
# ─────────────────────────────────────────────
def freeze_all_features(model: nn.Module) -> None:
    """Freeze all feature-extraction layers; only classifier trains."""
    for param in model.features.parameters():
        param.requires_grad = False

def unfreeze_last_n_blocks(model: nn.Module, n: int) -> None:
    """
    EfficientNet-B0 has 9 sequential blocks inside model.features (indices 0-8).
    Unfreeze the last `n` blocks.  n == -1 unfreezes everything.
    """
    if n == -1:
        for param in model.features.parameters():
            param.requires_grad = True
        return
    total_blocks = len(model.features)          # 9 for EfficientNet-B0
    unfreeze_from = max(0, total_blocks - n)
    for i, block in enumerate(model.features):
        requires = (i >= unfreeze_from)
        for param in block.parameters():
            param.requires_grad = requires

def get_optimizer(model: nn.Module, lr: float, wd: float) -> optim.Optimizer:
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    return optim.AdamW(trainable, lr=lr, weight_decay=wd)

# ─────────────────────────────────────────────
# 7.  Training and validation routines
# ─────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct   += (logits.argmax(1) == labels).sum().item()
            total     += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader):
    """Return loss, accuracy, and raw predictions/probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            total_loss += loss.item() * imgs.size(0)
            total      += imgs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        total_loss / total,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )

# ─────────────────────────────────────────────
# 8.  Progressive training loop
# ─────────────────────────────────────────────
best_val_f1   = -1.0
best_state    = None
global_epoch  = 0

print("\n" + "=" * 60)
print("PROGRESSIVE TRAINING")
print("=" * 60)

for phase_idx, phase in enumerate(SCHEDULE, 1):
    # -- configure layer freezing
    freeze_all_features(model)
    unfreeze_last_n_blocks(model, phase["unfreeze"])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Phase {phase_idx}: unfreeze={phase['unfreeze']} | "
          f"lr={phase['lr']} | trainable params={trainable_params:,} ---")

    optimizer = get_optimizer(model, phase["lr"], phase["wd"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase["epochs"], eta_min=phase["lr"] * 0.05
    )

    for ep in range(1, phase["epochs"] + 1):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, train=True)
        vl_loss, vl_labels, vl_preds, _ = evaluate(model, val_loader)
        vl_acc  = (vl_labels == vl_preds).mean()
        vl_f1   = f1_score(vl_labels, vl_preds, average="macro", zero_division=0)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {global_epoch:3d} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} val_macroF1={vl_f1:.3f} | "
              f"{elapsed:.1f}s")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

print(f"\nBest val macro-F1 achieved: {best_val_f1:.4f}")

# ─────────────────────────────────────────────
# 9.  Final evaluation on test set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST SET EVALUATION  (best checkpoint)")
print("=" * 60)

model.load_state_dict(best_state)
te_loss, te_labels, te_preds, te_probs = evaluate(model, test_loader)

# Core metrics
accuracy    = (te_labels == te_preds).mean()
macro_f1    = f1_score(te_labels, te_preds, average="macro", zero_division=0)
bal_acc     = balanced_accuracy_score(te_labels, te_preds)
per_recall  = recall_score(te_labels, te_preds, average=None, zero_division=0)

# One-vs-rest macro AUC
te_labels_bin = label_binarize(te_labels, classes=list(range(NUM_CLASSES)))
try:
    macro_auc = roc_auc_score(te_labels_bin, te_probs,
                              average="macro", multi_class="ovr")
except ValueError as exc:
    macro_auc = float("nan")
    print(f"  AUC warning: {exc}")

# ─────────────────────────────────────────────
# 10.  Print results
# ─────────────────────────────────────────────
print(f"\n  Test Loss              : {te_loss:.4f}")
print(f"  Accuracy               : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Macro F1               : {macro_f1:.4f}")
print(f"  Balanced Accuracy      : {bal_acc:.4f}")
print(f"  Macro OvR AUC          : {macro_auc:.4f}")
print()
print("  Per-Class Recall:")
for idx, cls_name in enumerate(CLASS_NAMES):
    print(f"    {cls_name:<22}: {per_recall[idx]:.4f}")

print()
print("  Full Classification Report:")
print(classification_report(te_labels, te_preds,
                             target_names=CLASS_NAMES, zero_division=0))
print("=" * 60)
print("Training complete.")