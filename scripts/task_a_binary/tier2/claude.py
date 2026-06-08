"""
Binary Classification of Robusta Coffee Leaves (Healthy vs Unhealthy)
Two-phase training strategy with custom CNN head on top of a pretrained backbone.

Optimal hyperparameter vector:
    theta = [d=0.3, eta_1=1e-3, eta_2=5e-5, b=32, u=128, w=none]

Run:
    python train_coffee_cnn.py
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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------------------------------------------------------------------------
# 0.  REPRODUCIBILITY
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1.  OPTIMAL HYPERPARAMETER VECTOR  theta = [d, eta_1, eta_2, b, u, w]
# ---------------------------------------------------------------------------
DROPOUT_RATE   = 0.3          # d  – moderate regularisation; avoids under/over-fitting
LR_PHASE1      = 1e-3         # eta_1 – fast head-only learning; backbone is frozen
LR_PHASE2      = 5e-5         # eta_2 – gentle fine-tuning; prevents catastrophic forgetting
BATCH_SIZE     = 32           # b  – good bias-variance trade-off for ~1 500 images
NUM_UNITS      = 128          # u  – sufficient capacity for 2-class head
CLASS_STRATEGY = "none"       # w  – dataset is ~balanced (791 vs 769); no correction needed

EPOCHS_PHASE1  = 15           # head-only warmup
EPOCHS_PHASE2  = 40           # full fine-tuning ceiling (early stopping will cut it short)
PATIENCE       = 7            # early-stopping patience (Phase 2)
DATA_ROOT      = "data/splits/task_a_binary"
IMG_SIZE       = 224

# ---------------------------------------------------------------------------
# 2.  DATA LOADERS
# ---------------------------------------------------------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

def build_loaders(root, batch_size):
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(root, "test"),  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    print(f"\n[Data] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[Data] Classes: {train_ds.classes}  (index map: {train_ds.class_to_idx})")
    return train_loader, val_loader, test_loader, train_ds.classes

# ---------------------------------------------------------------------------
# 3.  MODEL DEFINITION
# ---------------------------------------------------------------------------
def build_model(num_units, dropout_rate):
    """
    Backbone: EfficientNet-B0 pretrained on ImageNet.
    Classification head: AdaptiveAvgPool -> Flatten -> BN -> Dropout ->
                         Linear(num_units) -> ReLU -> Dropout -> Linear(1)
    Output is a raw logit (BCEWithLogitsLoss handles sigmoid internally).
    """
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = backbone.classifier[1].in_features  # 1280 for B0

    # Replace the original classifier with our custom head
    backbone.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_units),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate / 2),   # lighter second dropout
        nn.Linear(num_units, 1),          # single logit for binary output
    )
    return backbone

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# 4.  TRAINING HELPERS
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc   = accuracy_score(all_labels, preds)
    return epoch_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc   = accuracy_score(all_labels, preds)
    return epoch_loss, acc, np.array(all_labels), np.array(all_probs)

# ---------------------------------------------------------------------------
# 5.  EARLY STOPPING
# ---------------------------------------------------------------------------
class EarlyStopping:
    """
    Monitors validation loss. Saves the best model state.
    Triggers after `patience` epochs without improvement.
    """
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None
        self.triggered  = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# ---------------------------------------------------------------------------
# 6.  PHASE 1 – HEAD-ONLY TRAINING
# ---------------------------------------------------------------------------
def phase1(model, train_loader, val_loader, criterion, device, epochs):
    print("\n" + "="*60)
    print("PHASE 1  –  Head-only training  (backbone frozen)")
    print(f"  LR={LR_PHASE1}  |  Trainable params: {count_trainable(model):,}")
    print("="*60)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = copy.deepcopy(model.state_dict())

        print(f"  Ep {epoch:02d}/{epochs} | "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"{elapsed:.1f}s")

    # restore best weights from Phase 1
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  Phase 1 done. Best val loss: {best_val_loss:.4f}")

# ---------------------------------------------------------------------------
# 7.  PHASE 2 – FULL FINE-TUNING
# ---------------------------------------------------------------------------
def phase2(model, train_loader, val_loader, criterion, device, epochs, patience):
    print("\n" + "="*60)
    print("PHASE 2  –  Full fine-tuning  (all layers unfrozen)")
    print(f"  LR={LR_PHASE2}  |  Trainable params: {count_trainable(model):,}")
    print("  Scheduler: CosineAnnealingLR  |  Early stopping patience:", patience)
    print("="*60)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR_PHASE2,
        weight_decay=1e-5,
    )

    # CosineAnnealingLR: decays from LR_PHASE2 to eta_min over T_max epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-7,
    )

    early_stop = EarlyStopping(patience=patience, min_delta=1e-4)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        early_stop.step(vl_loss, model)

        print(f"  Ep {epoch:02d}/{epochs} | "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"LR={current_lr:.2e} | {elapsed:.1f}s"
              + (" [*]" if early_stop.counter == 0 else ""))

        if early_stop.triggered:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    early_stop.restore_best(model)
    print(f"\n  Phase 2 done. Best val loss: {early_stop.best_loss:.4f}")

# ---------------------------------------------------------------------------
# 8.  FINAL EVALUATION
# ---------------------------------------------------------------------------
def final_evaluation(model, test_loader, criterion, device, class_names):
    print("\n" + "="*60)
    print("FINAL EVALUATION on held-out TEST SET")
    print("="*60)

    _, _, labels, probs = evaluate(model, test_loader, criterion, device)
    preds = (probs >= 0.5).astype(int)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, probs)

    print(f"\n  Class mapping: {class_names}")
    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("="*60)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

# ---------------------------------------------------------------------------
# 9.  MAIN
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cpu")
    print(f"\n[Config] Device: {device}")
    print(f"[Config] theta = [d={DROPOUT_RATE}, eta1={LR_PHASE1}, "
          f"eta2={LR_PHASE2}, b={BATCH_SIZE}, u={NUM_UNITS}, w={CLASS_STRATEGY}]")

    # --- Data ---
    train_loader, val_loader, test_loader, class_names = build_loaders(DATA_ROOT, BATCH_SIZE)

    # --- Model ---
    model = build_model(NUM_UNITS, DROPOUT_RATE).to(device)

    # Loss: BCEWithLogitsLoss is numerically stable for binary tasks.
    # No pos_weight needed because the dataset is balanced.
    criterion = nn.BCEWithLogitsLoss()

    # -----------------------------------------------------------------------
    # PHASE 1: freeze backbone, train head only
    # -----------------------------------------------------------------------
    freeze_backbone(model)
    phase1(model, train_loader, val_loader, criterion, device, EPOCHS_PHASE1)

    # -----------------------------------------------------------------------
    # PHASE 2: unfreeze everything, fine-tune with cosine LR + early stopping
    # -----------------------------------------------------------------------
    unfreeze_all(model)
    phase2(model, train_loader, val_loader, criterion, device, EPOCHS_PHASE2, PATIENCE)

    # -----------------------------------------------------------------------
    # FINAL TEST EVALUATION
    # -----------------------------------------------------------------------
    metrics = final_evaluation(model, test_loader, criterion, device, class_names)

    # --- Save model ---
    save_path = "coffee_classifier_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[Saved] Model weights -> {save_path}")

    return metrics


if __name__ == "__main__":
    main()