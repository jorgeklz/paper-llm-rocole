# script.py

import os
import copy
import time
import random
import numpy as np

from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Device (CPU ONLY)
# ============================================================

device = torch.device("cpu")

# ============================================================
# Paths
# ============================================================

DATA_DIR = "data/splits/task_b_3class"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# ============================================================
# Hyperparameters
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Phase 1
PHASE1_EPOCHS = 15
PHASE1_LR = 1e-3
PHASE1_PATIENCE = 5

# Phase 2
PHASE2_EPOCHS = 20
PHASE2_LR = 1e-4
PHASE2_PATIENCE = 7

WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
DENSE_UNITS = 256

NUM_CLASSES = 3

# ============================================================
# Justification of selected hyperparameters
# ============================================================

"""
1. Batch size = 32
   - Good compromise between gradient stability and CPU memory usage.
   - Suitable for medium-sized datasets (~1560 images).

2. Dropout = 0.5
   - Strong regularization for a CNN trained from scratch.
   - Reduces overfitting risk caused by limited dataset size.

3. Dense units = 256
   - Sufficient representational capacity after global average pooling.
   - Avoids excessive parameters before final classification.

4. Learning rates:
   - Phase 1 = 1e-3
     Higher LR is appropriate because only the classifier head is trained.
   - Phase 2 = 1e-4
     Fine-tuning entire CNN requires smaller updates to preserve learned
     low-level representations and stabilize convergence.

5. Class imbalance handling:
   - Weighted CrossEntropyLoss is used instead of focal loss.
   - The imbalance is MODERATE, not extreme.
   - Weighted CE preserves gradient contribution from minority class
     red_spider_mite while remaining stable and interpretable.
   - It improves minority recall without making optimization unstable.
"""

# ============================================================
# Data Augmentation and Normalization
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3)
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# Datasets and DataLoaders
# ============================================================

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = train_dataset.classes

print("\nClasses:")
for idx, cls in enumerate(class_names):
    print(f"{idx}: {cls}")

# ============================================================
# Class Imbalance Handling
# ============================================================

targets = train_dataset.targets
class_counts = Counter(targets)

print("\nTraining class distribution:")
for cls_idx, count in class_counts.items():
    print(f"{class_names[cls_idx]}: {count}")

# Inverse-frequency weighting
total_samples = sum(class_counts.values())

class_weights = []

for i in range(NUM_CLASSES):
    weight = total_samples / (NUM_CLASSES * class_counts[i])
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float32)

print("\nClass weights:")
for i, w in enumerate(class_weights):
    print(f"{class_names[i]}: {w:.4f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ============================================================
# CNN Building Blocks
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class ParallelConvBlock(nn.Module):
    """
    Parallel 3x3 and 5x5 paths.
    Used ONLY in stages 3 and 4.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 2

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        b1 = self.branch3x3(x)
        b2 = self.branch5x5(x)

        x = torch.cat([b1, b2], dim=1)
        x = self.fusion(x)

        return x

# ============================================================
# Custom Deep CNN
# ============================================================

class CoffeeLeafCNN(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        # Stage 1: 32 filters
        self.stage1 = ConvBlock(3, 32)

        # Stage 2: 64 filters
        self.stage2 = ConvBlock(32, 64)

        # Stage 3: parallel paths
        self.stage3 = ParallelConvBlock(64, 128)

        # Stage 4: parallel paths
        self.stage4 = ParallelConvBlock(128, 256)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, DENSE_UNITS),
            nn.ReLU(inplace=True),

            nn.Dropout(DROPOUT),

            nn.Linear(DENSE_UNITS, num_classes)
        )

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)

        x = self.classifier(x)

        return x


model = CoffeeLeafCNN(num_classes=NUM_CLASSES).to(device)

print("\nModel architecture:")
print(model)

# ============================================================
# Freeze / Unfreeze Utilities
# ============================================================

def freeze_feature_extractor(model):
    for name, param in model.named_parameters():

        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:

    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None

    def step(self, val_loss, model):

        if val_loss < self.best_loss:

            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())

            return False

        else:

            self.counter += 1

            if self.counter >= self.patience:
                return True

            return False

# ============================================================
# Training Functions
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss


def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss

# ============================================================
# Phase 1
# ============================================================

print("\n====================================================")
print("PHASE 1 - Training classification head only")
print("====================================================")

freeze_feature_extractor(model)

optimizer_phase1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=PHASE1_LR,
    weight_decay=WEIGHT_DECAY
)

early_stopping = EarlyStopping(patience=PHASE1_PATIENCE)

for epoch in range(PHASE1_EPOCHS):

    start = time.time()

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer_phase1
    )

    val_loss = validate(
        model,
        val_loader,
        criterion
    )

    elapsed = time.time() - start

    print(
        f"Epoch [{epoch+1}/{PHASE1_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    stop = early_stopping.step(val_loss, model)

    if stop:
        print("\nEarly stopping triggered in Phase 1.")
        break

model.load_state_dict(early_stopping.best_weights)

# ============================================================
# Phase 2
# ============================================================

print("\n====================================================")
print("PHASE 2 - Fine-tuning entire network")
print("====================================================")

unfreeze_all(model)

optimizer_phase2 = optim.Adam(
    model.parameters(),
    lr=PHASE2_LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_phase2,
    T_max=PHASE2_EPOCHS
)

early_stopping = EarlyStopping(patience=PHASE2_PATIENCE)

for epoch in range(PHASE2_EPOCHS):

    start = time.time()

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer_phase2
    )

    val_loss = validate(
        model,
        val_loader,
        criterion
    )

    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]

    elapsed = time.time() - start

    print(
        f"Epoch [{epoch+1}/{PHASE2_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"LR: {current_lr:.6f} | "
        f"Time: {elapsed:.1f}s"
    )

    stop = early_stopping.step(val_loss, model)

    if stop:
        print("\nEarly stopping triggered in Phase 2.")
        break

model.load_state_dict(early_stopping.best_weights)

# ============================================================
# Test Evaluation
# ============================================================

print("\n====================================================")
print("FINAL TEST EVALUATION")
print("====================================================")

model.eval()

all_labels = []
all_preds = []
all_probs = []

test_loss = 0.0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        test_loss += loss.item() * images.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert to numpy
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ============================================================
# Metrics
# ============================================================

test_loss = test_loss / len(test_loader.dataset)

accuracy = accuracy_score(all_labels, all_preds)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average="macro"
)

balanced_acc = balanced_accuracy_score(
    all_labels,
    all_preds
)

per_class_recall = recall_score(
    all_labels,
    all_preds,
    average=None
)

auc_macro_ovr = roc_auc_score(
    all_labels,
    all_probs,
    multi_class="ovr",
    average="macro"
)

# ============================================================
# Results
# ============================================================

print(f"\nTest Loss           : {test_loss:.4f}")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Macro F1            : {macro_f1:.4f}")
print(f"Balanced Accuracy   : {balanced_acc:.4f}")
print(f"Macro AUC (OvR)     : {auc_macro_ovr:.4f}")

print("\nPer-class Recall:")

for idx, recall in enumerate(per_class_recall):
    print(f"{class_names[idx]:20s}: {recall:.4f}")

print("\nDone.")