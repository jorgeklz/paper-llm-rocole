# script.py

import os
import copy
import numpy as np
from collections import Counter

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
    roc_auc_score
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/splits/task_a_binary"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

IMG_SIZE = 224

BATCH_SIZE = 32
NUM_WORKERS = 0  # safer for CPU portability

PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 20

PHASE1_LR = 1e-3
PHASE2_LR = 1e-4

PHASE1_PATIENCE = 5
PHASE2_PATIENCE = 7

DROPOUT_RATE = 0.5
DENSE_UNITS = 256

DEVICE = torch.device("cpu")

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
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
        ratio=(0.3, 3.3),
        value='random'
    )
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=eval_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ============================================================
# CLASS WEIGHTING
# ============================================================

train_targets = train_dataset.targets
class_counts = Counter(train_targets)

num_samples = len(train_targets)
num_classes = len(class_counts)

class_weights = []

for cls_idx in range(num_classes):
    cls_count = class_counts[cls_idx]
    weight = num_samples / (num_classes * cls_count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print("\nClass Weights:", class_weights)

# ============================================================
# MODEL COMPONENTS
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

        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        b1 = self.branch3x3(x)
        b2 = self.branch5x5(x)

        x = torch.cat([b1, b2], dim=1)
        x = self.pool(x)

        return x


# ============================================================
# CUSTOM CNN
# ============================================================

class CoffeeLeafCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # Stage 1
        self.stage1 = ConvBlock(3, 32)

        # Stage 2
        self.stage2 = ConvBlock(32, 64)

        # Stage 3 (parallel)
        self.stage3 = ParallelConvBlock(64, 128)

        # Stage 4 (parallel)
        self.stage4 = ParallelConvBlock(128, 256)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, DENSE_UNITS),
            nn.ReLU(inplace=True),

            nn.Dropout(DROPOUT_RATE),

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


# ============================================================
# INITIALIZE MODEL
# ============================================================

model = CoffeeLeafCNN(num_classes=2).to(DEVICE)

# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ============================================================
# UTILITIES
# ============================================================

def freeze_feature_extractor(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


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
# PHASE 1
# ============================================================

print("\n===================================================")
print("PHASE 1 - TRAIN CLASSIFICATION HEAD")
print("===================================================\n")

freeze_feature_extractor(model)

optimizer_phase1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=PHASE1_LR
)

early_stopping_1 = EarlyStopping(patience=PHASE1_PATIENCE)

for epoch in range(PHASE1_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer_phase1,
        criterion
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion
    )

    print(
        f"[Phase 1][Epoch {epoch+1}/{PHASE1_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    stop = early_stopping_1.step(val_loss, model)

    if stop:
        print("\nEarly stopping triggered in Phase 1.")
        break

model.load_state_dict(early_stopping_1.best_weights)

# ============================================================
# PHASE 2
# ============================================================

print("\n===================================================")
print("PHASE 2 - FINE TUNING ENTIRE NETWORK")
print("===================================================\n")

unfreeze_all_layers(model)

optimizer_phase2 = optim.Adam(
    model.parameters(),
    lr=PHASE2_LR
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_phase2,
    T_max=PHASE2_EPOCHS,
    eta_min=1e-6
)

early_stopping_2 = EarlyStopping(patience=PHASE2_PATIENCE)

for epoch in range(PHASE2_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer_phase2,
        criterion
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion
    )

    scheduler.step()

    current_lr = optimizer_phase2.param_groups[0]['lr']

    print(
        f"[Phase 2][Epoch {epoch+1}/{PHASE2_EPOCHS}] "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    stop = early_stopping_2.step(val_loss, model)

    if stop:
        print("\nEarly stopping triggered in Phase 2.")
        break

model.load_state_dict(early_stopping_2.best_weights)

# ============================================================
# TEST EVALUATION
# ============================================================

print("\n===================================================")
print("FINAL TEST EVALUATION")
print("===================================================\n")

model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:, 1]

        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nClass Mapping:")
print(train_dataset.class_to_idx)

# ============================================================
# HYPERPARAMETER JUSTIFICATION
# ============================================================

print("\n===================================================")
print("HYPERPARAMETER JUSTIFICATION")
print("===================================================\n")

print(
    "Dropout Rate (0.5): "
    "Chosen to reduce overfitting in the dense classifier "
    "while preserving representation capacity."
)

print(
    "Learning Rates: "
    "1e-3 for head training accelerates convergence of randomly "
    "initialized dense layers. 1e-4 during fine-tuning stabilizes "
    "gradient updates across the entire CNN."
)

print(
    "Batch Size (32): "
    "Provides a strong trade-off between gradient stability, "
    "CPU memory efficiency, and training speed."
)

print(
    "Dense Units (256): "
    "Large enough to learn discriminative leaf representations "
    "after global pooling without excessive parameter growth."
)