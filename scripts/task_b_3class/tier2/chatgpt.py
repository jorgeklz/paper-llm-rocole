# script.py

import os
import copy
import math
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/splits/task_b_3class"

IMG_SIZE = 224

BATCH_SIZE = 32
NUM_CLASSES = 3

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 30

LR_PHASE1 = 5e-4
LR_PHASE2 = 1e-5

DROPOUT = 0.3
DENSE_UNITS = 128

PATIENCE = 7
MIN_DELTA = 0.001

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "coffee_leaf_rust",
    "healthy",
    "red_spider_mite"
]


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
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

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):

        ce_loss = self.ce(logits, targets)

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# ============================================================
# CLASS WEIGHTS FOR FOCAL LOSS
# ============================================================

class_counts = np.array([602, 791, 167], dtype=np.float32)

weights = 1.0 / class_counts
weights = weights / weights.sum()
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

criterion = FocalLoss(alpha=weights, gamma=2.0)


# ============================================================
# MODEL
# ============================================================

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_features, DENSE_UNITS),
    nn.ReLU(),
    nn.Dropout(DROPOUT),
    nn.Linear(DENSE_UNITS, NUM_CLASSES)
)

model = model.to(DEVICE)


# ============================================================
# FREEZE BACKBONE (PHASE 1)
# ============================================================

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True


# ============================================================
# OPTIMIZER PHASE 1
# ============================================================

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LR_PHASE1
)


# ============================================================
# TRAINING UTILITIES
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):

    model.eval()

    running_loss = 0.0

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            running_loss += loss.item() * images.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return avg_loss, macro_f1


# ============================================================
# EARLY STOPPING
# ============================================================

best_model_wts = copy.deepcopy(model.state_dict())

best_f1 = -1.0
patience_counter = 0


# ============================================================
# PHASE 1 TRAINING
# ============================================================

print("\n========== PHASE 1 ==========")

for epoch in range(PHASE1_EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_f1 = evaluate(
        model,
        val_loader,
        criterion
    )

    print(
        f"Epoch {epoch+1}/{PHASE1_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Macro-F1: {val_f1:.4f}"
    )


# ============================================================
# UNFREEZE ENTIRE NETWORK (PHASE 2)
# ============================================================

for param in model.parameters():
    param.requires_grad = True


optimizer = optim.Adam(
    model.parameters(),
    lr=LR_PHASE2
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,
    eta_min=1e-7
)


# ============================================================
# PHASE 2 TRAINING
# ============================================================

print("\n========== PHASE 2 ==========")

for epoch in range(PHASE2_EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_f1 = evaluate(
        model,
        val_loader,
        criterion
    )

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{PHASE2_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Macro-F1: {val_f1:.4f}"
    )

    # Early stopping on macro-F1

    if val_f1 > best_f1 + MIN_DELTA:

        best_f1 = val_f1
        best_model_wts = copy.deepcopy(model.state_dict())

        patience_counter = 0

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(best_model_wts)


# ============================================================
# TEST EVALUATION
# ============================================================

model.eval()

y_true = []
y_pred = []
y_prob = []

test_loss = 0.0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        test_loss += loss.item() * images.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

test_loss /= len(test_loader.dataset)

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average='macro'
)

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

per_class_recall = recall_score(
    y_true,
    y_pred,
    average=None
)

auc = roc_auc_score(
    y_true,
    np.array(y_prob),
    multi_class='ovr',
    average='macro'
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n========== TEST RESULTS ==========")

print(f"Test Loss           : {test_loss:.4f}")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Macro-F1            : {macro_f1:.4f}")
print(f"Balanced Accuracy   : {balanced_acc:.4f}")
print(f"Macro OVR AUC       : {auc:.4f}")

print("\nPer-class Recall:")

for idx, recall in enumerate(per_class_recall):
    print(f"{CLASS_NAMES[idx]} : {recall:.4f}")