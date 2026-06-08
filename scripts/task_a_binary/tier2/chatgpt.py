# script.py

import os
import copy
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data/splits/task_a_binary"

BATCH_SIZE = 32
NUM_CLASSES = 2

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20

LR_PHASE1 = 5e-4
LR_PHASE2 = 1e-5

DROPOUT = 0.3
DENSE_UNITS = 128

PATIENCE = 5
MIN_DELTA = 1e-4

IMAGE_SIZE = 224

DEVICE = torch.device("cpu")


# ============================================================
# Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Datasets
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
# Model
# ============================================================

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(in_features, DENSE_UNITS),
    nn.ReLU(),
    nn.Dropout(DROPOUT),
    nn.Linear(DENSE_UNITS, 1)
)

model = model.to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# Utilities
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    running_loss = 0.0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss


def evaluate_loss(model, loader, criterion):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss


def evaluate_metrics(model, loader):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            preds = (probs >= 0.5).astype(int)

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    return acc, precision, recall, f1, auc


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:

    def __init__(self, patience=5, min_delta=1e-4):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = np.inf
        self.counter = 0

        self.best_weights = None

    def step(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:

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
# Phase 1
# Freeze backbone
# ============================================================

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

optimizer_phase1 = optim.Adam(
    model.fc.parameters(),
    lr=LR_PHASE1
)

print("\n==============================")
print("PHASE 1: TRAINING CLASSIFIER")
print("==============================")

for epoch in range(PHASE1_EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer_phase1,
        criterion
    )

    val_loss = evaluate_loss(
        model,
        val_loader,
        criterion
    )

    print(
        f"Epoch [{epoch+1}/{PHASE1_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


# ============================================================
# Phase 2
# Fine-tune entire network
# ============================================================

for param in model.parameters():
    param.requires_grad = True

optimizer_phase2 = optim.Adam(
    model.parameters(),
    lr=LR_PHASE2
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_phase2,
    T_max=15,
    eta_min=1e-6
)

early_stopping = EarlyStopping(
    patience=PATIENCE,
    min_delta=MIN_DELTA
)

print("\n==============================")
print("PHASE 2: FINE-TUNING")
print("==============================")

for epoch in range(PHASE2_EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer_phase2,
        criterion
    )

    val_loss = evaluate_loss(
        model,
        val_loader,
        criterion
    )

    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch [{epoch+1}/{PHASE2_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"LR: {current_lr:.8f}"
    )

    stop = early_stopping.step(val_loss, model)

    if stop:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# Restore best model
# ============================================================

model.load_state_dict(early_stopping.best_weights)


# ============================================================
# Final Test Evaluation
# ============================================================

acc, precision, recall, f1, auc = evaluate_metrics(
    model,
    test_loader
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")