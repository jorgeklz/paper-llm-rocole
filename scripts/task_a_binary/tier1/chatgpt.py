# script.py
# ============================================================
# Binary classification of robusta coffee leaf images
# Classes:
#   - healthy
#   - unhealthy
#
# Features:
#   - ImageFolder loading
#   - ImageNet normalization
#   - Progressive transfer learning:
#         Stage 1 -> train classifier head only
#         Stage 2 -> unfreeze last CNN block
#         Stage 3 -> fine-tune entire network
#   - CPU-only training
#   - Final evaluation on test set:
#         Accuracy, Precision, Recall, F1, ROC-AUC
#
# Run:
#   python script.py
# ============================================================

import copy
import os
import time
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data/splits/task_a_binary"

BATCH_SIZE = 16
NUM_WORKERS = 0  # safer for Windows/CPU

IMAGE_SIZE = 224

# Progressive training schedule
STAGE1_EPOCHS = 5   # train classifier only
STAGE2_EPOCHS = 5   # unfreeze last block
STAGE3_EPOCHS = 5   # fine-tune all layers

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-4

SEED = 42

DEVICE = torch.device("cpu")

# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# TRANSFORMS
# ============================================================

# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"),
    transform=eval_transform
)

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

class_names = train_dataset.classes

print("Classes:", class_names)
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))

# ============================================================
# MODEL
# ============================================================

# ResNet18 is a strong baseline for moderate datasets
# Lightweight and effective on CPU.
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace final classifier
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 1)
)

model = model.to(DEVICE)

# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()

# ============================================================
# UTILITIES
# ============================================================

def set_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(outputs) >= 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_probs = []
    all_preds = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()

        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.cpu().numpy().flatten())
        all_probs.extend(probs.cpu().numpy().flatten())
        all_preds.extend(preds.cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)

    return (
        epoch_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds)
    )


def run_training_stage(
    stage_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs
):
    print("\n" + "=" * 60)
    print(f"{stage_name}")
    print("=" * 60)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(epochs):

        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, _, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        elapsed = time.time() - start

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    return model


# ============================================================
# STAGE 1
# Freeze backbone, train classifier head only
# ============================================================

set_requires_grad(model, False)

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE_HEAD
)

model = run_training_stage(
    stage_name="STAGE 1: Train classifier head",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=STAGE1_EPOCHS
)

# ============================================================
# STAGE 2
# Unfreeze last residual block
# ============================================================

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE_FINE
)

model = run_training_stage(
    stage_name="STAGE 2: Fine-tune last ResNet block",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=STAGE2_EPOCHS
)

# ============================================================
# STAGE 3
# Fine-tune entire network
# ============================================================

set_requires_grad(model, True)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE_FINE
)

model = run_training_stage(
    stage_name="STAGE 3: Fine-tune entire network",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=STAGE3_EPOCHS
)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_loss, test_acc, y_true, y_probs, y_pred = evaluate(
    model,
    test_loader,
    criterion
)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Accuracy      : {test_acc:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"ROC-AUC       : {auc:.4f}")

# ============================================================
# OPTIONAL: SAVE MODEL
# ============================================================

torch.save(model.state_dict(), "coffee_leaf_resnet18.pth")

print("\nModel saved to: coffee_leaf_resnet18.pth")