# script.py
# ------------------------------------------------------------
# Deep CNN training for robusta coffee leaf classification
# Classes:
#   - healthy
#   - red_spider_mite
#   - coffee_leaf_rust
#
# Progressive transfer learning strategy:
#   Phase 1 -> train classifier head only
#   Phase 2 -> unfreeze last residual block
#   Phase 3 -> fine-tune entire network
#
# CPU-only execution
#
# Run:
#   python script.py
# ------------------------------------------------------------

import os
import copy
import random
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

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(os.cpu_count())

DEVICE = torch.device("cpu")


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_ROOT = "data/splits/task_b_3class"

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 16

PHASE1_EPOCHS = 8
PHASE2_EPOCHS = 8
PHASE3_EPOCHS = 10

LR_HEAD = 1e-3
LR_STAGE2 = 5e-4
LR_STAGE3 = 1e-4

WEIGHT_DECAY = 1e-4

NUM_CLASSES = 3


# ------------------------------------------------------------
# ImageNet normalization
# ------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------
# Data augmentation
# ------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

class_names = train_dataset.classes

print("\nClass mapping:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")


# ------------------------------------------------------------
# Handle class imbalance
# WeightedRandomSampler
# ------------------------------------------------------------
targets = train_dataset.targets

class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts

sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
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


# ------------------------------------------------------------
# Model
# Standard and effective approach:
# ResNet50 pretrained on ImageNet
# ------------------------------------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)


# ------------------------------------------------------------
# Freeze / unfreeze helpers
# ------------------------------------------------------------
def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_layer4(model):
    for param in model.layer4.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ------------------------------------------------------------
# Loss function
# Weighted CrossEntropy
# ------------------------------------------------------------
class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer):
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

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss


# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------
def evaluate(model, loader):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            running_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

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

    try:
        auc = roc_auc_score(
            all_labels,
            np.array(all_probs),
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        auc = float("nan")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_recall": per_class_recall,
        "auc_ovr_macro": auc
    }


# ------------------------------------------------------------
# Generic phase trainer
# ------------------------------------------------------------
def run_training_phase(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    phase_name
):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    print(f"\n{'='*60}")
    print(f"{phase_name}")
    print(f"{'='*60}")

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_metrics = evaluate(model, val_loader)

        val_f1 = val_metrics["macro_f1"]

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

    model.load_state_dict(best_model_wts)

    return model


# ------------------------------------------------------------
# PHASE 1
# Freeze backbone
# Train classifier only
# ------------------------------------------------------------
freeze_all_layers(model)

for param in model.fc.parameters():
    param.requires_grad = True

optimizer_phase1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_HEAD,
    weight_decay=WEIGHT_DECAY
)

model = run_training_phase(
    model,
    train_loader,
    val_loader,
    optimizer_phase1,
    PHASE1_EPOCHS,
    "PHASE 1 - CLASSIFIER HEAD TRAINING"
)


# ------------------------------------------------------------
# PHASE 2
# Unfreeze layer4
# ------------------------------------------------------------
unfreeze_layer4(model)

optimizer_phase2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_STAGE2,
    weight_decay=WEIGHT_DECAY
)

model = run_training_phase(
    model,
    train_loader,
    val_loader,
    optimizer_phase2,
    PHASE2_EPOCHS,
    "PHASE 2 - LAST BLOCK FINE-TUNING"
)


# ------------------------------------------------------------
# PHASE 3
# Full fine-tuning
# ------------------------------------------------------------
unfreeze_all(model)

optimizer_phase3 = optim.Adam(
    model.parameters(),
    lr=LR_STAGE3,
    weight_decay=WEIGHT_DECAY
)

model = run_training_phase(
    model,
    train_loader,
    val_loader,
    optimizer_phase3,
    PHASE3_EPOCHS,
    "PHASE 3 - FULL NETWORK FINE-TUNING"
)


# ------------------------------------------------------------
# Final test evaluation
# ------------------------------------------------------------
print(f"\n{'='*60}")
print("FINAL TEST EVALUATION")
print(f"{'='*60}")

test_metrics = evaluate(model, test_loader)

print(f"\nTest Loss: {test_metrics['loss']:.4f}")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
print(f"Macro One-vs-Rest AUC: {test_metrics['auc_ovr_macro']:.4f}")

print("\nPer-class Recall:")

for class_name, recall_value in zip(
    class_names,
    test_metrics["per_class_recall"]
):
    print(f"{class_name}: {recall_value:.4f}")

print("\nTraining completed successfully.")