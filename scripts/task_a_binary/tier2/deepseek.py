#!/usr/bin/env python3
"""
Deep CNN for Robusta Coffee Leaf Disease Classification
Two-phase training with optimal hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
from copy import deepcopy

# ==================== CONFIGURATION ====================
# Optimal hyperparameters theta = [d, eta_1, eta_2, b, u, w]
DROPOUT_RATE = 0.3
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
BATCH_SIZE = 32
DENSE_UNITS = 128
USE_CLASS_WEIGHTS = True  # w = class_weights

# Paths (adjust if different)
DATA_ROOT = 'data/splits/task_a_binary'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# Training settings
NUM_EPOCHS_PHASE1 = 20
NUM_EPOCHS_PHASE2 = 30
PATIENCE = 7
IMG_SIZE = 224
DEVICE = torch.device('cpu')  # Force CPU usage

# ==================== DATA LOADING ====================
def get_transforms():
    """Image transforms with augmentation for training"""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_class_weights(train_dataset):
    """Compute class weights for imbalance handling"""
    class_counts = []
    for idx in range(len(train_dataset.classes)):
        class_counts.append(len(train_dataset.targets) - 
                           sum([1 for t in train_dataset.targets if t != idx]))
    
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    
    sample_weights = [weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

# Load datasets
train_transform, val_transform = get_transforms()
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

# Data loaders with class balancing if enabled
if USE_CLASS_WEIGHTS:
    sampler = get_class_weights(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            sampler=sampler, num_workers=0)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=0)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                       shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)} (healthy: {train_dataset.targets.count(0)}, "
      f"unhealthy: {train_dataset.targets.count(1)})")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ==================== MODEL ARCHITECTURE ====================
class CustomCNN(nn.Module):
    """Custom CNN with frozen feature extractor and custom head"""
    def __init__(self, dropout_rate=0.3, dense_units=128):
        super(CustomCNN, self).__init__()
        # Use pretrained ResNet18 as feature extractor
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove original classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1)  # Binary classification
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(1)

# Initialize model
model = CustomCNN(dropout_rate=DROPOUT_RATE, dense_units=DENSE_UNITS)
print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    return total_loss / len(loader), acc, precision, recall, f1, auc

def early_stopping(val_loss, best_loss, patience_counter, patience, model, best_model):
    """Early stopping logic"""
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        best_model = deepcopy(model.state_dict())
    else:
        patience_counter += 1
    
    return best_loss, patience_counter, best_model

# ==================== PHASE 1: FROZEN BACKBONE ====================
print("\n" + "="*50)
print("PHASE 1: Training classification head only")
print("="*50)

# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Loss function with class weights if enabled
if USE_CLASS_WEIGHTS:
    # Compute weights based on training set
    n_healthy = train_dataset.targets.count(0)
    n_unhealthy = train_dataset.targets.count(1)
    pos_weight = torch.tensor([n_unhealthy / n_healthy])  # Weight for positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using class weights: pos_weight={pos_weight.item():.3f}")
else:
    criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LR_PHASE1)

best_val_loss = float('inf')
best_model_state = None
patience_counter = 0

for epoch in range(NUM_EPOCHS_PHASE1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion, DEVICE)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS_PHASE1} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Early stopping
    best_val_loss, patience_counter, best_model_state = early_stopping(
        val_loss, best_val_loss, patience_counter, PATIENCE, model, best_model_state
    )
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Restore best model
model.load_state_dict(best_model_state)

# ==================== PHASE 2: FINE-TUNE ENTIRE NETWORK ====================
print("\n" + "="*50)
print("PHASE 2: Fine-tuning entire network")
print("="*50)

# Unfreeze backbone
for param in model.backbone.parameters():
    param.requires_grad = True

# Lower learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=LR_PHASE2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_PHASE2, eta_min=1e-6)

best_val_loss = float('inf')
best_model_state = None
patience_counter = 0

for epoch in range(NUM_EPOCHS_PHASE2):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion, DEVICE)
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS_PHASE2} - LR: {current_lr:.6f} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
          f"F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
    
    # Early stopping
    best_val_loss, patience_counter, best_model_state = early_stopping(
        val_loss, best_val_loss, patience_counter, PATIENCE, model, best_model_state
    )
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Restore best model
model.load_state_dict(best_model_state)

# ==================== FINAL EVALUATION ON TEST SET ====================
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
    model, test_loader, criterion, DEVICE
)

print(f"\nTest Set Performance:")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"AUC:       {test_auc:.4f}")

# ==================== SAVE MODEL ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': {
        'dropout_rate': DROPOUT_RATE,
        'dense_units': DENSE_UNITS,
        'lr_phase1': LR_PHASE1,
        'lr_phase2': LR_PHASE2,
        'batch_size': BATCH_SIZE,
        'class_weight_strategy': 'class_weights'
    },
    'test_metrics': {
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'auc': test_auc
    }
}, 'coffee_leaf_model.pth')

print("\nModel saved as 'coffee_leaf_model.pth'")