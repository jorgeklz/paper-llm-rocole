import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score
)

# =============================================================================
# HYPERPARAMETERS & JUSTIFICATIONS
# =============================================================================
# Batch Size: 32
# Justification: Balances gradient stability with CPU memory constraints. 
# Provides a reliable noise-to-signal ratio for ~1500 training samples.
BATCH_SIZE = 32

# Learning Rates
# Phase 1 LR: 1e-3. Standard starting point for training a randomly initialized 
# classification head. Ensures fast convergence without gradient explosion.
# Phase 2 LR: 1e-4. 10x smaller than P1. Required to gently adjust feature 
# extractors without destroying the representations learned in P1.
LR_P1 = 1e-3
LR_P2 = LR_P1 / 10

# Dropout Rate: 0.3
# Justification: Moderate dropout in the dense head prevents feature co-adaptation 
# while preserving gradient flow. Optimal for the 256 -> 128 -> 2 bottleneck.
DROPOUT_RATE = 0.3

# Dense Layer Units: 128
# Justification: Acts as a bottleneck after GAP's 256-dim output. Forces the 
# network to compress spatial features into discriminative latent representations 
# before classification, reducing overfitting on small datasets.
DENSE_UNITS = 128

# Training Strategy Config
EPOCHS_P1, PATIENCE_P1 = 15, 5
EPOCHS_P2, PATIENCE_P2 = 20, 7

# Force CPU execution as requested
DEVICE = torch.device("cpu")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class CoffeeLeafCNN(nn.Module):
    def __init__(self, dropout_rate=0.3, dense_units=128):
        super().__init__()
        
        # Stage 1: 3 -> 32
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 32 -> 64
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 64 -> 128 (Parallel Paths 3x3 & 5x5)
        # Each branch outputs 64 channels, concatenated to 128
        self.stage3_path3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage3_path5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Stage 4: 128 -> 256 (Parallel Paths 3x3 & 5x5)
        # Each branch outputs 128 channels, concatenated to 256
        self.stage4_path3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage4_path5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling & Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, 1)
        )
        
    def forward(self, x):
        # Stage 1 & 2
        x = self.stage1(x)
        x = self.stage2(x)
        
        # Stage 3: Parallel Convolution & Concatenation
        x3 = self.stage3_path3(x)
        x5 = self.stage3_path5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.stage3_pool(x)
        
        # Stage 4: Parallel Convolution & Concatenation
        x4_3 = self.stage4_path3(x)
        x4_5 = self.stage4_path5(x)
        x = torch.cat([x4_3, x4_5], dim=1)
        x = self.stage4_pool(x)
        
        # GAP & Head
        x = self.gap(x)
        logits = self.head(x)
        return logits.squeeze(-1) # Shape: (N,) for binary BCE

# =============================================================================
# DATA PREPARATION
# =============================================================================
def get_datasets_and_loaders(base_dir, batch_size):
    # Training Augmentation Pipeline
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test Pipeline
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(base_dir, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(base_dir, "test"),  transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Compute class weights for imbalance handling
    targets = torch.tensor(train_dataset.targets)
    pos_count = (targets == 1).float().sum()
    neg_count = (targets == 0).float().sum()
    # pos_weight scales loss contribution of the minority class
    pos_weight = neg_count / pos_count
    
    return train_loader, val_loader, test_loader, pos_weight

# =============================================================================
# TRAINING & EVALUATION HELPERS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))
            
    val_loss = running_loss / len(loader.dataset)
    # Early metrics for logging (not used for loss tracking)
    val_acc = accuracy_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))
    return val_loss, val_acc

def evaluate_test(model, loader, device):
    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int).flatten()
            
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    return accuracy, precision, recall, f1, auc

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    DATA_DIR = "data/splits/task_a_binary"
    
    # 1. Initialize Data & Loss
    train_loader, val_loader, test_loader, pos_weight = get_datasets_and_loaders(DATA_DIR, BATCH_SIZE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = CoffeeLeafCNN(dropout_rate=DROPOUT_RATE, dense_units=DENSE_UNITS).to(DEVICE)
    
    # ======================================================================
    # PHASE 1: Train Classification Head Only (Frozen Convs)
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 1: Training Classification Head (Frozen Backbone)")
    print("="*60)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P1, weight_decay=1e-4)
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss, patience = float('inf'), 0
    
    for epoch in range(1, EPOCHS_P1 + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch}/{EPOCHS_P1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE_P1:
                print(f"Early stopping at epoch {epoch}")
                break
                
    model.load_state_dict(best_wts)
    print("Phase 1 Complete. Best validation weights restored.")
    
    # ======================================================================
    # PHASE 2: Unfreeze & Fine-tune with Cosine LR Decay
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 2: Full Model Fine-tuning (Unfrozen)")
    print("="*60)
    
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(model.parameters(), lr=LR_P2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_P2, eta_min=1e-6)
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss, patience = float('inf'), 0
    
    for epoch in range(1, EPOCHS_P2 + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch}/{EPOCHS_P2} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE_P2:
                print(f"Early stopping at epoch {epoch}")
                break
                
    model.load_state_dict(best_wts)
    print("Phase 2 Complete. Best validation weights restored.")
    
    # ======================================================================
    # FINAL TEST EVALUATION
    # ======================================================================
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    test_acc, test_prec, test_rec, test_f1, test_auc = evaluate_test(model, test_loader, DEVICE)
    
    print(f"Final Test Accuracy : {test_acc:.4f}")
    print(f"Final Test Precision: {test_prec:.4f}")
    print(f"Final Test Recall   : {test_rec:.4f}")
    print(f"Final Test F1-Score : {test_f1:.4f}")
    print(f"Final Test AUC-ROC  : {test_auc:.4f}")