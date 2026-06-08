import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, 
                             recall_score, roc_auc_score)

# ==============================================================================
# CONFIGURATION & DEVICE
# ==============================================================================
DEVICE = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = 'data/splits/task_b_3class'
BATCH_SIZE = 32
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 20
PHASE1_PATIENCE = 5
PHASE2_PATIENCE = 7
PHASE1_LR = 1e-3
PHASE2_LR = PHASE1_LR / 10.0

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
class RobustaCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, dense_units=128, num_classes=3):
        super().__init__()
        # Stage 1: 3 -> 32
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 32 -> 64
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: Parallel paths (3x3 & 5x5) -> Concat -> 128
        self.s3_path3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.s3_path5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.s3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Stage 4: Parallel paths (3x3 & 5x5) -> Concat -> 256
        self.s4_path3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.s4_path5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.s4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, num_classes)
        )
        
    def _get_conv_modules(self):
        """Helper to collect all convolutional parameters for freezing/unfreezing."""
        modules = []
        modules.extend(list(self.stage1.parameters()))
        modules.extend(list(self.stage2.parameters()))
        modules.extend(list(self.s3_path3.parameters()) + list(self.s3_path5.parameters()))
        modules.extend(list(self.s4_path3.parameters()) + list(self.s4_path5.parameters()))
        return modules

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        
        # Stage 3 parallel
        x3_a = self.s3_path3(x)
        x3_b = self.s3_path5(x)
        x = torch.cat([x3_a, x3_b], dim=1)  # -> 128 channels
        x = self.s3_pool(x)
        
        # Stage 4 parallel
        x4_a = self.s4_path3(x)
        x4_b = self.s4_path5(x)
        x = torch.cat([x4_a, x4_b], dim=1)  # -> 256 channels
        x = self.s4_pool(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ==============================================================================
# DATA LOADING & AUGMENTATION
# ==============================================================================
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Class weighting for moderate imbalance
class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=3).float()
total_samples = sum(class_counts)
class_weights = total_samples / (3.0 * class_counts)  # Inverse frequency normalized
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None
        self.min_delta = min_delta
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  Early stopping triggered at patience {self.counter}/{self.patience}")

def evaluate(model, loader, criterion, return_probs=False):
    model.eval()
    all_labels, all_preds, all_probs, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    avg_loss = total_loss / len(loader.dataset)
    
    if return_probs:
        return avg_loss, y_true, y_pred, y_prob
    return avg_loss, y_true, y_pred

def compute_metrics(y_true, y_pred, y_prob, avg_loss):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    per_class_recall = recall_score(y_true, y_pred, average=None)
    try:
        auc_ovr_macro = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
    except ValueError:
        auc_ovr_macro = 0.0
    return {
        'accuracy': acc,
        'macro_f1': f1_macro,
        'balanced_accuracy': bal_acc,
        'test_loss': avg_loss,
        'per_class_recall': per_class_recall,
        'macro_auc_ovr': auc_ovr_macro
    }

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    model = RobustaCNN(dropout_rate=0.5, dense_units=128).to(DEVICE)
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Class weights applied: {class_weights.tolist()}")
    print("-" * 60)

    # ==================== PHASE 1 ====================
    print("\n[PHASE 1] Freezing conv layers, training classifier head only...")
    for param in model._get_conv_modules():
        param.requires_grad = False
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    es = EarlyStopping(patience=PHASE1_PATIENCE)
    
    for epoch in range(PHASE1_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataset)
        scheduler.step()
            
        val_loss, _, _ = evaluate(model, val_loader, criterion)
        print(f"  Epoch {epoch+1}/{PHASE1_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        es(val_loss, model)
        if es.early_stop: break
        
    print("Phase 1 completed. Loading best model state...")
    model.load_state_dict(es.best_state)

    # ==================== PHASE 2 ====================
    print("\n[PHASE 2] Unfreezing all layers, fine-tuning with cosine annealing...")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=PHASE2_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)
    es = EarlyStopping(patience=PHASE2_PATIENCE)
    
    for epoch in range(PHASE2_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataset)
        scheduler.step()
            
        val_loss, _, _ = evaluate(model, val_loader, criterion)
        print(f"  Epoch {epoch+1}/{PHASE2_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        es(val_loss, model)
        if es.early_stop: break
        
    print("Phase 2 completed. Evaluating on held-out TEST set...")
    model.load_state_dict(es.best_state)
    
    test_loss, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, return_probs=True)
    metrics = compute_metrics(y_true, y_pred, y_prob, test_loss)
    
    print("\n" + "="*60)
    print(f"FINAL TEST METRICS")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Test Loss         : {metrics['test_loss']:.4f}")
    print(f"Macro OVR AUC     : {metrics['macro_auc_ovr']:.4f}")
    print("Per-class Recall:")
    for i, cls in enumerate(class_names):
        print(f"  {cls:<25}: {metrics['per_class_recall'][i]:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()