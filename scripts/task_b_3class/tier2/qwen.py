import os
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')

# Force CPU execution
device = torch.device('cpu')

# ==============================================================================
# 1. HYPERPARAMETER VECTOR THETA
# ==============================================================================
THETA = {
    'd': 0.3,          # dropout rate
    'eta_1': 1e-3,     # Phase 1 learning rate
    'eta_2': 1e-5,     # Phase 2 learning rate
    'b': 32,           # batch size
    'u': 128,          # dense layer units
    'w': 'focal_loss'  # imbalance strategy
}

# ==============================================================================
# 2. CUSTOM CNN ARCHITECTURE
# ==============================================================================
class CoffeeCNN(nn.Module):
    def __init__(self, num_classes=3, dense_units=128, dropout=0.3):
        super().__init__()
        # Feature Extractor (Backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1)
        )
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# ==============================================================================
# 3. FOCAL LOSS IMPLEMENTATION
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)  # Probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ==============================================================================
# 4. EARLY STOPPING CALLBACK
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=8, path='best_checkpoint.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.model_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==============================================================================
# 5. DATA LOADING & TRANSFORMS
# ==============================================================================
train_dir = 'data/splits/task_b_3class/train'
val_dir   = 'data/splits/task_b_3class/val'
test_dir  = 'data/splits/task_b_3class/test'

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
val_dataset   = datasets.ImageFolder(val_dir, transform=eval_tf)
test_dataset  = datasets.ImageFolder(test_dir, transform=eval_tf)

train_loader = DataLoader(train_dataset, batch_size=THETA['b'], shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=THETA['b'], shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=THETA['b'], shuffle=False, num_workers=0)

class_names = train_dataset.classes
class_idx = {name: i for i, name in enumerate(class_names)}

# ==============================================================================
# 6. TRAINING LOOP (TWO-PHASE STRATEGY)
# ==============================================================================
criterion = FocalLoss(gamma=2.0) if THETA['w'] == 'focal_loss' else nn.CrossEntropyLoss()
model = CoffeeCNN(num_classes=3, dense_units=THETA['u'], dropout=THETA['d']).to(device)

early_stopper = EarlyStopping(patience=8, path='best_model.pth')

# --- PHASE 1: Freeze Backbone, Train Head ---
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer1 = optim.Adam(model.classifier.parameters(), lr=THETA['eta_1'], weight_decay=1e-4)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.5)

print("=== PHASE 1: Training Classification Head (Frozen Backbone) ===")
for epoch in range(1, 16):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer1.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer1.step()
        running_loss += loss.item() * inputs.size(0)
    scheduler1.step()
    print(f"Epoch {epoch:2d} | Train Loss: {running_loss/len(train_dataset):.4f}")

# --- PHASE 2: Unfreeze, Fine-tune Entire Network ---
for param in model.backbone.parameters():
    param.requires_grad = True

optimizer2 = optim.Adam(model.parameters(), lr=THETA['eta_2'], weight_decay=1e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=40, eta_min=1e-7)

print("\n=== PHASE 2: Fine-tuning Entire Network ===")
for epoch in range(1, 41):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer2.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()
        running_loss += loss.item() * inputs.size(0)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            v_loss = criterion(outputs, labels)
            val_loss += v_loss.item() * inputs.size(0)
    val_loss /= len(val_dataset)
    scheduler2.step()
    
    print(f"Epoch {epoch:2d} | Train Loss: {running_loss/len(train_dataset):.4f} | Val Loss: {val_loss:.4f}")
    early_stopper(val_loss, model)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch}. Restoring best weights...")
        break

# ==============================================================================
# 7. TEST EVALUATION & METRICS
# ==============================================================================
print("\n=== EVALUATING ON HELD-OUT TEST SET ===")
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

all_labels, all_preds, all_probs = [], [], []
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        test_loss += loss.item() * inputs.size(0)

all_labels = np.concatenate(all_labels)
all_preds  = np.concatenate(all_preds)
all_probs  = np.concatenate(all_probs)

avg_test_loss = test_loss / len(test_dataset)
acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
bal_acc = balanced_accuracy_score(all_labels, all_preds)
per_class_rec = recall_score(all_labels, all_preds, average=None)
macro_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')

print(f"\n--- Final Test Metrics ---")
print(f"Test Loss          : {avg_test_loss:.4f}")
print(f"Accuracy           : {acc:.4f}")
print(f"Macro-F1           : {macro_f1:.4f}")
print(f"Balanced Accuracy  : {bal_acc:.4f}")
print(f"Macro OvR AUC      : {macro_auc:.4f}")
print("Per-Class Recall:")
for i, cls in enumerate(class_names):
    print(f"  {cls:<20}: {per_class_rec[i]:.4f}")

# Cleanup
if os.path.exists('best_model.pth'):
    os.remove('best_model.pth')
print("\nDone. Temporary checkpoint removed.")