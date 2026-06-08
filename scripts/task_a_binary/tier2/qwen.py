import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# ---------------- CONFIGURATION ----------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cpu')
print(f"Running on: {DEVICE}")

# Optimal hyperparameter vector
THETA = {
    'd': 0.3,          # dropout
    'eta_1': 1e-3,     # phase 1 lr
    'eta_2': 1e-4,     # phase 2 lr
    'b': 32,           # batch size
    'u': 128,          # dense units
    'w': 'none'        # class imbalance strategy
}

DATA_DIR = 'data/splits/task_a_binary'
print(f"Theta: {THETA}")

# ---------------- DATA PREP ----------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datasets_dict = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transform),
    'test': datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_test_transform)
}

dataloaders = {
    'train': DataLoader(datasets_dict['train'], batch_size=THETA['b'], shuffle=True, num_workers=0),
    'val': DataLoader(datasets_dict['val'], batch_size=THETA['b'], shuffle=False, num_workers=0),
    'test': DataLoader(datasets_dict['test'], batch_size=THETA['b'], shuffle=False, num_workers=0)
}

class_idx = datasets_dict['train'].class_to_idx
print(f"Classes: {class_idx}")

# ---------------- MODEL ----------------
class CustomCoffeeCNN(nn.Module):
    def __init__(self, num_units, dropout_rate):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(num_units, 1)  # Single logit for BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomCoffeeCNN(THETA['u'], THETA['d']).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()

# ---------------- TRAINING HELPER ----------------
def run_phase(model, dataloaders, optimizer, scheduler=None, num_epochs=20, patience=5, phase_name="Phase"):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    trigger_cnt = 0

    print(f"\n{'='*20} {phase_name} {'='*20}")
    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss /= len(dataloaders['train'].dataset)

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss /= len(dataloaders['val'].dataset)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # Scheduler step (CosineAnnealingLR expects step after epoch)
        if scheduler is not None:
            scheduler.step()

        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_wts = copy.deepcopy(model.state_dict())
            trigger_cnt = 0
        else:
            trigger_cnt += 1
            if trigger_cnt >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_wts)
    return model

# ---------------- PHASE 1: HEAD ONLY ----------------
print("\n🔒 Freezing feature extractor, training classification head...")
for param in model.features.parameters():
    param.requires_grad = False

optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=THETA['eta_1'], weight_decay=1e-4)
model = run_phase(model, dataloaders, optimizer1, num_epochs=10, patience=7, phase_name="Phase 1 (Head)")

# ---------------- PHASE 2: FULL FINE-TUNE ----------------
print("\n🔓 Unfreezing feature extractor, fine-tuning entire network...")
for param in model.parameters():
    param.requires_grad = True

optimizer2 = optim.Adam(model.parameters(), lr=THETA['eta_2'], weight_decay=1e-4)
scheduler2 = CosineAnnealingLR(optimizer2, T_max=30, eta_min=1e-6)
model = run_phase(model, dataloaders, optimizer2, scheduler=scheduler2, num_epochs=30, patience=7, phase_name="Phase 2 (Full)")

# ---------------- TEST EVALUATION ----------------
print("\n📊 Evaluating on held-out test set...")
model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(DEVICE)
        logits = model(inputs).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())
        all_probs.extend(probs.numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
try:
    auc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auc = float('nan')

print("\n✅ FINAL TEST METRICS:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")
print("="*40)