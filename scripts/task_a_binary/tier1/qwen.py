import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================
SEED = 42
BATCH_SIZE = 16
DATA_DIR = os.path.join('data', 'splits', 'task_a_binary')
DEVICE = torch.device('cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")

# =============================================================================
# DATA LOADING & TRANSFORMS
# =============================================================================
# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ImageFolder automatically assigns class indices alphabetically:
# healthy -> 0, unhealthy -> 1
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Dataset sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# =============================================================================
# MODEL DEFINITION & PROGRESSIVE TRAINING STRATEGY
# =============================================================================
def build_model():
    """Load pretrained ResNet18 and replace the classifier head for binary classification."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)  # Single logit for BCEWithLogitsLoss
    return model

def set_requires_grad(model, requires_grad):
    """Helper to freeze/unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = requires_grad

# Initialize model, loss, and optimizer
model = build_model().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()

# Track best validation loss for model restoration
best_val_loss = float('inf')
best_model_state = None

# =============================================================================
# TRAINING & VALIDATION FUNCTIONS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        logits = model(inputs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

# =============================================================================
# PHASE 1: TRAIN CLASSIFIER HEAD (BACKBONE FROZEN)
# =============================================================================
print("\n" + "="*50)
print("PHASE 1: Freezing backbone, training classification head...")
print("="*50)

set_requires_grad(model, requires_grad=False)
# Only train the newly added FC layer
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

PHASE1_EPOCHS = 8
for epoch in range(PHASE1_EPOCHS):
    tr_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
    
    # Save best model state
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{PHASE1_EPOCHS} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

# =============================================================================
# PHASE 2: FULL FINE-TUNING (BACKBONE UNFROZEN)
# =============================================================================
print("\n" + "="*50)
print("PHASE 2: Unfreezing backbone, fine-tuning entire network...")
print("="*50)

# Restore best weights from Phase 1 before fine-tuning
model.load_state_dict(best_model_state)

# Unfreeze entire model
set_requires_grad(model, requires_grad=True)

# Use a smaller LR and weight decay for stable fine-tuning
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

PHASE2_EPOCHS = 15
best_val_loss = float('inf')
best_model_state = None

for epoch in range(PHASE2_EPOCHS):
    tr_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    if (epoch + 1) % 3 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:2d}/{PHASE2_EPOCHS} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

# Load the absolute best model for final evaluation
print("\nLoading best validation checkpoint for test evaluation...")
model.load_state_dict(best_model_state)

# =============================================================================
# TEST EVALUATION & METRICS COMPUTATION
# =============================================================================
print("\n" + "="*50)
print("EVALUATING ON HELD-OUT TEST SET")
print("="*50)

model.eval()
all_labels = []
all_probs  = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        logits = model(inputs).squeeze(1)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.numpy())
        all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# Pure Python/NumPy metric calculation (fully self-contained)
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC via trapezoidal integration of ROC curve
    total_pos = np.sum(y_true)
    total_neg = len(y_true) - total_pos
    if total_pos == 0 or total_neg == 0:
        return accuracy, precision, recall, f1, 0.5
        
    sorted_idx = np.argsort(y_prob)[::-1]
    tpr_list, fpr_list = [0.0], [0.0]
    tp, fp = 0, 0
    
    for idx in sorted_idx:
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)
        
    auc = np.trapz(tpr_list, fpr_list)
    return accuracy, precision, recall, f1, auc

acc, prec, rec, f1, auc = compute_metrics(all_labels, all_probs)

print(f"\n{'='*30}")
print(f"FINAL TEST METRICS")
print(f"{'='*30}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(f"{'='*30}")