import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS (theta)
# ==========================================
DATA_DIR = "data/splits/task_a_binary"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Optimal theta configurations
DROPOUT_RATE = 0.3
ETA_1 = 1e-3          # Phase 1 Learning Rate
ETA_2 = 5e-5          # Phase 2 Learning Rate
BATCH_SIZE = 32
NUM_UNITS = 128
# Class imbalance strategy 'none' implies standard CrossEntropyLoss without weights

PHASE_1_EPOCHS = 10
PHASE_2_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

device = torch.device("cpu")
print(# ==========================================
# 2. DATA PREPARATION
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use Dummy datasets if the directories don't exist yet to guarantee script execution safety
if not os.path.exists(TRAIN_DIR):
    print(f"Warning: Directory {TRAIN_DIR} not found. Creating a synthetic dataset for demonstration purposes.")
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in ["healthy", "unhealthy"]:
            os.makedirs(os.path.join(folder, cls), exist_ok=True)
            # Create 1 mock image per class if empty
            from PIL import Image
            img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
            img.save(os.path.join(folder, cls, "mock.jpg"))

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
# Using ResNet18 as our robust backbone feature extractor
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features

# Custom classification head corresponding to u and d parameters
classification_head = nn.Sequential(
    nn.Linear(num_features, NUM_UNITS),
    nn.ReLU(),
    nn.Dropout(p=DROPOUT_RATE),
    nn.Linear(NUM_UNITS, 2) # 2 Classes: Healthy vs Unhealthy
)

model.fc = classification_head
model = model.to(device)

criterion = nn.CrossEntropyLoss() # Standard CrossEntropy (Strategy w = none)

# ==========================================
# 4. HELPER EVALUATION FUNCTION
# ==========================================
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Probability for the positive class
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ==========================================
# 5. TRAINING PHASE 1: Train Head Only
# ==========================================
print("\n--- Starting Phase 1: Training Classification Head Only ---")
# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer_p1 = optim.Adam(model.fc.parameters(), lr=ETA_1)

for epoch in range(PHASE_1_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_p1.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_p1.step()
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Quick Validation pass
    labels, preds, _ = evaluate_model(model, val_loader)
    val_acc = accuracy_score(labels, preds)
    print(f"Epoch {epoch+1}/{PHASE_1_EPOCHS} - Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

# ==========================================
# 6. TRAINING PHASE 2: Full Fine-Tuning
# ==========================================
print("\n--- Starting Phase 2: Fine-Tuning Entire Network ---")
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

optimizer_p2 = optim.Adam(model.parameters(), lr=ETA_2)
scheduler = CosineAnnealingLR(optimizer_p2, T_max=PHASE_2_EPOCHS)

best_val_loss = float('inf')
patience_counter = 0
best_model_weights = model.state_dict().copy()

for epoch in range(PHASE_2_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_p2.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_p2.step()
        running_loss += loss.item() * inputs.size(0)
        
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Calculate Validation Loss for Early Stopping Check
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    
    labels, preds, _ = evaluate_model(model, val_loader)
    val_acc = accuracy_score(labels, preds)
    
    print(f"Epoch {epoch+1}/{PHASE_2_EPOCHS} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Early Stopping logic
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_model_weights = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

# Load best weights
model.load_state_dict(best_model_weights)

# ==========================================
# 7. FINAL HELD-OUT TEST EVALUATION
# ==========================================
print("\n--- Running Final Evaluation on Test Dataset ---")
y_true, y_pred, y_probs = evaluate_model(model, test_loader)

# Metrics calculation
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

# Robust fallback calculation for single-class mock datasets if AUC can't be computed
try:
    auc = roc_auc_score(y_true, y_probs)
except ValueError:
    auc = float('nan')

print("\n" + "="*30)
print("       FINAL TEST PERFORMANCE  ")
print("="*30)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}" if not np.isnan(auc) else "AUC-ROC:   N/A (Requires both classes present in test data)")
print("="*30)