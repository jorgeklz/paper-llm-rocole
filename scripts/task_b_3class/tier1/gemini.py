import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, roc_auc_score

# --------------------------------------------------------
# 1. Configuration & Hyperparameters
# --------------------------------------------------------
DATA_DIR = "data/splits/task_b_3class"
BATCH_SIZE = 32
NUM_CLASSES = 3
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 7
DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# Imbalance handling: Class counts provided by user
# Classes order in ImageFolder alphabetized: coffee_leaf_rust (602), healthy (791), red_spider_mite (167)
class_counts = np.array([602, 791, 167]) 
total_samples = sum(class_counts)
# Inverse frequency weighting
class_weights = total_samples / (NUM_CLASSES * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Calculated class weights for Loss Function: {class_weights}")

# --------------------------------------------------------
# 2. Data Transformations & DataLoaders
# --------------------------------------------------------
# ImageNet normalization constants
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Create datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=val_test_transforms)
test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=val_test_transforms)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(class_dataset_mapping := f"Class mapping detected: {train_dataset.class_to_idx}")

# --------------------------------------------------------
# 3. Model Definition (EfficientNet-B0)
# --------------------------------------------------------
# Using weights=models.EfficientNet_B0_Weights.DEFAULT for pre-trained weights
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Modify the classifier head for 3 classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# Define Weighted Loss Function
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# --------------------------------------------------------
# 4. Training Engine Function
# --------------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

# --------------------------------------------------------
# 5. Progressive Training Execution
# --------------------------------------------------------

# --- PHASE 1: Feature Extractor Frozen (Warm-up Head) ---
print("\n--- Phase 1: Training Classifier Head Only ---")
for param in model.features.parameters():
    param.requires_grad = False

# High learning rate for the fresh head
optimizer_phase1 = optim.Adam(model.classifier.parameters(), lr=1e-3)

for epoch in range(PHASE1_EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer_phase1, DEVICE)
    val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)
    print(f"Epoch {epoch+1}/{PHASE1_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# --- PHASE 2: Deeper Layers Unfrozen (Fine-Tuning) ---
print("\n--- Phase 2: Fine-Tuning Top Backbone Blocks & Head ---")
# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

# Alternative: selectively freeze early layers if desired. Here we fine-tune everything with a tiny LR.
optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

for epoch in range(PHASE2_EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer_phase2, DEVICE)
    val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)
    print(f"Epoch {epoch+1}/{PHASE2_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# --------------------------------------------------------
# 6. Comprehensive Evaluation on Held-Out Test Set
# --------------------------------------------------------
print("\n--- Evaluating Final Model on Held-Out Test Set ---")
model.eval()

all_labels = []
all_preds = []
all_probs = []
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        
        # Softmax for probabilities (Needed for Multi-class One-vs-Rest AUC)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

test_loss /= len(test_loader.dataset)
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Calculate Target Metrics
acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
balanced_acc = balanced_accuracy_score(all_labels, all_preds)
macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

# Per-class recall extraction from classification report
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)

# --------------------------------------------------------
# 7. Final Metrics Output Display
# --------------------------------------------------------
print("\n================ FINAL TEST METRICS ================")
print(f"Test Loss:            {test_loss:.4f}")
print(f"Final Accuracy:       {acc * 100:.2f}%")
print(f"Balanced Accuracy:    {balanced_acc * 100:.2f}%")
print(f"Macro-F1 Score:       {macro_f1:.4f}")
print(f"Macro OvR AUC:        {macro_auc:.4f}")
print("----------------------------------------------------")
print("Per-Class Recall:")
for cls in test_dataset.classes:
    print(f"  - {cls}: {report[cls]['recall'] * 100:.2f}%")
print("====================================================")