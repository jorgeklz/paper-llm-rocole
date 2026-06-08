import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    classification_report, roc_auc_score
)
from PIL import Image

# ==========================================
# 0. SETUP MOCK DATA IF PATHS DO NOT EXIST
# ==========================================
DATA_DIR = "data/splits/task_b_3class"
CLASSES = ["healthy", "red_spider_mite", "coffee_leaf_rust"]

def create_mock_dataset():
    """Generates dummy images to ensure the script is instantly executable."""
    print("Creating mock dataset directory structure and dummy images...")
    splits = ['train', 'val', 'test']
    counts = {
        'train': {'healthy': 500, 'coffee_leaf_rust': 400, 'red_spider_mite': 100},
        'val': {'healthy': 145, 'coffee_leaf_rust': 101, 'red_spider_mite': 33},
        'test': {'healthy': 146, 'coffee_leaf_rust': 101, 'red_spider_mite': 34}
    }
    
    for split in splits:
        for cls in CLASSES:
            path = os.path.join(DATA_DIR, split, cls)
            os.makedirs(path, exist_ok=True)
            # Create a few dummy images per class to allow code execution
            num_images = min(counts[split][cls], 5) # Cap dummy images for swift execution
            for i in range(num_images):
                img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
                img.save(os.path.join(path, f"dummy_{i}.jpg"))

if not os.path.exists(DATA_DIR):
    create_mock_dataset()

# ==========================================
# 1. HYPERPARAMETERS (theta) & CONFIGURATION
# ==========================================
D = 0.3                  # Dropout rate
ETA_1 = 1e-3             # Phase 1 Learning rate
ETA_2 = 5e-5             # Phase 2 Learning rate
B = 32                   # Batch size
U = 128                  # Dense units
W = 'class_weights'      # Imbalance strategy

PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 5
NUM_CLASSES = 3

# Exact class counts from user prompt for precise class weights
CLASS_COUNTS = torch.tensor([791.0, 167.0, 602.0]) # healthy, red_spider_mite, coffee_leaf_rust
total_samples = CLASS_COUNTS.sum()
# Inverse frequency weighting: total / (num_classes * class_samples)
CLASS_WEIGHTS = total_samples / (NUM_CLASSES * CLASS_COUNTS)

print(f"Calculated Class Weights: {CLASS_WEIGHTS.tolist()}")

# ==========================================
# 2. DATA LOADERS & TRANSFORMS
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val_test']),
    'test': datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), data_transforms['val_test'])
}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=B, shuffle=(x == 'train'), num_workers=0)
               for x in ['train', 'val', 'test']}

# Ensure mappings match expected order
print(f"Dataset Class-to-Index Mapping: {image_datasets['train'].class_to_idx}")

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
# Using a lightweight ResNet18 as our core CNN feature extractor
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features

# Custom classification head incorporating hyperparameters 'd' and 'u'
custom_head = nn.Sequential(
    nn.Linear(num_ftrs, U),
    nn.ReLU(),
    nn.Dropout(p=D),
    nn.Linear(U, NUM_CLASSES)
)
model.fc = custom_head

# Loss function mapping the 'w' strategy
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

# ==========================================
# 4. TRAINING ENGINE & VAL FUNCTIONS
# ==========================================
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, phase_name):
    print(f"\n--- Starting {phase_name} ---")
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        if scheduler:
            scheduler.step()
            
        train_loss = running_loss / len(dataloaders['train'].dataset)
        val_loss, _, _, _ = evaluate_model(model, dataloaders['val'], criterion)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping verification
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered in {phase_name}!")
                break
                
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 5. EXECUTE TWO-PHASE TRAINING
# ==========================================

# --- PHASE 1: Train Classification Head Only ---
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer_p1 = optim.Adam(model.fc.parameters(), lr=ETA_1)
model = train_model(model, dataloaders, criterion, optimizer_p1, scheduler=None, num_epochs=PHASE1_EPOCHS, phase_name="Phase 1 (Head Only)")

# --- PHASE 2: Fine-tune Entire Network ---
for param in model.parameters():
    param.requires_grad = True

optimizer_p2 = optim.Adam(model.parameters(), lr=ETA_2)
scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=PHASE2_EPOCHS)
model = train_model(model, dataloaders, criterion, optimizer_p2, scheduler=scheduler_p2, num_epochs=PHASE2_EPOCHS, phase_name="Phase 2 (Full Fine-Tuning)")

# ==========================================
# 6. EVALUATION ON HELD-OUT TEST SET
# ==========================================
print("\n=== Final Evaluation on Held-Out Test Set ===")
test_loss, labels, preds, probs = evaluate_model(model, dataloaders['test'], criterion)

# Metric Calculations
acc = accuracy_score(labels, preds)
balanced_acc = balanced_accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average='macro')

# Per-class Recall Extraction
report = classification_report(labels, preds, target_names=CLASSES, output_dict=True, zero_division=0)
recall_healthy = report['healthy']['recall']
recall_mite = report['red_spider_mite']['recall']
recall_rust = report['coffee_leaf_rust']['recall']

# One-vs-Rest Macro AUC
# Binarize outputs to compute multi-class One-vs-Rest AUC safely
try:
    macro_auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
except ValueError:
    # Fallback if mock dataset lacks structural diversity during testing configurations
    macro_auc = float('nan')

# Output Final Results Matrix
print("-" * 50)
print(f"Final Test Loss        : {test_loss:.4f}")
print(f"Overall Accuracy       : {acc * 100:.2f}%")
print(f"Balanced Accuracy      : {balanced_acc * 100:.2f}%")
print(f"Macro F1-Score         : {macro_f1:.4f}")
print(f"Macro One-vs-Rest AUC  : {macro_auc:.4f}" if not np.isnan(macro_auc) else "Macro One-vs-Rest AUC  : N/A (Insufficient mock class variations)")
print("-" * 50)
print("Per-Class Recall Rates:")
print(f"  - Healthy            : {recall_healthy * 100:.2f}%")
print(f"  - Coffee Leaf Rust   : {recall_rust * 100:.2f}%")
print(f"  - Red Spider Mite    : {recall_mite * 100:.2f}% [MINORITY CLASS]")
print("-" * 50)