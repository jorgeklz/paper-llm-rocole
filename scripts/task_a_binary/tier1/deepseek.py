import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import os
import copy

# Set device (CPU only as requested)
device = torch.device('cpu')
print(f"Using device: {device}")

# Configuration
data_root = 'data/splits/task_a_binary'
img_size = 224
batch_size = 32
num_epochs_stage1 = 10
num_epochs_stage2 = 15
num_epochs_stage3 = 20
learning_rate = 1e-3
weight_decay = 1e-4

# Data transforms with ImageNet normalization
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("Loading datasets...")
train_dataset = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_root, 'test'), transform=val_test_transform)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model definition using ResNet18 (good balance for moderate-sized datasets)
def create_model():
    model = models.resnet18(pretrained=True)
    # Replace the final fully connected layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, accuracy, precision, recall, f1, auc

# Progressive training strategy
print("\n" + "="*60)
print("STAGE 1: Train only the final FC layer (frozen backbone)")
print("="*60)

model = create_model().to(device)

# Freeze all layers except the final FC layer
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=learning_rate, weight_decay=weight_decay)

best_val_accuracy = 0.0
best_model_state = None

for epoch in range(num_epochs_stage1):
    print(f"\nEpoch {epoch+1}/{num_epochs_stage1}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

# Load best model from stage 1
model.load_state_dict(best_model_state)

print("\n" + "="*60)
print("STAGE 2: Unfreeze last residual block (fine-tune deeper layers)")
print("="*60)

# Unfreeze the last residual block (layer4) and keep earlier layers frozen
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Reduce learning rate for fine-tuning
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=learning_rate * 0.1, weight_decay=weight_decay)

best_val_accuracy = 0.0
best_model_state = None

for epoch in range(num_epochs_stage2):
    print(f"\nEpoch {epoch+1}/{num_epochs_stage2}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

# Load best model from stage 2
model.load_state_dict(best_model_state)

print("\n" + "="*60)
print("STAGE 3: Unfreeze all layers (full fine-tuning)")
print("="*60)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Use an even smaller learning rate for full fine-tuning
optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.01, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

best_val_accuracy = 0.0
best_model_state = None

for epoch in range(num_epochs_stage3):
    print(f"\nEpoch {epoch+1}/{num_epochs_stage3}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

# Load the best model from stage 3
model.load_state_dict(best_model_state)

# Final evaluation on test set
print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)

test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate(model, test_loader, criterion)

print("\n" + "="*60)
print("TEST SET METRICS")
print("="*60)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"AUC:       {test_auc:.4f}")
print("="*60)

# Additional per-class breakdown
print("\nPer-class performance:")
from sklearn.metrics import classification_report
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=train_dataset.classes, digits=4))