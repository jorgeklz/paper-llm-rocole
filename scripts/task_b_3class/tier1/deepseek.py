import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             balanced_accuracy_score, roc_auc_score)
import numpy as np
from collections import Counter
import copy
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DATA_ROOT = 'data/splits/task_b_3class'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS_PHASE1 = 15  # Frozen backbone
NUM_EPOCHS_PHASE2 = 25  # Fine-tune entire network
LEARNING_RATE_PHASE1 = 0.001
LEARNING_RATE_PHASE2 = 0.0001
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 3
CLASS_NAMES = ['healthy', 'red_spider_mite', 'coffee_leaf_rust']

# Image transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_datasets():
    """Load datasets with class weights for imbalance handling"""
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, 'train'), 
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, 'val'), 
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, 'test'), 
        transform=val_test_transform
    )
    
    # Calculate class weights for imbalance
    class_counts = [0, 0, 0]
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    return train_dataset, val_dataset, test_dataset, sampler, class_counts

def create_model():
    """Create ResNet50 model with modified classifier head"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device='cpu'):
    """Evaluate model on validation/test set"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_labels, all_preds, all_probs

def print_metrics(all_labels, all_preds, all_probs, class_names):
    """Print comprehensive evaluation metrics"""
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Per-class recall
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    
    # Loss (cross-entropy, computed earlier)
    
    # One-vs-Rest AUC
    # Convert predictions to one-hot for AUC
    n_classes = len(class_names)
    all_labels_onehot = np.eye(n_classes)[all_labels]
    try:
        ovr_auc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')
    except ValueError:
        ovr_auc = 0.0  # In case some classes have no samples
    
    # Print results
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro One-vs-Rest AUC: {ovr_auc:.4f}")
    print("\nPer-class Recall:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:20s}: {per_class_recall[i]:.4f}")
    print("="*60)

def main():
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset, sampler, class_counts = load_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Class distribution - Healthy: {class_counts[0]}, Red_spider_mite: {class_counts[1]}, Coffee_leaf_rust: {class_counts[2]}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Class weights for loss function
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # PHASE 1: Train only the classifier head (backbone frozen)
    print("\n" + "="*60)
    print("PHASE 1: Training classifier head (backbone frozen)")
    print("="*60)
    
    optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_PHASE1, weight_decay=WEIGHT_DECAY)
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(NUM_EPOCHS_PHASE1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_phase1, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS_PHASE1} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # Load best model from phase 1
    model.load_state_dict(best_model_wts)
    
    # PHASE 2: Unfreeze and fine-tune entire network
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning entire network")
    print("="*60)
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Use lower learning rate for fine-tuning
    optimizer_phase2 = optim.Adam(model.parameters(), lr=LEARNING_RATE_PHASE2, weight_decay=WEIGHT_DECAY)
    
    best_val_acc_phase2 = 0.0
    best_model_wts_phase2 = copy.deepcopy(model.state_dict())
    
    for epoch in range(NUM_EPOCHS_PHASE2):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_phase2, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS_PHASE2} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            best_model_wts_phase2 = copy.deepcopy(model.state_dict())
    
    # Load best model from phase 2
    model.load_state_dict(best_model_wts_phase2)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, device)
    
    # Print comprehensive metrics
    print_metrics(test_labels, test_preds, test_probs, CLASS_NAMES)
    print(f"\nTest Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()