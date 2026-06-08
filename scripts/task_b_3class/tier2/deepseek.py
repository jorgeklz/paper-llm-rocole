"""
Deep CNN for Three-Class Classification of Robusta Coffee Leaf Images
Two-Phase Training Strategy with Class Imbalance Handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, 
    balanced_accuracy_score, roc_auc_score, confusion_matrix
)
from collections import Counter
import copy
import os
from PIL import Image

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==================== CONFIGURATION ====================
# Optimal hyperparameter vector theta based on best practices:
# - d = 0.5: Higher dropout for classification head to prevent overfitting 
#   given moderate dataset size and fine-tuning phase
# - eta_1 = 1e-3: Higher initial learning rate for training only the head
# - eta_2 = 1e-4: Lower learning rate for gentle fine-tuning of entire network
# - b = 32: Balanced batch size for stable gradient estimates with moderate memory
# - u = 256: Sufficient capacity for 3-class problem with feature abstraction
# - w = focal_loss: Best for minority class (10.7%), focuses on hard examples

CONFIG = {
    'd': 0.5,           # Dropout rate
    'eta_1': 1e-3,      # Phase 1 learning rate
    'eta_2': 1e-4,      # Phase 2 learning rate
    'b': 32,            # Batch size
    'u': 256,           # Dense layer units
    'w': 'focal_loss',  # Class imbalance handling
    'num_classes': 3,
    'img_size': 224,
    'epochs_phase1': 30,
    'epochs_phase2': 50,
    'patience': 7,      # Early stopping patience
    'data_dir': 'data/splits/task_b_3class'
}

# Class names and distribution
CLASSES = ['healthy', 'coffee_leaf_rust', 'red_spider_mite']
CLASS_DIST = {'healthy': 791, 'coffee_leaf_rust': 602, 'red_spider_mite': 167}
CLASS_WEIGHTS = torch.tensor([1.0/791, 1.0/602, 1.0/167])  # Inverse frequency
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum() * 3  # Normalize

# ==================== FOCAL LOSS IMPLEMENTATION ====================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard, misclassified examples.
    gamma=2 focuses more on hard examples, alpha helps with class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==================== MODEL DEFINITION ====================
class CustomCNN(nn.Module):
    """
    Custom CNN with ResNet34 backbone and configurable classification head
    """
    def __init__(self, num_classes=3, dropout_rate=0.5, num_units=256):
        super(CustomCNN, self).__init__()
        # Use pretrained ResNet34 as feature extractor
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Get the number of features from ResNet's final layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_units, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# ==================== DATA LOADING ====================
def get_data_loaders(config):
    """Create data loaders with appropriate transforms and samplers"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(config['data_dir'], 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(config['data_dir'], 'val'),
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(config['data_dir'], 'test'),
        transform=val_test_transform
    )
    
    # Handle class imbalance with oversampling for minority class
    if config['w'] == 'oversampling':
        # Get class samples count
        targets = [label for _, label in train_dataset]
        class_counts = Counter(targets)
        
        # Calculate weights for each sample
        sample_weights = [1.0 / class_counts[label] for label in targets]
        
        # Create sampler
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(
            train_dataset, batch_size=config['b'], sampler=sampler, 
            num_workers=0, pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config['b'], shuffle=True,
            num_workers=0, pin_memory=False
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['b'], shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['b'], shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, loader, criterion, optimizer, device='cpu'):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in loader:
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
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device='cpu'):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_labels, all_preds, all_probs

def get_criterion(config, device='cpu'):
    """Get appropriate loss function based on configuration"""
    if config['w'] == 'class_weights':
        class_weights = torch.tensor([1.0/CLASS_DIST[c] for c in CLASSES])
        class_weights = class_weights / class_weights.sum() * 3
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif config['w'] == 'focal_loss':
        alpha = torch.tensor([1.0/CLASS_DIST[c] for c in CLASSES])
        alpha = alpha / alpha.sum() * 3
        return FocalLoss(alpha=alpha.to(device), gamma=2.0)
    else:  # 'none'
        return nn.CrossEntropyLoss()

# ==================== EARLY STOPPING ====================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        return self.early_stop

# ==================== EVALUATION METRICS ====================
def print_metrics(labels, preds, probs, dataset_name="Test"):
    """Print comprehensive evaluation metrics"""
    # Per-class recall
    recalls = recall_score(labels, preds, average=None)
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} SET RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"Macro-F1: {f1_score(labels, preds, average='macro'):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(labels, preds):.4f}")
    print(f"Loss: {nn.CrossEntropyLoss()(torch.tensor(probs).log(), torch.tensor(labels)).item():.4f}")
    
    # Per-class recall
    print("\nPer-Class Recall:")
    for i, class_name in enumerate(CLASSES):
        print(f"  {class_name:20s}: {recalls[i]:.4f}")
    
    # Macro one-vs-rest AUC (OvR)
    try:
        # Binarize labels for OvR AUC
        labels_onehot = np.eye(3)[labels]
        auc_ovr = roc_auc_score(labels_onehot, probs, average='macro', multi_class='ovr')
        print(f"\nMacro OvR AUC: {auc_ovr:.4f}")
    except:
        print("\nMacro OvR AUC: Could not compute (might need all classes in batch)")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 " + "  ".join([f"{c[:3]:>4}" for c in CLASSES]))
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name:20s} {cm[i]}")
    
    print(f"{'='*60}\n")

# ==================== MAIN TRAINING SCRIPT ====================
def main():
    print("="*60)
    print("COFFEE LEAF DISEASE CLASSIFICATION - TWO-PHASE TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dropout (d): {CONFIG['d']}")
    print(f"  Phase1 LR (eta_1): {CONFIG['eta_1']}")
    print(f"  Phase2 LR (eta_2): {CONFIG['eta_2']}")
    print(f"  Batch Size (b): {CONFIG['b']}")
    print(f"  Dense Units (u): {CONFIG['u']}")
    print(f"  Imbalance Handling (w): {CONFIG['w']}")
    print(f"  Class Distribution: {CLASS_DIST}\n")
    
    # Setup device (CPU only)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(CONFIG)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = CustomCNN(
        num_classes=CONFIG['num_classes'],
        dropout_rate=CONFIG['d'],
        num_units=CONFIG['u']
    )
    model.to(device)
    
    # ==================== PHASE 1: Train only classifier head ====================
    print("\n" + "="*60)
    print("PHASE 1: Training Classification Head Only (Feature Extractors Frozen)")
    print("="*60)
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Phase 1 optimizer and criterion
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['eta_1'])
    criterion = get_criterion(CONFIG, device)
    
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs_phase1']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer1, device)
        val_loss, val_acc, _, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Phase 1 - Epoch {epoch+1}/{CONFIG['epochs_phase1']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Load best model from phase 1
    model.load_state_dict(best_model_state)
    
    # ==================== PHASE 2: Fine-tune entire network ====================
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning Entire Network")
    print("="*60)
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Phase 2 optimizer with lower learning rate
    optimizer2 = optim.Adam(model.parameters(), lr=CONFIG['eta_2'])
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer2, T_max=CONFIG['epochs_phase2'], eta_min=CONFIG['eta_2']/100)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['patience'])
    best_val_loss_phase2 = float('inf')
    
    for epoch in range(CONFIG['epochs_phase2']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer2, device)
        val_loss, val_acc, _, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Phase 2 - Epoch {epoch+1}/{CONFIG['epochs_phase2']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model from phase 2
    model.load_state_dict(early_stopping.best_model)
    
    # ==================== FINAL EVALUATION ON TEST SET ====================
    print("\n" + "="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)
    
    # Evaluate on test set
    criterion_eval = nn.CrossEntropyLoss()  # Standard loss for reporting
    test_loss, test_acc, test_labels, test_preds, test_probs = validate_epoch(
        model, test_loader, criterion_eval, device
    )
    
    # Print comprehensive metrics
    print_metrics(test_labels, test_preds, test_probs, "Test")
    
    # Save model
    torch.save(model.state_dict(), 'best_coffee_leaf_model.pth')
    print("Model saved as 'best_coffee_leaf_model.pth'")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()