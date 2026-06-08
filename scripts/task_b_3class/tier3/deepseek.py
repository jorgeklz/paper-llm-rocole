"""
Deep Learning Training Script for Coffee Leaf Disease Classification
Three classes: healthy, red_spider_mite, coffee_leaf_rust
Custom CNN with two-phase training strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    recall_score, roc_auc_score, classification_report
)
import numpy as np
from collections import Counter
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
set_seed(42)

# ============================================================================
# Model Architecture: Custom CNN with Parallel Convolutional Paths
# ============================================================================

class ParallelConvBlock(nn.Module):
    """Parallel convolutional block with 3x3 and 5x5 kernels"""
    def __init__(self, in_channels, out_channels):
        super(ParallelConvBlock, self).__init__()
        # 3x3 convolution path
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 5x5 convolution path (using two 3x3 convolutions for efficiency)
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        return out3x3 + out5x5  # Feature fusion via addition

class CustomCNN(nn.Module):
    """Custom CNN with 4 stages, parallel convolutions in stages 3-4"""
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        
        # Stage 1: 32 filters
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 64 filters
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 128 filters with parallel convolutions
        self.stage3 = nn.Sequential(
            ParallelConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 256 filters with parallel convolutions
        self.stage4 = nn.Sequential(
            ParallelConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),  # Slightly lower dropout for second layer
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ============================================================================
# Data Loading and Augmentation
# ============================================================================

def get_data_transforms():
    """Define train and validation transforms with augmentation"""
    
    # ImageNet statistics for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def create_dataloaders(data_root='data/splits/task_b_3class', batch_size=32):
    """Create train, validation, and test dataloaders with class balancing"""
    
    train_transform, val_transform = get_data_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_root, 'val'),
        transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_root, 'test'),
        transform=val_transform
    )
    
    # Calculate class weights for imbalance handling
    class_counts = Counter(train_dataset.targets)
    num_samples = len(train_dataset)
    num_classes = len(class_counts)
    
    # Compute class weights (inverse frequency)
    class_weights = torch.FloatTensor([
        num_samples / (num_classes * class_counts[i]) 
        for i in range(num_classes)
    ])
    
    print(f"Class distribution in training set:")
    print(f"  Class indices mapping: {train_dataset.class_to_idx}")
    print(f"  Samples per class: {dict(class_counts)}")
    print(f"  Class weights: {class_weights.numpy()}")
    
    # Create weighted sampler for balanced batch sampling
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=0  # CPU-only, set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader, class_weights

# ============================================================================
# Training and Evaluation Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def train_epoch(model, loader, criterion, optimizer, device='cpu'):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device='cpu'):
    """Validation loop"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

def compute_metrics(all_labels, all_preds, all_probs, num_classes=3):
    """Compute comprehensive metrics"""
    # Basic metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    
    # One-vs-Rest AUC
    try:
        # Convert labels to one-hot for AUC computation
        labels_onehot = np.zeros((len(all_labels), num_classes))
        labels_onehot[np.arange(len(all_labels)), all_labels] = 1
        
        # Compute macro AUC
        auc_scores = []
        for i in range(num_classes):
            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(labels_onehot[:, i], 
                                   [probs[i] for probs in all_probs])
                auc_scores.append(auc)
        macro_auc = np.mean(auc_scores) if auc_scores else 0.0
    except:
        macro_auc = 0.0
    
    # Loss (cross-entropy)
    loss = F.cross_entropy(torch.tensor(all_probs).float(), 
                          torch.tensor(all_labels).long()).item()
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'per_class_recall': per_class_recall,
        'macro_auc': macro_auc,
        'loss': loss
    }

# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    # Configuration
    BATCH_SIZE = 32  # Balanced for CPU memory, allows stable gradient estimates
    PHASE1_LR = 0.001  # Higher LR for training only the head
    PHASE2_LR = PHASE1_LR / 10  # 10x smaller for fine-tuning: 0.0001
    DROPOUT_RATE = 0.5  # Standard dropout, prevents overfitting with small minority class
    NUM_EPOCHS_PHASE1 = 15
    NUM_EPOCHS_PHASE2 = 20
    PATIENCE_PHASE1 = 5
    PATIENCE_PHASE2 = 7
    DEVICE = 'cpu'
    
    print("="*60)
    print("Coffee Leaf Disease Classification - Training Pipeline")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Phase 1 LR: {PHASE1_LR}, Phase 2 LR: {PHASE2_LR}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print("="*60)
    
    # Create dataloaders
    print("\n[1/5] Loading datasets...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        batch_size=BATCH_SIZE
    )
    
    # Initialize model
    print("\n[2/5] Initializing model...")
    model = CustomCNN(num_classes=3, dropout_rate=DROPOUT_RATE)
    model = model.to(DEVICE)
    
    # Use Focal Loss for class imbalance (better than class weighting for moderate imbalance)
    # Why Focal Loss? 
    # - Dynamically focuses on hard examples, preventing majority class domination
    # - Particularly effective for minority class (red_spider_mite 10.7%)
    # - Reduces relative loss for well-classified examples, emphasizing misclassifications
    # - Preserves per-class recall by preventing gradient vanishing from easy examples
    focal_loss = FocalLoss(alpha=class_weights.to(DEVICE), gamma=2.0)
    
    # ========================================================================
    # PHASE 1: Train only classification head
    # ========================================================================
    print("\n[3/5] PHASE 1: Training classification head (frozen backbone)...")
    
    # Freeze convolutional layers
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Verify freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Optimizer for phase 1
    optimizer_phase1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=1e-4
    )
    
    # Training loop for phase 1
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS_PHASE1):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_PHASE1}")
        
        train_loss, train_acc = train_epoch(model, train_loader, focal_loss, 
                                           optimizer_phase1, DEVICE)
        val_loss, val_acc, _, _, _ = validate_epoch(model, val_loader, focal_loss, DEVICE)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  New best model! (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
            if patience_counter >= PATIENCE_PHASE1:
                print(f"  Early stopping triggered!")
                break
    
    # Load best model from phase 1
    model.load_state_dict(best_model_state)
    
    # ========================================================================
    # PHASE 2: Fine-tune all layers
    # ========================================================================
    print("\n[4/5] PHASE 2: Fine-tuning all layers...")
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Optimizer for phase 2 with lower learning rate
    optimizer_phase2 = optim.Adam(
        model.parameters(),
        lr=PHASE2_LR,
        weight_decay=1e-4
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, 
        T_max=NUM_EPOCHS_PHASE2,
        eta_min=PHASE2_LR / 100
    )
    
    # Training loop for phase 2
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS_PHASE2):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_PHASE2}")
        
        train_loss, train_acc = train_epoch(model, train_loader, focal_loss,
                                           optimizer_phase2, DEVICE)
        val_loss, val_acc, _, _, _ = validate_epoch(model, val_loader, focal_loss, DEVICE)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  New best model! (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
            if patience_counter >= PATIENCE_PHASE2:
                print(f"  Early stopping triggered!")
                break
    
    # Load best model from phase 2
    model.load_state_dict(best_model_state)
    
    # ========================================================================
    # Final Evaluation on Test Set
    # ========================================================================
    print("\n[5/5] Final evaluation on test set...")
    _, _, _, all_labels, all_probs = validate_epoch(model, test_loader, 
                                                    focal_loss, DEVICE)
    
    # Get predictions for metrics
    all_preds = np.argmax(all_probs, axis=1)
    
    # Compute all metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes=3)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Loss:                         {metrics['loss']:.4f}")
    print(f"Accuracy:                     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Balanced Accuracy:            {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"Macro F1-Score:               {metrics['macro_f1']:.4f}")
    print(f"Macro One-vs-Rest AUC:        {metrics['macro_auc']:.4f}")
    
    print("\nPer-class Recall:")
    class_names = ['healthy', 'coffee_leaf_rust', 'red_spider_mite']
    for i, (name, recall) in enumerate(zip(class_names, metrics['per_class_recall'])):
        print(f"  {name:20s}: {recall:.4f} ({recall*100:.2f}%)")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()