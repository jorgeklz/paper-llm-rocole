"""
Binary Classification of Robusta Coffee Leaf Images
Healthy vs Unhealthy using Custom CNN with Two-Phase Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(42)

# ============================================================================
# Data Configuration
# ============================================================================
DATA_ROOT = "data/splits/task_a_binary"
BATCH_SIZE = 32  # Balanced for CPU training - large enough for stable gradients
IMG_SIZE = 224
NUM_CLASSES = 2

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Data Augmentation and Loading
# ============================================================================
def get_transforms(phase='train'):
    """Get data transforms for different phases"""
    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    return transform

def create_dataloaders():
    """Create train, val, and test dataloaders with class imbalance handling"""
    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, 'train'),
        transform=get_transforms('train')
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, 'val'),
        transform=get_transforms('val')
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, 'test'),
        transform=get_transforms('val')
    )
    
    # Handle class imbalance using WeightedRandomSampler
    train_targets = [label for _, label in train_dataset]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,  # Set to 0 for CPU
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Dataset Statistics:")
    print(f"  Train: {len(train_dataset)} images (Healthy: {class_counts[0]}, Unhealthy: {class_counts[1]})")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"  Class weights: Health={class_weights[0]:.3f}, Unhealthy={class_weights[1]:.3f}")
    
    return train_loader, val_loader, test_loader, class_weights

# ============================================================================
# Custom CNN Architecture
# ============================================================================
class ParallelConvBlock(nn.Module):
    """Parallel convolution block with multiple kernel sizes"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)
        out = torch.cat([out3, out5], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

class CoffeeLeafCNN(nn.Module):
    """Custom CNN for coffee leaf classification with parallel convolutions"""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        
        # Stage 1: 32 filters
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Stage 2: 64 filters
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Stage 3: Parallel convs with 128 filters (64+64 from parallel paths)
        self.stage3 = ParallelConvBlock(64, 64)  # Output: 128 channels
        
        # Stage 4: Parallel convs with 256 filters (128+128 from parallel paths)
        self.stage4 = ParallelConvBlock(128, 128)  # Output: 256 channels
        
        # Global pooling and classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head - justification: 
        # 256 features → 128 (reduction) → 64 (compression) → num_classes
        # This hierarchy allows learning complex feature interactions
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Dropout rate 0.5 prevents overfitting
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.8),  # Slightly less aggressive
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling and classification
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# Training Utilities
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - more robust than class weighting"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def compute_metrics(model, data_loader):
    """Compute comprehensive evaluation metrics"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.numpy())
            all_probs.extend(probs[:, 1].numpy())  # Probability for positive class
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train_epoch(model, loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return val_loss / len(loader), 100. * correct / total

# ============================================================================
# Training Phases
# ============================================================================
def phase1_train(model, train_loader, val_loader, class_weights):
    """Phase 1: Train only classification head"""
    print("\n" + "="*60)
    print("PHASE 1: Training Classification Head (Frozen Convolutional Layers)")
    print("="*60)
    
    # Freeze all convolutional layers
    for param in model.stage1.parameters():
        param.requires_grad = False
    for param in model.stage2.parameters():
        param.requires_grad = False
    for param in model.stage3.parameters():
        param.requires_grad = False
    for param in model.stage4.parameters():
        param.requires_grad = False
    
    # Verify only classifier is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Hyperparameters for Phase 1:
    # Learning rate = 0.01 - Higher because we're only training the head
    # Criterion: Focal Loss for imbalance robustness
    learning_rate = 0.01
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    
    for epoch in range(1, 16):  # Max 15 epochs
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f'Phase 1 - Epoch {epoch:2d}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_phase1_model.pth')
            print(f'  -> New best model saved (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  -> Early stopping triggered after epoch {epoch}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_phase1_model.pth'))
    return model

def phase2_train(model, train_loader, val_loader, class_weights):
    """Phase 2: Fine-tune all layers"""
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning All Layers")
    print("="*60)
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Hyperparameters for Phase 2:
    # Learning rate = 0.001 (10x smaller than Phase 1)
    learning_rate = 0.001
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 7
    
    for epoch in range(1, 21):  # Max 20 epochs
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Phase 2 - Epoch {epoch:2d}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
              f'LR: {current_lr:.6f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_phase2_model.pth')
            print(f'  -> New best model saved (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  -> Early stopping triggered after epoch {epoch}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_phase2_model.pth'))
    return model

# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    print("\n" + "="*60)
    print("COFFEE LEAF BINARY CLASSIFICATION - TRAINING PIPELINE")
    print("="*60)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders()
    
    # Initialize model
    # Dropout rate justification: 0.5 is standard for moderate-sized models
    # Provides good regularization without underfitting given ~1500 training samples
    model = CoffeeLeafCNN(num_classes=NUM_CLASSES, dropout_rate=0.5)
    print(f"\nModel Architecture:\n{model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase 1: Train only classification head
    model = phase1_train(model, train_loader, val_loader, class_weights)
    
    # Phase 2: Fine-tune all layers
    model = phase2_train(model, train_loader, val_loader, class_weights)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)
    
    test_metrics = compute_metrics(model, test_loader)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {test_metrics['f1']*100:.2f}%")
    print(f"  AUC-ROC:   {test_metrics['auc']*100:.2f}%")
    
    # Clean up checkpoint files
    if os.path.exists('best_phase1_model.pth'):
        os.remove('best_phase1_model.pth')
    if os.path.exists('best_phase2_model.pth'):
        os.remove('best_phase2_model.pth')
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()