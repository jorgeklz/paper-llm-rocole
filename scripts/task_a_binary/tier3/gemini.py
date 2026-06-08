import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# =====================================================================
# 1. ARCHITECTURE DEFINITION
# =====================================================================

class ParallelConvBlock(nn.Module):
    """Parallel convolutional paths with 3x3 and 5x5 kernels."""
    def __init__(self, in_channels, out_channels):
        super(ParallelConvBlock, self).__init__()
        # Split channels evenly between the two paths
        self.path_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.path_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return torch.cat([self.path_3x3(x), self.path_5x5(x)], dim=1)

class StandardConvBlock(nn.Module):
    """Standard sequential convolutional block."""
    def __init__(self, in_channels, out_channels):
        super(StandardConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class CoffeeLeafCNN(nn.Module):
    def __init__(self):
        super(CoffeeLeafCNN, self).__init__()
        
        # Stage 1 & 2: Standard progression
        self.stage1 = StandardConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.stage2 = StandardConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Stage 3 & 4: Parallel Paths (3x3 and 5x5)
        self.stage3 = ParallelConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.stage4 = ParallelConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 2)  # 2 Classes: Healthy, Unhealthy
        )
        
    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.pool4(self.stage4(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def freeze_features(self, freeze=True):
        """Freezes or unfreezes all layers except the classification head."""
        for name, child in self.named_children():
            if name != 'classifier':
                for param in child.parameters():
                    param.requires_grad = not freeze

# =====================================================================
# 2. EARLY STOPPING CLASS
# =====================================================================

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# =====================================================================
# 3. TRAINING ENGINE
# =====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# =====================================================================
# 4. MAIN EXECUTION PIPELINE
# =====================================================================

def main():
    device = torch.device("cpu")
    print("Running training script strictly on CPU...")

    # Paths
    base_dir = "data/splits/task_a_binary"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Class Weighting Setup
    # Hardcoded/calculated baseline logic for (791 healthy, 769 unhealthy) dynamic balance robustness
    class_counts = np.bincount(train_dataset.targets)
    total_samples = sum(class_counts)
    # Inverse frequency weighting
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    print(f"Dataset Loaded. Class balance weights: {class_weights}")

    # Model Initialization
    model = CoffeeLeafCNN().to(device)

    # -----------------------------------------------------------------
    # PHASE 1: Train classification head only
    # -----------------------------------------------------------------
    print("\n--- Starting Phase 1: Training Head Only ---")
    model.freeze_features(freeze=True)
    
    # Filter optimizer to only track parameters that require gradients
    optimizer_p1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    early_stopping_p1 = EarlyStopping(patience=5)

    for epoch in range(15):
        train_loss = train_epoch(model, train_loader, criterion, optimizer_p1, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:02d}/15 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        early_stopping_p1(val_loss, model)
        if early_stopping_p1.early_stop:
            print("Early stopping triggered in Phase 1.")
            break
            
    # Load the best weights from Phase 1 before moving to Phase 2
    model.load_state_dict(early_stopping_p1.best_model_wts)

    # -----------------------------------------------------------------
    # PHASE 2: Fine-tune all layers
    # -----------------------------------------------------------------
    print("\n--- Starting Phase 2: Fine-Tuning Entire Network ---")
    model.freeze_features(freeze=False)
    
    optimizer_p2 = optim.Adam(model.parameters(), lr=1e-4) # 10x smaller learning rate
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=20)
    early_stopping_p2 = EarlyStopping(patience=7)

    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, criterion, optimizer_p2, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler_p2.step()
        
        print(f"Epoch {epoch+1:02d}/20 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler_p2.get_last_lr()[0]:.6f}")
        
        early_stopping_p2(val_loss, model)
        if early_stopping_p2.early_stop:
            print("Early stopping triggered in Phase 2.")
            break

    # Restore overall best weights
    model.load_state_dict(early_stopping_p2.best_model_wts)

    # =====================================================================
    # 5. HELD-OUT TEST EVALUATION
    # =====================================================================
    print("\n--- Final Evaluation on Held-Out Test Set ---")
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Probability for the positive class (unhealthy)

    # Metrics computation
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    auc = roc_auc_score(all_targets, all_probs)

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall   : {recall:.4f}")
    print(f"Test F1-Score : {f1:.4f}")
    print(f"Test AUC-ROC  : {auc:.4f}")

if __name__ == '__main__':
    main()