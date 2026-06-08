import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, roc_auc_score

# ==========================================
# 1. ARCHITECTURE DEFINITION
# ==========================================

class ParallelConvBlock(nn.Module):
    """Parallel convolutional paths with 3x3 and 5x5 kernels used in stages 3 and 4."""
    def __init__(self, in_channels, out_channels):
        super(ParallelConvBlock, self).__init__()
        # Split channels evenly between the two paths
        mid_channels = out_channels // 2
        
        # Path 1: 3x3 Convolution
        self.path_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        # Path 2: 5x5 Convolution (using padding=2 to keep spatial dimensions identical)
        self.path_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return torch.cat([self.path_3x3(x), self.path_5x5(x)], dim=1)

class CustomCoffeeCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCoffeeCNN, self).__init__()
        
        # Stage 1: Standard Conv (Input: 3x224x224 -> Output: 32x112x112)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: Standard Conv (Input: 32x112x112 -> Output: 64x56x56)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: Parallel Conv Paths (Input: 64x56x56 -> Output: 128x28x28)
        self.stage3 = nn.Sequential(
            ParallelConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: Parallel Conv Paths (Input: 128x28x28 -> Output: 256x14x14)
        self.stage4 = nn.Sequential(
            ParallelConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

# ==========================================
# 2. EARLY STOPPING UTILITY
# ==========================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0

# ==========================================
# 3. TRAINING ENGINE LOOP
# ==========================================

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

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================

def main():
    # Force CPU execution as requested
    device = torch.device("cpu")
    print(device)
    print(f"Running completely on: {device}\n")
    
    # Paths
    base_dir = "data/splits/task_b_3class"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    
    # Transformations & Augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
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
    
    # Explicit Class Balance Weighting calculation based on provided distributions
    # Counts: healthy: 791, coffee_leaf_rust: 602, red_spider_mite: 167 (Total: 1560)
    class_counts = torch.tensor([602.0, 791.0, 167.0]) # Alphabetical sort order: coffee_leaf_rust, healthy, red_spider_mite
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)
    
    print(f"Class mapping indices: {train_dataset.class_to_idx}")
    print(f"Calculated balancing loss weights: {class_weights}\n")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = CustomCoffeeCNN(num_classes=3).to(device)
    
    # ----------------------------------------------------
    # PHASE 1: Classifier Head Training (Backbone Frozen)
    # ----------------------------------------------------
    print("--- Starting Phase 1: Training Classification Head ---")
    model.freeze_backbone()
    
    # Filter optimizer to only track parameters requiring gradients
    optimizer_p1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    early_stopping_p1 = EarlyStopping(patience=5)
    
    for epoch in range(15):
        train_loss = train_epoch(model, train_loader, criterion, optimizer_p1, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:02d}/15 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        early_stopping_p1(val_loss, model)
        if early_stopping_p1.early_stop:
            print("Early stopping triggered for Phase 1.")
            break
            
    # Load best Phase 1 weights before transitioning
    model.load_state_dict(early_stopping_p1.best_model_wts)
    
    # ----------------------------------------------------
    # PHASE 2: Full Network Fine-Tuning (All Layers Unfrozen)
    # ----------------------------------------------------
    print("\n--- Starting Phase 2: Full Network Fine-Tuning ---")
    model.unfreeze_all()
    
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
            print("Early stopping triggered for Phase 2.")
            break
            
    # Load optimal model weights obtained across full training
    model.load_state_dict(early_stopping_p2.best_model_wts)
    
    # ----------------------------------------------------
    # 5. EVALUATION ON HELD-OUT TEST SET
    # ----------------------------------------------------
    print("\n--- Final Evaluation on Test Set ---")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    test_loss /= len(test_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metric Computations
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
    
    # Print formatted output
    print(f"Test Loss:             {test_loss:.4f}")
    print(f"Overall Accuracy:      {acc:.4f}")
    print(f"Balanced Accuracy:     {balanced_acc:.4f}")
    print(f"Macro F1-Score:        {macro_f1:.4f}")
    print(f"Macro OvR AUC:         {macro_auc:.4f}\n")
    
    print("Per-Class Recall:")
    for cls in test_dataset.classes:
        print(f"  - {cls:<18}: {report[cls]['recall']:.4f}")

if __name__ == "__main__":
    main()