import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def main():
    # ----------------------------------------------------
    # 1. Hyperparameters & Settings
    # ----------------------------------------------------
    DATA_DIR = "data/splits/task_a_binary"
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    
    # Phase 1: Warmup Classifier Head
    PHASE1_EPOCHS = 3
    PHASE1_LR = 1e-3
    
    # Phase 2: Fine-Tuning Entire Network
    PHASE2_EPOCHS = 5
    PHASE2_LR = 1e-5  # Noticeably smaller to protect learned weights
    
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----------------------------------------------------
    # 2. Data Augmentation and DataLoaders
    # ----------------------------------------------------
    # Standard ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    # Data augmentation is vital for a ~1500 image dataset
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    # Loading datasets via ImageFolder
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Dataset loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Class mapping: {train_dataset.class_to_idx}") # Ensure 'healthy' and 'unhealthy' map cleanly

    # ----------------------------------------------------
    # 3. Model Definition (ResNet-50)
    # ----------------------------------------------------
    # Using ResNet50 as a strong, standard feature extractor
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace the final FC layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Single output for binary (using BCEWithLogitsLoss)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # ----------------------------------------------------
    # Helper Training Function
    # ----------------------------------------------------
    def run_epoch(model, loader, optimizer, criterion, is_train=True):
        if is_train:
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1) # Match shapes for BCE
                
                if is_train:
                    optimizer.zero_grad()
                    
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if is_train:
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
                
                # Apply sigmoid to extract probabilities
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / len(loader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Convert probabilities to binary 0 or 1
        binary_preds = (all_preds >= 0.5).astype(int)
        epoch_acc = accuracy_score(all_labels, binary_preds)
        
        return epoch_loss, epoch_acc

    # ----------------------------------------------------
    # 4. Phase 1: Train Classifier Head Only (Feature Extraction)
    # ----------------------------------------------------
    print("\n--- Phase 1: Training Classifier Head Only ---")
    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze only the custom classifier layer
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=PHASE1_LR)

    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer_phase1, criterion, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion, is_train=False)
        print(f"Epoch {epoch+1}/{PHASE1_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # ----------------------------------------------------
    # 5. Phase 2: Fine-Tuning Whole Network
    # ----------------------------------------------------
    print("\n--- Phase 2: Fine-Tuning Entire Network ---")
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Use a significantly smaller learning rate for the backbone to keep features intact
    optimizer_phase2 = optim.Adam(model.parameters(), lr=PHASE2_LR)

    best_val_loss = float('inf')
    
    for epoch in range(PHASE2_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer_phase2, criterion, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion, is_train=False)
        print(f"Epoch {epoch+1}/{PHASE2_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Simple checkpoint mechanism to keep the best weights based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_coffee_model.pth')

    # ----------------------------------------------------
    # 6. Final Evaluation on Held-out Test Set
    # ----------------------------------------------------
    print("\n--- Final Evaluation on Test Set ---")
    # Load the best weights discovered during fine-tuning
    if os.path.exists('best_coffee_model.pth'):
        model.load_state_dict(torch.load('best_coffee_model.pth'))
    
    model.eval()
    test_preds_probs = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            test_preds_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_preds_probs = np.array(test_preds_probs)
    test_labels = np.array(test_labels)
    test_preds_binary = (test_preds_probs >= 0.5).astype(int)

    # Calculate requested performance metrics
    acc = accuracy_score(test_labels, test_preds_binary)
    prec = precision_score(test_labels, test_preds_binary, zero_division=0)
    rec = recall_score(test_labels, test_preds_binary, zero_division=0)
    f1 = f1_score(test_labels, test_preds_binary, zero_division=0)
    auc = roc_auc_score(test_labels, test_preds_probs)

    print("=========================================")
    print(f"Final Accuracy:  {acc:.4f}")
    print(f"Final Precision: {prec:.4f}")
    print(f"Final Recall:    {rec:.4f}")
    print(f"Final F1-Score:  {f1:.4f}")
    print(f"Final ROC-AUC:   {auc:.4f}")
    print("=========================================")

if __name__ == '__main__':
    main()