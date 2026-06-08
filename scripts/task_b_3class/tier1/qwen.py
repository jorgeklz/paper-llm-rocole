import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    recall_score, roc_auc_score
)

def main():
    # Reproducibility & Device Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')  # Strictly CPU as requested
    
    DATA_DIR = 'data/splits/task_b_3class'
    BATCH_SIZE = 32
    PHASE1_EPOCHS = 10
    PHASE2_EPOCHS = 20

    # 1. Data Loading & Transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        split: datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=data_transforms[split])
        for split in ['train', 'val', 'test']
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split], 
            batch_size=BATCH_SIZE, 
            shuffle=(split == 'train'), 
            num_workers=0
        )
        for split in ['train', 'val', 'test']
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    dataset_sizes = {split: len(image_datasets[split]) for split in ['train', 'val', 'test']}

    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # 2. Handle Class Imbalance (Inverse Frequency Weighting)
    train_targets = [label for _, label in image_datasets['train'].samples]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    total_samples = sum(class_counts)
    # Standard inverse-frequency weighting
    class_weights = torch.tensor(total_samples / (num_classes * class_counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model Initialization (ResNet18 + Pretrained Weights)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final classifier layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Track best validation model across both phases
    best_val_acc = 0.0
    best_model_state = None

    def train_and_validate(model, optimizer, num_epochs, phase_name):
        nonlocal best_val_acc, best_model_state
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            # Validation step
            model.eval()
            val_correct = 0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += torch.sum(preds == labels.data)

            epoch_train_loss = running_loss / dataset_sizes['train']
            epoch_val_loss = val_loss / dataset_sizes['val']
            epoch_val_acc = val_correct.double() / dataset_sizes['val']

            print(f'{phase_name} | Epoch {epoch+1}/{num_epochs} | '
                  f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                # Deep copy state dict to avoid reference issues
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ================= PHASE 1: Train Classifier Head =================
    print("\n" + "="*50)
    print("Phase 1: Training classifier head (Backbone FROZEN)")
    print("="*50)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer_phase1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    train_and_validate(model, optimizer_phase1, PHASE1_EPOCHS, "Phase 1")
    
    # Restore best weights before fine-tuning
    model.load_state_dict(best_model_state)

    # ================= PHASE 2: Fine-tune Entire Network =================
    print("\n" + "="*50)
    print("Phase 2: Fine-tuning entire network (Backbone UNFROZEN)")
    print("="*50)
    
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-5)
    train_and_validate(model, optimizer_phase2, PHASE2_EPOCHS, "Phase 2")

    # Load the globally best model for final evaluation
    print(f"\nBest validation accuracy achieved: {best_val_acc:.4f}")
    model.load_state_dict(best_model_state)
    model.eval()

    # ================= 4. Evaluation on Held-Out Test Set =================
    print("\n" + "="*50)
    print("Evaluating on Test Set...")
    print("="*50)
    
    true_labels = []
    pred_labels = []
    pred_probs = []
    test_losses = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            test_losses.append(loss.item())
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            pred_probs.extend(probs.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)
    avg_test_loss = np.mean(test_losses)

    # Compute requested metrics
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    bal_acc = balanced_accuracy_score(true_labels, pred_labels)
    per_class_recall = recall_score(true_labels, pred_labels, average=None)
    
    # Safe AUC computation (handles edge cases where a class might be absent in test predictions)
    try:
        macro_auc_ovr = roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr')
    except ValueError:
        macro_auc_ovr = float('nan')

    # ================= 5. Print Final Metrics =================
    print("\n" + "="*45)
    print("FINAL TEST METRICS")
    print("="*45)
    print(f"Accuracy:            {acc:.4f}")
    print(f"Macro-F1:            {macro_f1:.4f}")
    print(f"Balanced Accuracy:   {bal_acc:.4f}")
    print(f"Test Loss:           {avg_test_loss:.4f}")
    print(f"Macro OvR AUC:       {macro_auc_ovr:.4f}")
    print("\nPer-class Recall:")
    for i, cls_name in enumerate(class_names):
        print(f"  {cls_name:<25} : {per_class_recall[i]:.4f}")
    print("="*45)

if __name__ == '__main__':
    main()