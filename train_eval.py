import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bars
from data_loader import get_dataloaders
from model import DSHTEEnsemble

def train_model(train_loader, val_loader, num_classes, epochs=10, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train ensemble with cross-entropy loss.
    """
    print(f"Using device: {device}")
    model = DSHTEEnsemble(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")
    
    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch+1}/{epochs}...")
        model.train()
        train_loss = 0
        # Progress bar for batches
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        batch_count = 0
        for imgs, labels in batch_iter:
            imgs, labels = imgs.to(device), labels.to(device)
            # Handle both old model (returns logits) and new model (returns tuple)
            model_output = model(imgs)
            if isinstance(model_output, tuple):
                logits, _ = model_output  # New model returns (logits, confidence)
                # New model already does fusion internally, no need to average
            else:
                logits = model_output  # Old model returns stacked [B, 3, C]
                logits = logits.mean(dim=1)  # Average ensemble for old model
            
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
            # Update progress with loss
            batch_iter.set_postfix({'Loss': f'{loss.item():.4f}'})
            # Optional: Print every 50 batches (fewer for speed)
            if batch_count % 50 == 0:
                print(f"  Batch {batch_count}/{len(train_loader)} - Avg Loss so far: {train_loss / batch_count:.4f}")
        
        # Val
        print("Evaluating validation...")
        model.eval()
        val_correct = 0
        val_total = 0
        val_iter = tqdm(val_loader, desc="Val", leave=False)
        with torch.no_grad():
            for imgs, labels in val_iter:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds, _ = model.predict_with_voting(imgs)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1} Complete: Train Loss {avg_train_loss:.4f}, Val Acc {val_acc:.4f}')
    
    return model

def evaluate_model(model, val_loader, class_to_idx, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate: Per-class metrics + overall.
    Returns DF with accuracy, prec, rec, f1, tpr(rec), fpr per class.
    """
    print("Running evaluation...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Eval", leave=False):
            imgs = imgs.to(device)
            preds, _ = model.predict_with_voting(imgs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    if not all_labels:  # Handle empty val_loader
        print("Warning: No validation samples. Skipping metrics.")
        empty_df = pd.DataFrame(index=sorted(class_to_idx.keys()), columns=['accuracy', 'precision', 'recall', 'f1-score', 'tpr', 'fpr']).fillna(0)
        empty_df.index.name = 'Letter'
        return empty_df, 0.0
    
    # Use valid class keys only (handles missing like BSL H/J)
    target_names = sorted(class_to_idx.keys())  # Valid class names in order
    num_classes = len(target_names)
    labels_range = list(range(num_classes))  # Sequential indices for all possible classes
    
    # Specify labels to include all classes, even if absent in val set
    report = classification_report(all_labels, all_preds, labels=labels_range, target_names=target_names, output_dict=True, zero_division=0)
    
    # Confusion matrix with full labels for consistent shape
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)
    
    metrics = {}
    for i, cls in enumerate(target_names):
        idx = i  # Direct index since labels_range is 0 to num_classes-1
        tn = cm.sum() - (cm[idx].sum() + cm[:, idx].sum() - cm[idx, idx])  # Proper TN
        fp = cm[:, idx].sum() - cm[idx, idx]
        fn = cm[idx, :].sum() - cm[idx, idx]
        tp = cm[idx, idx]
        prec = report[f'{cls}']['precision']
        rec = report[f'{cls}']['recall']
        f1 = report[f'{cls}']['f1-score']
        acc = (tp + tn) / cm.sum() if cm.sum() > 0 else 0  # Per-class acc
        tpr = rec
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics[cls] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1-score': f1, 'tpr': tpr, 'fpr': fpr}
    
    # DF
    df = pd.DataFrame(metrics).T
    df.index.name = 'Letter'
    # Robust overall_acc handling
    overall_acc = report.get('accuracy', 0.0) if 'accuracy' in report else 0.0
    print(f'Overall Accuracy: {overall_acc:.4f}')
    
    return df, overall_acc

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load empty model for state_dict loading (skip training).
    """
    print(f"Loading model structure on {device}...")
    model = DSHTEEnsemble(num_classes).to(device)
    return model