import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import random

class CustomImageFolder(Dataset):
    """Custom dataset to handle empty classes."""
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        self.targets = []
        
        # Build samples: Only include non-empty classes
        for class_name, class_idx in class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                valid_files = [f for f in os.listdir(class_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                for file in valid_files:
                    self.samples.append((os.path.join(class_dir, file), class_idx))
                    self.targets.append(class_idx)
        
        if not self.samples:
            raise ValueError("No valid images found in dataset.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

def get_dataloaders(data_dir, batch_size=16, val_split=0.2, max_samples=None, sample_ratio=1.0):
    """
    Load images from folder (A-Z subdirs), skip empty classes, split train/val.
    Assumes data_dir/{A-Z}/images.
    sample_ratio: float (0.0-1.0), fraction of dataset to use (e.g., 0.1 for 10%)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Build class_to_idx only for A-Z subdirs with files
    all_subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    valid_classes = []
    class_to_idx = {}
    class_idx = 0
    for subdir in sorted(all_subdirs):  # Sorted for consistent order
        if subdir.isalpha() and len(subdir) == 1:  # A-Z single letters
            class_dir = os.path.join(data_dir, subdir)
            valid_files = [f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            if valid_files:  # Non-empty
                valid_classes.append(subdir)
                class_to_idx[subdir] = class_idx
                class_idx += 1
    
    num_classes = len(valid_classes)
    if num_classes == 0:
        raise ValueError("No valid A-Z classes with images found.")
    if num_classes < 26:
        missing = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(valid_classes)
        print(f"Warning: Found {num_classes} classes (missing: {', '.join(sorted(missing))}). Expected 26.")
    
    # Create custom dataset
    full_dataset = CustomImageFolder(data_dir, class_to_idx, transform=transform)
    
    # Apply sample_ratio if less than 1.0
    if sample_ratio < 1.0:
        target_samples = int(len(full_dataset) * sample_ratio)
        print(f"Using {sample_ratio*100:.0f}% of dataset: {target_samples}/{len(full_dataset)} images.")
        sampled_indices = []
        per_class_target = max(1, target_samples // num_classes)
        for cls_idx in range(num_classes):
            cls_indices = [i for i, tgt in enumerate(full_dataset.targets) if tgt == cls_idx]
            num_sample = min(len(cls_indices), per_class_target)
            sampled = random.sample(cls_indices, num_sample)
            sampled_indices.extend(sampled)
        full_dataset = torch.utils.data.Subset(full_dataset, sampled_indices)
    # Subsample for speed (random up to max_samples, balanced if possible)
    elif max_samples and len(full_dataset) > max_samples:
        print(f"Subsampling to {max_samples} images for faster training.")
        sampled_indices = []
        per_class_target = max_samples // num_classes
        for cls_idx in range(num_classes):
            cls_indices = [i for i, tgt in enumerate(full_dataset.targets) if tgt == cls_idx]
            num_sample = min(len(cls_indices), per_class_target)
            sampled = random.sample(cls_indices, num_sample)
            sampled_indices.extend(sampled)
        full_dataset = torch.utils.data.Subset(full_dataset, sampled_indices)
    
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Handle edge case for small datasets
    if val_size == 0 and len(full_dataset) > 0:
        val_size = 1
        train_size -= 1
    
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded: {len(train_ds)} train, {len(val_ds)} val samples across {num_classes} classes.")
    
    return train_loader, val_loader, class_to_idx  # Return class_to_idx for eval