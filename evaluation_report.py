import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_dataloaders
from model import DSHTEEnsemble
from train_eval import load_model

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Datasets
datasets = ['asl', 'bsl', 'isl']
data_base = 'datasets'
models_dir = 'models'
results_dir = 'results'
new_sl_dir = 'new_sl'

os.makedirs(results_dir, exist_ok=True)
os.makedirs(new_sl_dir, exist_ok=True)

def evaluate_dataset_level(model, val_loader, class_to_idx):
    """
    Evaluate model and return dataset-level metrics and confusion matrix.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds, _ = model.predict_with_voting(imgs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    if not all_labels:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1-score': 0}, None

    # Get classification report
    target_names = sorted(class_to_idx.keys())
    num_classes = len(target_names)
    labels_range = list(range(num_classes))

    report = classification_report(all_labels, all_preds, labels=labels_range, target_names=target_names, output_dict=True, zero_division=0)

    # Dataset-level metrics (macro avg)
    accuracy = report.get('accuracy', 0.0) * 100
    precision = report['macro avg']['precision'] * 100
    recall = report['macro avg']['recall'] * 100
    f1 = report['macro avg']['f1-score'] * 100

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1-score': f1}, cm

def plot_confusion_matrix(cm, classes, title, save_path):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_universal_alphabet():
    """
    Create hybrid universal alphabet U by selecting best images based on F1 scores.
    """
    if not os.path.exists(os.path.join(results_dir, 'metrics_table.csv')):
        print("Metrics table not found. Run evaluation first.")
        return

    combined_df = pd.read_csv(os.path.join(results_dir, 'metrics_table.csv'), index_col=['Letter', 'dataset'])

    best_per_letter = {}
    for letter in combined_df.index.levels[0]:
        f1_series = combined_df.loc[letter, 'f1-score']
        if not f1_series.empty:
            best_ds = f1_series.idxmax()
            best_f1 = f1_series.max()
            best_per_letter[letter] = {'dataset': best_ds.lower(), 'f1': best_f1}

    import shutil
    contents_data = []
    for letter, info in best_per_letter.items():
        src_dir = os.path.join(data_base, info['dataset'], letter)
        if not os.path.exists(src_dir):
            print(f"Warning: Source dir for {letter} not found in {info['dataset']}.")
            contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': 'N/A', 'Num Images': 0})
            continue
        dst_dir = os.path.join(new_sl_dir, letter)
        os.makedirs(dst_dir, exist_ok=True)

        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            first_img = images[0]
            shutil.copy(os.path.join(src_dir, first_img), os.path.join(dst_dir, first_img))
            contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': first_img, 'Num Images': 1})
        else:
            contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': 'N/A', 'Num Images': 0})

    contents_df = pd.DataFrame(contents_data)
    contents_csv = os.path.join(results_dir, 'new_sl_contents.csv')
    contents_df.to_csv(contents_csv, index=False)
    print(f"Universal alphabet created with {len([c for c in contents_data if c['Num Images'] > 0])} letters.")

# Main evaluation
results = {}
confusion_matrices = {}

for ds in datasets:
    data_dir = os.path.join(data_base, ds)
    if not os.path.exists(data_dir):
        print(f"Dataset {ds} not found.")
        continue

    model_path = os.path.join(models_dir, f'{ds}_model.pth')
    if not os.path.exists(model_path):
        print(f"Model for {ds} not found.")
        continue

    print(f"Evaluating {ds.upper()}...")
    _, val_loader, class_to_idx = get_dataloaders(data_dir, max_samples=1000)  # Limit to 1000 samples for faster eval
    num_classes = len(class_to_idx)
    model = load_model(num_classes, device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    metrics, cm = evaluate_dataset_level(model, val_loader, class_to_idx)
    results[ds.upper()] = metrics
    confusion_matrices[ds.upper()] = cm

    # Plot confusion matrix
    classes = sorted(class_to_idx.keys())
    plot_confusion_matrix(cm, classes, f'Confusion Matrix for {ds.upper()}', os.path.join(results_dir, f'confusion_matrix_{ds}.png'))

# Create table
table_data = []
for lang in ['ASL', 'BSL', 'ISL']:
    if lang in results:
        m = results[lang]
        table_data.append({
            'Language': lang,
            'Acc (%)': f"{m['accuracy']:.2f}",
            'Prec (%)': f"{m['precision']:.2f}",
            'Rec (%)': f"{m['recall']:.2f}",
            'F1 (%)': f"{m['f1-score']:.2f}"
        })
    else:
        table_data.append({
            'Language': lang,
            'Acc (%)': '–',
            'Prec (%)': '–',
            'Rec (%)': '–',
            'F1 (%)': '–'
        })

table_df = pd.DataFrame(table_data)
print("\nTABLE I: Dataset-Level Validation Performance")
print(table_df.to_string(index=False))

# Save table
table_df.to_csv(os.path.join(results_dir, 'dataset_level_performance.csv'), index=False)

# Generate per-letter F1 table
if os.path.exists(os.path.join(results_dir, 'metrics_table.csv')):
    combined_df = pd.read_csv(os.path.join(results_dir, 'metrics_table.csv'), index_col=['Letter', 'dataset'])
    f1_table_data = []
    for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        row = {'Letter': letter}
        for ds in ['ASL', 'BSL', 'ISL']:
            if (letter, ds) in combined_df.index:
                f1 = combined_df.loc[(letter, ds), 'f1-score'] * 100
                row[f'{ds} F1 (%)'] = f"{f1:.2f}"
            else:
                row[f'{ds} F1 (%)'] = '–'
        f1_table_data.append(row)
    f1_table_df = pd.DataFrame(f1_table_data)
    print("\nPer-Letter F1 Scores")
    print(f1_table_df.to_string(index=False))
    f1_table_df.to_csv(os.path.join(results_dir, 'per_letter_f1.csv'), index=False)

# Generate selected language table
if os.path.exists(os.path.join(results_dir, 'selected_letters_summary.csv')):
    selected_df = pd.read_csv(os.path.join(results_dir, 'selected_letters_summary.csv'))
    selected_table_data = []
    for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        row = selected_df[selected_df['Letter'] == letter]
        if not row.empty:
            lang = row['Selected Dataset'].values[0]
            selected_table_data.append({'Letter': letter, 'Selected Language': lang})
        else:
            selected_table_data.append({'Letter': letter, 'Selected Language': '–'})
    selected_table_df = pd.DataFrame(selected_table_data)
    print("\nSelected Language per Letter")
    print(selected_table_df.to_string(index=False))
    selected_table_df.to_csv(os.path.join(results_dir, 'selected_language_per_letter.csv'), index=False)

# Create universal alphabet
create_universal_alphabet()

print("Evaluation complete. Check results/ for outputs.")