import os
import shutil
from pathlib import Path
import pandas as pd
import torch  # For load_state_dict
from train_eval import train_model, evaluate_model, save_model, load_model
from data_loader import get_dataloaders

datasets = ['asl', 'isl', 'bsl']
data_base = 'datasets'
results_dir = 'results'
new_sl_dir = 'new_sl'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(new_sl_dir, exist_ok=True)
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Mode selection
print("Select mode:")
print("1. Train/Eval (train if no model, eval if exists)")
print("2. Eval Only (load existing models, re-eval val sets)")
print("3. Check New SL Dataset (list images per letter)")
mode_choice = input("Enter 1, 2, or 3: ").strip()

if mode_choice == '3':
    # Mode 3 - Check New SL Dataset
    print(f"\n--- Checking New SL Dataset in {new_sl_dir}/ ---")
    if not os.path.exists(new_sl_dir):
        print("New SL dataset not found. Run mode 1 or 2 first to generate it.")
        exit(1)
    
    # Load best_per_letter from saved CSV (recompute if exists)
    selection_df = None
    contents_data = []
    if os.path.exists(os.path.join(results_dir, 'metrics_table.csv')):
        combined_df = pd.read_csv(os.path.join(results_dir, 'metrics_table.csv'), index_col=['Letter', 'dataset'])
        # Recompute best from F1
        best_per_letter = {}
        for letter in combined_df.index.levels[0]:
            f1_series = combined_df.loc[letter, 'f1-score']
            if not f1_series.empty:
                best_ds = f1_series.idxmax()
                best_f1 = f1_series.max()
                best_per_letter[letter] = {'dataset': best_ds.upper(), 'f1': best_f1}
        
        selection_data = []
        for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            if letter in best_per_letter:
                info = best_per_letter[letter]
                selection_data.append({
                    'Letter': letter,
                    'Selected Dataset': info['dataset'],
                    'Best F1-Score': f"{info['f1']:.4f}"
                })
            else:
                selection_data.append({
                    'Letter': letter,
                    'Selected Dataset': 'None (Missing in datasets)',
                    'Best F1-Score': 'N/A'
                })
        selection_df = pd.DataFrame(selection_data)
        print("\n--- Selected Letters Summary (from last run) ---")
        print(selection_df.to_string(index=False))
        
        # FIXED: Save to CSV
        summary_csv = os.path.join(results_dir, 'selected_letters_summary.csv')
        selection_df.to_csv(summary_csv, index=False)
        print(f"Saved selected letters summary to {summary_csv}")
    
    # List images in new_sl/ and save contents
    print("\n--- New SL Dataset Contents (1 image per letter) ---")
    total_images = 0
    for letter in sorted(os.listdir(new_sl_dir)):
        letter_dir = os.path.join(new_sl_dir, letter)
        if os.path.isdir(letter_dir):
            images = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            num_imgs = len(images)
            if num_imgs > 0:
                img_name = images[0]  # First image
                ds = selection_df.loc[selection_df['Letter'] == letter, 'Selected Dataset'].values[0] if selection_df is not None and letter in selection_df['Letter'].values else 'Unknown'
                print(f"Letter {letter}: {num_imgs} image(s) from {ds} - {img_name}")
                contents_data.append({'Letter': letter, 'Selected Dataset': ds, 'Image Name': img_name, 'Num Images': num_imgs})
                total_images += num_imgs
            else:
                print(f"Letter {letter}: No images (empty folder)")
                contents_data.append({'Letter': letter, 'Selected Dataset': 'None', 'Image Name': 'N/A', 'Num Images': 0})
        else:
            print(f"Letter {letter}: Invalid folder")
            contents_data.append({'Letter': letter, 'Selected Dataset': 'Invalid', 'Image Name': 'N/A', 'Num Images': 0})
    print(f"\nTotal letters with images: {total_images}/26")
    
    # FIXED: Save contents to CSV
    contents_df = pd.DataFrame(contents_data)
    contents_csv = os.path.join(results_dir, 'new_sl_contents.csv')
    contents_df.to_csv(contents_csv, index=False)
    print(f"Saved new SL contents to {contents_csv}")
    
    print("Check complete! Folders: new_sl/{A-Z}/")
    exit(0)  # End after check

# (Rest of code for modes 1/2)
if mode_choice == '2':
    force_eval = True  # Always re-eval, even if model exists
    selected_datasets = datasets  # Eval all by default
    print("Running Eval Only on all datasets...")
else:
    force_eval = False
    print("Available datasets: asl, isl, bsl")
    user_input = input("Enter datasets to process (comma-separated, e.g., 'bsl' or 'all' for all): ").strip().lower()
    if user_input == 'all':
        selected_datasets = datasets
    else:
        selected_datasets = [ds.strip() for ds in user_input.split(',') if ds.strip() in datasets]
    if not selected_datasets:
        print("No valid datasets processed. Exiting.")
        exit(1)
    print(f"Processing: {selected_datasets} (Train if needed)")

# Run for each selected dataset
all_metrics = {}
best_per_letter = {}  # {letter: best_dataset}

for ds in selected_datasets:
    data_dir = os.path.join(data_base, ds)
    if not os.path.exists(data_dir):
        print(f"Dataset {ds} not found in {data_dir}. Download from Kaggle.")
        continue
    
    model_path = os.path.join(models_dir, f'{ds}_model.pth')
    
    print(f"\n--- Processing {ds.upper()} ---")
    train_loader, val_loader, class_to_idx = get_dataloaders(data_dir)
    num_classes = len(class_to_idx)  # Use actual num_classes from valid keys
    
    if force_eval or os.path.exists(model_path):
        if os.path.exists(model_path):
            print(f"Model exists: Loading {model_path}.")
            model = load_model(num_classes)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Safe load
        else:
            print("No model found: Skipping (Eval Only mode requires existing models).")
            continue
        print("Running evaluation...")
    else:
        print("No model found: Training from scratch.")
        model = train_model(train_loader, val_loader, num_classes)
        save_model(model, model_path)
        print("Running evaluation...")
    
    metrics_df, _ = evaluate_model(model, val_loader, class_to_idx)
    metrics_df['dataset'] = ds.upper()
    all_metrics[ds] = metrics_df
    
    # Track best F1 per letter (only if letter in metrics)
    for letter in metrics_df.index:
        f1 = metrics_df.loc[letter, 'f1-score']
        if letter not in best_per_letter or f1 > best_per_letter[letter]['f1']:
            best_per_letter[letter] = {'dataset': ds, 'f1': f1, 'source_dir': data_dir}

# Generate comparison table with proper index handling
if all_metrics:
    # Concat, reset_index to make 'Letter' a column, then set multi-index
    combined_df = pd.concat([df.assign(dataset=ds) for ds, df in all_metrics.items()], ignore_index=False).reset_index(names='Letter')
    combined_df = combined_df.set_index(['Letter', 'dataset']).sort_index()
    table_path = os.path.join(results_dir, 'metrics_table.csv')
    combined_df.to_csv(table_path)
    print("\n--- Full Metrics Table (All Letters) ---")
    print(combined_df)  # Print FULL table
    
    # Print & Save Selected Letters Summary Table
    selection_data = []
    for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        if letter in best_per_letter:
            info = best_per_letter[letter]
            selection_data.append({
                'Letter': letter,
                'Selected Dataset': info['dataset'],
                'Best F1-Score': f"{info['f1']:.4f}"
            })
        else:
            selection_data.append({
                'Letter': letter,
                'Selected Dataset': 'None (Missing in datasets)',
                'Best F1-Score': 'N/A'
            })
    selection_df = pd.DataFrame(selection_data)
    print("\n--- Selected Letters Summary ---")
    print(selection_df.to_string(index=False))  # Clean table print
    
    # FIXED: Save to CSV
    summary_csv = os.path.join(results_dir, 'selected_letters_summary.csv')
    selection_df.to_csv(summary_csv, index=False)
    print(f"Saved selected letters summary to {summary_csv}")
    
    # Plot F1-scores for visualization
    import matplotlib.pyplot as plt
    f1_df = combined_df['f1-score'].unstack(level='dataset').fillna(0)
    f1_df.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title('F1-Score Comparison by Letter Across Datasets')
    plt.ylabel('F1-Score')
    plt.xlabel('Letters')
    plt.xticks(rotation=45)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'f1_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"F1 comparison plot saved to {plot_path}")
else:
    print("No datasets processed. No table generated.")

# Create new SL dataset (only if metrics available)
if best_per_letter:
    contents_data = []  # For CSV
    for letter, info in best_per_letter.items():
        src_dir = os.path.join(info['source_dir'], letter)
        if not os.path.exists(src_dir):
            print(f"Warning: Source dir for {letter} not found in {info['dataset']}. Skipping.")
            contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': 'N/A', 'Num Images': 0})
            continue
        dst_dir = os.path.join(new_sl_dir, letter)
        os.makedirs(dst_dir, exist_ok=True)
        
        # Copy exactly 1 image (first valid one; skip if none)
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            first_img = images[0]  # Take first
            shutil.copy(os.path.join(src_dir, first_img), os.path.join(dst_dir, first_img))
            print(f"Copied 1 image ({first_img}) for {letter} from {info['dataset']}")
            contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': first_img, 'Num Images': 1})
        else:
            print(f"No images for {letter} in {info['dataset']}. Skipping.")
            contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': 'N/A', 'Num Images': 0})

    print(f"\nNew hybrid dataset saved to {new_sl_dir}/ (1 image per letter)")
    
    # FIXED: Save contents to CSV (for modes 1/2)
    contents_df = pd.DataFrame(contents_data)
    contents_csv = os.path.join(results_dir, 'new_sl_contents.csv')
    contents_df.to_csv(contents_csv, index=False)
    print(f"Saved new SL contents to {contents_csv}")
    
    # Print contents summary (brief)
    print("\n--- New SL Dataset Contents Summary ---")
    for row in contents_df[contents_df['Num Images'] > 0].itertuples():
        print(f"Letter {row.Letter}: 1 image from {row.Selected_Dataset} - {row.Image_Name}")
print("Full table: results/metrics_table.csv")
print("Project complete! Best letters selected based on F1-score.")