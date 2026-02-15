import os
import shutil
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from train_eval import train_model, evaluate_model, save_model, load_model
from data_loader import get_dataloaders

DATASETS = ['asl', 'bsl', 'isl']
DATA_BASE = 'datasets'
RESULTS_DIR = 'results'
NEW_SL_DIR = 'new_sl'
MODELS_DIR = 'models'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(NEW_SL_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def run_pipeline(datasets, train, eval_only, batch_size, epochs, sample_ratio, max_samples, create_universal):
    """
    PBSS pipeline:
    - Train 3 independent classifiers (ASL, BSL, ISL) or load existing
    - Evaluate and produce per-letter macro-F1
    - PBSS: pick best language per letter by F1
    - Generate metrics tables and universal alphabet dataset (new_sl)
    """
    all_metrics = {}
    best_per_letter = {}

    for ds in datasets:
        data_dir = os.path.join(DATA_BASE, ds)
        if not os.path.exists(data_dir):
            print(f"Dataset {ds} not found in {data_dir}. Skipping.")
            continue

        print(f"\n--- Processing {ds.upper()} ---")
        train_loader, val_loader, class_to_idx = get_dataloaders(
            data_dir,
            batch_size=batch_size,
            val_split=0.2,
            max_samples=max_samples,
            sample_ratio=sample_ratio,
        )
        num_classes = len(class_to_idx)
        model_path = os.path.join(MODELS_DIR, f"{ds}_model.pth")

        # Training / Loading
        if eval_only:
            if not os.path.exists(model_path):
                print(f"No model for {ds} at {model_path}; skipping (eval-only).")
                continue
            model = load_model(num_classes)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded model for {ds} from {model_path}.")
        else:
            if not os.path.exists(model_path) or train:
                print("Training from scratch...")
                model = train_model(train_loader, val_loader, num_classes, epochs=epochs)
                save_model(model, model_path)
            else:
                print(f"Model exists: Loading {model_path}.")
                model = load_model(num_classes)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # Evaluation
        metrics_df, overall_acc = evaluate_model(model, val_loader, class_to_idx)
        metrics_df['dataset'] = ds.upper()
        all_metrics[ds] = metrics_df

        # Track PBSS best by F1
        for letter in metrics_df.index:
            f1 = metrics_df.loc[letter, 'f1-score']
            if letter not in best_per_letter or f1 > best_per_letter[letter]['f1']:
                best_per_letter[letter] = {'dataset': ds.upper(), 'f1': f1, 'source_dir': data_dir}

    # Build combined metrics table
    if not all_metrics:
        print("No datasets processed; exiting.")
        return

    combined_df = pd.concat(
        [df.assign(dataset=ds.upper()) for ds, df in all_metrics.items()],
        ignore_index=False,
    ).reset_index(names='Letter')
    combined_df = combined_df.set_index(['Letter', 'dataset']).sort_index()
    metrics_csv = os.path.join(RESULTS_DIR, 'metrics_table.csv')
    combined_df.to_csv(metrics_csv)
    print(f"Saved metrics table: {metrics_csv}")

    # Selected letters summary (PBSS)
    selection_data = []
    for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        if letter in best_per_letter:
            info = best_per_letter[letter]
            selection_data.append({
                'Letter': letter,
                'Selected Dataset': info['dataset'],
                'Best F1-Score': f"{info['f1']:.4f}",
            })
        else:
            selection_data.append({
                'Letter': letter,
                'Selected Dataset': 'None (Missing)',
                'Best F1-Score': 'N/A',
            })
    selection_df = pd.DataFrame(selection_data)
    summary_csv = os.path.join(RESULTS_DIR, 'selected_letters_summary.csv')
    selection_df.to_csv(summary_csv, index=False)
    print(f"Saved PBSS summary: {summary_csv}")

    # Per-letter F1 table
    f1_table_rows = []
    for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        row = {'Letter': letter}
        for ds in ['ASL', 'BSL', 'ISL']:
            key = (letter, ds)
            if key in combined_df.index:
                f1 = combined_df.loc[key, 'f1-score'] * 100
                row[f'{ds} F1 (%)'] = f"{f1:.2f}"
            else:
                row[f'{ds} F1 (%)'] = '–'
        f1_table_rows.append(row)
    f1_df = pd.DataFrame(f1_table_rows)
    per_letter_csv = os.path.join(RESULTS_DIR, 'per_letter_f1.csv')
    f1_df.to_csv(per_letter_csv, index=False)
    print(f"Saved per-letter F1: {per_letter_csv}")

    # Selected language per letter
    selected_lang_rows = []
    for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        row = selection_df[selection_df['Letter'] == letter]
        lang = row['Selected Dataset'].values[0] if not row.empty else '–'
        selected_lang_rows.append({'Letter': letter, 'Selected Language': lang})
    selected_lang_df = pd.DataFrame(selected_lang_rows)
    selected_lang_csv = os.path.join(RESULTS_DIR, 'selected_language_per_letter.csv')
    selected_lang_df.to_csv(selected_lang_csv, index=False)
    print(f"Saved selected language per letter: {selected_lang_csv}")

    # Create universal alphabet dataset (one image per letter)
    if create_universal:
        contents_data = []
        for letter, info in best_per_letter.items():
            src_dir = os.path.join(info['source_dir'], letter)
            if not os.path.exists(src_dir):
                print(f"Warning: Missing source {src_dir}")
                contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': 'N/A', 'Num Images': 0})
                continue
            dst_dir = os.path.join(NEW_SL_DIR, letter)
            os.makedirs(dst_dir, exist_ok=True)
            images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                first_img = images[0]
                shutil.copy(os.path.join(src_dir, first_img), os.path.join(dst_dir, first_img))
                contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': first_img, 'Num Images': 1})
            else:
                contents_data.append({'Letter': letter, 'Selected Dataset': info['dataset'], 'Image Name': 'N/A', 'Num Images': 0})
        contents_df = pd.DataFrame(contents_data)
        contents_csv = os.path.join(RESULTS_DIR, 'new_sl_contents.csv')
        contents_df.to_csv(contents_csv, index=False)
        print(f"Saved universal dataset contents: {contents_csv}")
        print(f"Universal dataset created in {NEW_SL_DIR}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PBSS Pipeline: Train/Eval/Select universal alphabet by F1.')
    parser.add_argument('--datasets', type=str, default='asl,bsl,isl', help='Comma-separated list among asl,bsl,isl')
    parser.add_argument('--train', action='store_true', help='Force training even if model exists')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate existing models (skip training)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--sample-ratio', type=float, default=1.0, help='Fraction of dataset to use (0-1)')
    parser.add_argument('--max-samples', type=int, default=None, help='Cap number of images for speed')
    parser.add_argument('--no-universal', action='store_true', help='Skip creating the universal dataset folder')
    args = parser.parse_args()

    datasets = [ds.strip().lower() for ds in args.datasets.split(',') if ds.strip().lower() in DATASETS]
    if not datasets:
        print('No valid datasets specified; use --datasets asl,bsl,isl')
        raise SystemExit(1)

    run_pipeline(
        datasets=datasets,
        train=args.train,
        eval_only=args.eval_only,
        batch_size=args.batch_size,
        epochs=args.epochs,
        sample_ratio=args.sample_ratio,
        max_samples=args.max_samples,
        create_universal=(not args.no_universal),
    )
