import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import DSHTEEnsemble
import argparse

def load_model_for_dataset(dataset):
    model_path = f'models/{dataset.lower()}_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    # Load state_dict to determine num_classes
    state_dict = torch.load(model_path, map_location='cpu')
    # Assume all classifiers have the same num_classes
    num_classes = state_dict['mobilenet.classifier.3.weight'].shape[0]
    model = DSHTEEnsemble(num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(img_tensor)  # [1, 3, 26]
        probs = torch.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=1)  # [1, 26]
        ensemble_pred = mean_probs.argmax(dim=1).item()
        ensemble_conf = mean_probs.max(dim=1)[0].item()
        
        # Individual predictions
        individual_preds = logits.argmax(dim=-1).squeeze(0).tolist()  # [3]
        individual_confs = probs.max(dim=-1)[0].squeeze(0).tolist()  # [3]
    
    return ensemble_pred, ensemble_conf, individual_preds, individual_confs

def generate_tabular_report():
    # Load summary
    summary_df = pd.read_csv('results/selected_letters_summary.csv')
    contents_df = pd.read_csv('results/new_sl_contents.csv')
    metrics_df = pd.read_csv('results/metrics_table.csv')
    
    # Filter to selected letters (those with a dataset, not 'None')
    selected_letters = summary_df[summary_df['Selected Dataset'] != 'None (Missing in datasets)']['Letter'].tolist()
    
    # Cache models
    models = {}
    
    results = []
    
    for letter in selected_letters:
        row = summary_df[summary_df['Letter'] == letter]
        selected_dataset = row['Selected Dataset'].values[0].upper()
        f1_score = row['Best F1-Score'].values[0]
        
        content_row = contents_df[contents_df['Letter'] == letter]
        if content_row.empty:
            continue
        image_name = content_row['Image Name'].values[0]
        image_path = f'new_sl/{letter}/{image_name}'
        if not os.path.exists(image_path):
            continue
        
        # Load model if not cached
        if selected_dataset not in models:
            models[selected_dataset] = load_model_for_dataset(selected_dataset)
        model = models[selected_dataset]
        
        # Predict
        ensemble_pred, ensemble_conf, ind_preds, ind_confs = predict_image(model, image_path)
        
        # Map indices to letters
        idx_to_letter = {i: chr(65+i) for i in range(26)}
        ensemble_letter = idx_to_letter[ensemble_pred]
        ind_letters = [idx_to_letter[p] for p in ind_preds]
        
        # Metrics
        letter_metrics = metrics_df[(metrics_df['Letter'] == letter) & (metrics_df['dataset'] == selected_dataset.lower())]
        if not letter_metrics.empty:
            acc = letter_metrics['accuracy'].values[0]
            prec = letter_metrics['precision'].values[0]
            rec = letter_metrics['recall'].values[0]
            tpr = letter_metrics['tpr'].values[0]
            fpr = letter_metrics['fpr'].values[0]
        else:
            acc = prec = rec = tpr = fpr = 'N/A'
        
        correct = 'Yes' if ensemble_letter == letter else 'No'
        
        results.append({
            'Letter': letter,
            'Selected Dataset': selected_dataset,
            'F1-Score': f1_score,
            'Image Name': image_name,
            'MobileNet Prediction': ind_letters[0],
            'MobileNet Confidence': f"{ind_confs[0]:.4f}",
            'EfficientNet Prediction': ind_letters[1],
            'EfficientNet Confidence': f"{ind_confs[1]:.4f}",
            'ResNet Prediction': ind_letters[2],
            'ResNet Confidence': f"{ind_confs[2]:.4f}",
            'Ensemble Prediction': ensemble_letter,
            'Ensemble Confidence': f"{ensemble_conf:.4f}",
            'Correct': correct,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'TPR': tpr,
            'FPR': fpr
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = 'results/all_letters_tabular_report.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Tabular report generated: {output_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    generate_tabular_report()