import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import DSHTEEnsemble
import argparse

def load_model_for_dataset(dataset, num_classes=26):
    model_path = f'models/{dataset.lower()}_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = DSHTEEnsemble(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
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

def generate_report(letter):
    # Load summary
    summary_df = pd.read_csv('results/selected_letters_summary.csv')
    row = summary_df[summary_df['Letter'] == letter]
    if row.empty:
        raise ValueError(f"Letter {letter} not found in summary.")
    selected_dataset = row['Selected Dataset'].values[0].upper()
    f1_score = row['Best F1-Score'].values[0]
    
    # Load contents
    contents_df = pd.read_csv('results/new_sl_contents.csv')
    content_row = contents_df[contents_df['Letter'] == letter]
    if content_row.empty:
        raise ValueError(f"Letter {letter} not found in contents.")
    image_name = content_row['Image Name'].values[0]
    image_path = f'new_sl/{letter}/{image_name}'
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load model
    model = load_model_for_dataset(selected_dataset)
    
    # Predict
    ensemble_pred, ensemble_conf, ind_preds, ind_confs = predict_image(model, image_path)
    
    # Map indices to letters
    idx_to_letter = {i: chr(65+i) for i in range(26)}
    ensemble_letter = idx_to_letter[ensemble_pred]
    ind_letters = [idx_to_letter[p] for p in ind_preds]
    
    # Load metrics for more details
    metrics_df = pd.read_csv('results/metrics_table.csv')
    letter_metrics = metrics_df[(metrics_df['Letter'] == letter) & (metrics_df['dataset'] == selected_dataset.lower())]
    if not letter_metrics.empty:
        acc = letter_metrics['accuracy'].values[0]
        prec = letter_metrics['precision'].values[0]
        rec = letter_metrics['recall'].values[0]
        tpr = letter_metrics['tpr'].values[0]
        fpr = letter_metrics['fpr'].values[0]
    else:
        acc = prec = rec = tpr = fpr = 'N/A'
    
    # Generate report
    report = f"""
Detailed Report for Letter: {letter}
=====================================

Selected Dataset: {selected_dataset}
Best F1-Score: {f1_score}

Image Used: {image_name}
Image Path: {image_path}

Model Predictions:
- MobileNetV3: {ind_letters[0]} (Confidence: {ind_confs[0]:.4f})
- EfficientNet-B0: {ind_letters[1]} (Confidence: {ind_confs[1]:.4f})
- ResNet-18: {ind_letters[2]} (Confidence: {ind_confs[2]:.4f})

Ensemble Prediction (Voting): {ensemble_letter} (Confidence: {ensemble_conf:.4f})

Detailed Metrics from Validation:
- Accuracy: {acc}
- Precision: {prec}
- Recall: {rec}
- F1-Score: {f1_score}
- True Positive Rate (TPR): {tpr}
- False Positive Rate (FPR): {fpr}

Correct Prediction: {'Yes' if ensemble_letter == letter else 'No'}
"""
    
    # Save report
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)
    report_path = f'{report_dir}/report_{letter}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report generated for letter {letter}: {report_path}")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate detailed report for a selected sign language letter.")
    parser.add_argument('letter', type=str, help="The letter to check (e.g., A)")
    args = parser.parse_args()
    
    letter = args.letter.upper()
    if len(letter) != 1 or not letter.isalpha():
        print("Invalid letter. Please provide a single alphabetic character.")
        exit(1)
    
    generate_report(letter)