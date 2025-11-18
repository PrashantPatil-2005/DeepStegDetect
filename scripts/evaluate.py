"""
Evaluate trained model on test set and generate metrics including confusion matrix.

Usage:
    python scripts/evaluate.py --model checkpoints/resnet50_best.pth --data_dir processed
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.modified_resnet import create_model


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Get model name from checkpoint or infer from filename
    model_name = ckpt.get('model_name', 'resnet50')
    if 'model_name' not in ckpt:
        # Try to infer from filename
        if 'resnet18' in checkpoint_path.lower():
            model_name = 'resnet18'
        elif 'resnet101' in checkpoint_path.lower():
            model_name = 'resnet101'
        else:
            model_name = 'resnet50'
    
    # Create model
    model = create_model(model_name=model_name, pretrained=False, freeze_until='none', device=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
    return model, model_name


def get_dataloader(data_dir, img_size, batch_size=8, split='test'):
    """Create dataloader for specified split"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return loader, dataset


def plot_confusion_matrix(y_true, y_pred, save_path='outputs/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cover', 'Stego'], 
                yticklabels=['Cover', 'Stego'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.close()


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print(f"📁 Model checkpoint: {args.model}")
    print(f"📁 Data directory: {args.data_dir}")
    
    # Load model
    print("\n📦 Loading model...")
    model, model_name = load_model(args.model, device)
    print(f"✅ Loaded {model_name} model")
    
    # Create test dataloader
    test_loader, test_dataset = get_dataloader(args.data_dir, args.img_size, args.batch_size, split='test')
    print(f"📊 Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\n🔄 Running evaluation...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating", ncols=100):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    y_pred_probs = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Cover  Stego")
    print(f"True  Cover    {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"      Stego    {cm[1][0]:5d}  {cm[1][1]:5d}")
    print(f"{'='*60}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, args.output)
    
    # Save metrics to file
    metrics_file = args.output.replace('.png', '_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"              Cover  Stego\n")
        f.write(f"True  Cover    {cm[0][0]:5d}  {cm[0][1]:5d}\n")
        f.write(f"      Stego    {cm[1][0]:5d}  {cm[1][1]:5d}\n")
    
    print(f"📄 Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="processed", 
                        help="Path to processed dataset directory")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output", type=str, default="outputs/confusion_matrix.png",
                        help="Output path for confusion matrix")
    
    args = parser.parse_args()
    evaluate(args)

