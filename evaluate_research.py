# evaluate_research.py
# Evaluate saved best_research.pth on test set and print metrics + confusion matrix.
# Usage:
# python evaluate_research.py --data_dir processed --ckpt checkpoints_research/best_research.pth --img_size 256

import argparse
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from srm_frontend import SRMFrontend
from pathlib import Path
from train_research import ResNetWithSRM  # uses same class

def load_dataloader(data_dir, img_size, batch_size=8):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    ds = datasets.ImageFolder(Path(data_dir) / 'test', transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader, ds = load_dataloader(args.data_dir, args.img_size, args.batch_size)

    model = ResNetWithSRM(device=device, pretrained=True, freeze_until='layer3')
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    y_pred = np.hstack(all_probs)
    y_true = np.hstack(all_labels)
    y_hat = (y_pred > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    print("Accuracy:", accuracy_score(y_true, y_hat))
    print("Precision:", precision_score(y_true, y_hat, zero_division=0))
    print("Recall:", recall_score(y_true, y_hat, zero_division=0))
    print("F1:", f1_score(y_true, y_hat, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_hat))

if __name__ == "__main__":
    main()
