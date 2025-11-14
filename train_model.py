import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out_dir', default='checkpoints')
    return parser.parse_args()


def get_dataloaders(data_dir, batch_size, size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=4,
            pin_memory=True
        )
        for x in ['train', 'val', 'test']
    }

    return dataloaders, image_datasets


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloaders, datasets_obj = get_dataloaders(args.data_dir, args.batch_size)

    print(f"Using device: {device}")

    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze early layers - train only layer4 + fc
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    model = model.to(device)

    # Loss function + optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=3,
        verbose=True
    )

    best_val_loss = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n──── Epoch {epoch}/{args.epochs} ────")

        model.train()
        running_loss = 0.0

        for xb, yb in dataloaders['train']:
            xb = xb.to(device)
            yb = yb.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(datasets_obj['train'])
        print(f"Train Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        preds = []
        labels = []

        with torch.no_grad():
            for xb, yb in dataloaders['val']:
                xb = xb.to(device)
                yb = yb.float().unsqueeze(1).to(device)

                outputs = model(xb)
                loss = criterion(outputs, yb)

                val_loss += loss.item() * xb.size(0)

                predictions = (outputs.cpu().numpy() > 0.5).astype(int)
                preds.extend(predictions.flatten().tolist())
                labels.extend(yb.cpu().numpy().flatten().tolist())

        val_loss /= len(datasets_obj['val'])
        val_acc = accuracy_score(labels, preds)

        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Save best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.out_dir, 'resnet50_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved Best Model → {save_path}")

    print("\n🔥 Training Complete!")


if __name__ == '__main__':
    args = parse_args()
    train(args)
