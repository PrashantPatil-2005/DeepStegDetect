"""
Train ResNet models (ResNet18/50/101) with SRM preprocessing for steganalysis.
Supports transfer learning with configurable freezing strategies.

Usage:
    python scripts/train_resnet_transfer.py --model resnet50 --data_dir processed --epochs 15 --batch 8
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.modified_resnet import create_model


def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_f1(preds, labels):
    """Compute F1 score robustly on binary 0/1 tensors"""
    preds = preds.int()
    labels = labels.int()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def get_dataloaders(data_dir, batch_size, img_size):
    """Create train, validation, and test dataloaders"""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)
    
    # Windows-friendly: num_workers=0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}, {"train": train_ds, "val": val_ds, "test": test_ds}


def train(args):
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print(f"📊 Model: {args.model}")
    print(f"📁 Data directory: {args.data_dir}")
    
    # Create dataloaders
    loaders, dsets = get_dataloaders(args.data_dir, args.batch_size, args.img_size)
    print(f"📦 Train samples: {len(dsets['train'])}, Val samples: {len(dsets['val'])}, Test samples: {len(dsets['test'])}")
    
    # Create model
    model = create_model(
        model_name=args.model,
        pretrained=True,
        freeze_until=args.freeze_until,
        device=device
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    
    # Training loop
    best_val_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        pbar = tqdm(loaders["train"], desc="Train", ncols=100)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            
            pbar.set_postfix(
                loss=f"{running_loss/running_total:.4f}",
                acc=f"{running_correct/running_total:.4f}"
            )
        
        epoch_loss = running_loss / len(dsets["train"])
        epoch_acc = running_correct / running_total
        print(f"✅ Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in tqdm(loaders["val"], desc="Val", ncols=100):
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > 0.5).int().numpy().ravel().tolist()
                labs = labels.numpy().ravel().tolist()
                all_preds.extend(preds)
                all_labels.extend(labs)
        
        preds_t = torch.tensor(all_preds)
        labels_t = torch.tensor(all_labels)
        val_f1 = safe_f1(preds_t, labels_t)
        val_acc = (preds_t == labels_t).sum().item() / len(labels_t) if len(labels_t) > 0 else 0.0
        print(f"✅ Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_best = os.path.join("checkpoints", f"{args.model}_best.pth")
            ckpt_last = os.path.join("checkpoints", f"{args.model}_last.pth")
            
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "model_name": args.model
            }, ckpt_best)
            print(f"💾 Saved best model -> {ckpt_best}")
        
        # Save last checkpoint
        ckpt_last = os.path.join("checkpoints", f"{args.model}_last.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
            "model_name": args.model
        }, ckpt_last)
        
        scheduler.step()
        print(f"📉 Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"\n{'='*60}")
    print(f"🎉 Training finished! Best Val F1: {best_val_f1:.4f}")
    print(f"{'='*60}")
    
    # Final test evaluation
    best_ckpt = os.path.join("checkpoints", f"{args.model}_best.pth")
    if os.path.exists(best_ckpt):
        print("\n📊 Evaluating on test set...")
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in tqdm(loaders["test"], desc="Test", ncols=100):
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > 0.5).int().numpy().ravel().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy().ravel().tolist())
        
        p = torch.tensor(all_preds)
        l = torch.tensor(all_labels)
        test_f1 = safe_f1(p, l)
        test_acc = (p == l).sum().item() / len(l) if len(l) > 0 else 0.0
        print(f"\n🎯 Test Results:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   F1 Score: {test_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet with SRM preprocessing for steganalysis")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50", "resnet101"],
                        help="ResNet model variant")
    parser.add_argument("--data_dir", type=str, default="processed",
                        help="Path to processed dataset directory (with train/val/test subdirectories)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--freeze_until", type=str, default="layer3", 
                        choices=["none", "layer1", "layer2", "layer3"],
                        help="Freeze layers until this point")
    parser.add_argument("--step_size", type=int, default=6, help="LR scheduler step size")
    
    args = parser.parse_args()
    train(args)

