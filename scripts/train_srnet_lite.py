"""
train_srnet_lite.py
Lightweight SRNet-style steganalysis trainer with SRM-30 filters.
No sklearn / pandas. Windows-friendly (num_workers=0).
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
from tqdm import tqdm
import math
import os

# ---------------------- Utilities ----------------------
def seed_everything(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

# ---------------------- Metrics ----------------------
def safe_f1(preds, labels):
    """Compute F1 robustly on binary 0/1 tensors"""
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

# ---------------------- SRM-30 Filters ----------------------
def make_srm30_kernels(device='cpu'):
    """
    Return a Tensor of shape (30,1,5,5) with the SRM-30 (compact) filters.
    These are standard residual kernels used in steganalysis front-ends.
    (This implementation uses a compact SRM-30 subset commonly used in lightweight variants.)
    """
    # For brevity and reproducibility we define a 30-filter bank using
    # standard residual patterns (combinations / rotations / scalings).
    # These are deterministic, not learned.
    import numpy as np

    base = []
    # Some commonly used small residual kernels (5x5 patterns)
    # We construct a few base patterns and then generate rotated/negated variations.
    k1 = np.array([[0,0,0,0,0],
                   [0,0,-1,0,0],
                   [0,-1,4,-1,0],
                   [0,0,-1,0,0],
                   [0,0,0,0,0]], dtype=np.float32)

    k2 = np.array([[0,0,0,0,0],
                   [0,-1,2,-1,0],
                   [0,2,-4,2,0],
                   [0,-1,2,-1,0],
                   [0,0,0,0,0]], dtype=np.float32)

    k3 = np.array([[0,0,0,0,0],
                   [0,1,-2,1,0],
                   [0,-2,4,-2,0],
                   [0,1,-2,1,0],
                   [0,0,0,0,0]], dtype=np.float32)

    k4 = np.array([[-1,2,-2,2,-1],
                   [2,-6,8,-6,2],
                   [-2,8,-12,8,-2],
                   [2,-6,8,-6,2],
                   [-1,2,-2,2,-1]], dtype=np.float32)

    base_kernels = [k1, k2, k3, k4]

    # Generate variations: rotations and sign flips
    kernels = []
    for bk in base_kernels:
        for flip in [1, -1]:
            for rot in [0, 1, 2]:
                k = np.rot90(bk * flip, k=rot)
                kernels.append(k)
                if len(kernels) >= 30:
                    break
            if len(kernels) >= 30:
                break
        if len(kernels) >= 30:
            break

    # If we have less than 30 due to construction, pad by repeating with small noise
    while len(kernels) < 30:
        kernels.append(kernels[len(kernels) % len(base_kernels)] * (1.0 + 1e-6 * len(kernels)))

    arr = np.stack(kernels[:30], axis=0)  # (30,5,5)
    arr = arr[:, None, :, :]  # (30,1,5,5)
    return torch.from_numpy(arr).float().to(device)

# ---------------------- Model (SRNet-Lite) ----------------------
class SRMFrontend(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        w = make_srm30_kernels(device=device)  # (30,1,5,5)
        self.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x_gray):
        # x_gray: [B,1,H,W]
        return nn.functional.conv2d(x_gray, self.weight, padding=2)  # -> [B,30,H,W]

class SRNetLite(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        # SRM frontend (30 filters)
        self.srm = SRMFrontend(device=device)
        # Map SRM 30 channels to 32 channels via 1x1 conv (learnable)
        self.map1 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Small convolutional backbone (lightweight residual blocks)
        def conv_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.layer1 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 64),
        )

        self.res1 = nn.Sequential(
            conv_block(64, 64),
            conv_block(64, 64),
        )

        self.layer2 = nn.Sequential(
            conv_block(64, 128, stride=2),  # downsample
            conv_block(128, 128),
        )

        self.res2 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128),
        )

        self.layer3 = nn.Sequential(
            conv_block(128, 256, stride=2),
            conv_block(256, 256),
        )

        # global pooling and classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [B,3,H,W] RGB in [0,1]
        # convert to grayscale
        xg = x.mean(dim=1, keepdim=True)  # [B,1,H,W]

        s = self.srm(xg)                   # [B,30,H,W]
        s = self.map1(s)                   # [B,32,H,W]

        x = self.layer1(s)
        x = x + self.res1(x)               # residual add (same shape)

        x = self.layer2(x)
        x = x + self.res2(x)

        x = self.layer3(x)

        x = self.pool(x)
        out = self.fc(x)                   # logits
        return out

# ---------------------- Data loaders ----------------------
def get_dataloaders(data_dir, batch_size, img_size):
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

    # Windows-friendly: num_workers=0, pin_memory=False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return {"train": train_loader, "val": val_loader, "test": test_loader}, {"train": train_ds, "val": val_ds, "test": test_ds}

# ---------------------- Training loop ----------------------
def train(args):
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    loaders, dsets = get_dataloaders(args.data_dir, args.batch_size, args.img_size)

    model = SRNetLite(device=device).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    best_val_f1 = 0.0
    safe_mkdir("checkpoints")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(loaders["train"], total=len(loaders["train"]), desc="Train", ncols=100)
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

            pbar.set_postfix(loss=f"{running_loss/running_total:.4f}", acc=f"{running_correct/running_total:.4f}")

        epoch_loss = running_loss / len(dsets["train"])
        epoch_acc = running_correct / running_total
        print(f" Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in loaders["val"]:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > 0.5).int().numpy().ravel().tolist()
                labs = labels.numpy().ravel().tolist()
                all_preds.extend(preds)
                all_labels.extend(labs)

        import torch as _torch
        preds_t = _torch.tensor(all_preds)
        labels_t = _torch.tensor(all_labels)
        val_f1 = safe_f1(preds_t, labels_t)
        val_acc = (preds_t == labels_t).sum().item() / len(labels_t) if len(labels_t) > 0 else 0.0
        print(f" Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f}")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt = os.path.join("checkpoints", f"srnet_lite_best.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_f1": val_f1}, ckpt)
            print(" Saved best ->", ckpt)

        scheduler.step()

    print("\nTraining finished. Best Val F1:", best_val_f1)

    # Final test evaluation load best
    best_ckpt = os.path.join("checkpoints", "srnet_lite_best.pth")
    if os.path.exists(best_ckpt):
        print("Loading best model for test evaluation...")
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in loaders["test"]:
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
        print(f"\nTest Acc: {test_acc:.4f} Test F1: {test_f1:.4f}")


# ---------------------- CLI ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="processed dataset dir (train/ val/ test folders)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=160)
    args = parser.parse_args()
    train(args)
