# -----------------------------------------------------------
# train_research_fast.py (SRM FIXED PROPERLY, NO SKLEARN)
# -----------------------------------------------------------

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# -------- SIMPLE F1 (No sklearn needed) --------
def simple_f1(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    tp = tp.item(); fp = fp.item(); fn = fn.item()

    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


# ---------------- SRM FILTER -------------------
class SRMFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        srm = torch.tensor([
            [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],

            [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],

            [[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
        ]).float()

        srm = srm.unsqueeze(1)
        self.weight = nn.Parameter(srm, requires_grad=False)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, padding=2)


def adapt_first_conv(model, in_ch):
    old = model.conv1
    new = nn.Conv2d(in_ch, old.out_channels,
                    kernel_size=old.kernel_size,
                    stride=old.stride,
                    padding=old.padding,
                    bias=False)
    w = old.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
    new.weight = nn.Parameter(w)
    return new


# ---------------- DATA -----------------
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

    sets = {
        "train": datasets.ImageFolder(f"{data_dir}/train", train_tf),
        "val": datasets.ImageFolder(f"{data_dir}/val", eval_tf),
        "test": datasets.ImageFolder(f"{data_dir}/test", eval_tf),
    }

    loaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k=="train"), num_workers=0)
        for k, v in sets.items()
    }

    return loaders, sets


# ---------------- MODEL -----------------
class ResearchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMFrontend()

        # ⭐ NEW FIX: convert SRM(3ch) → RGB 3ch
        self.srm_to_rgb = nn.Conv2d(3, 3, kernel_size=1)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.ReLU(),
        )

        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for name, p in base.named_parameters():
            if "layer3" not in name:
                p.requires_grad = False

        base.conv1 = adapt_first_conv(base, 3)
        base.fc = nn.Linear(base.fc.in_features, 1)
        self.base = base

    def forward(self, x):

        # Convert RGB → gray → SRM
        x = x.mean(dim=1, keepdim=True)

        x = self.srm(x)

        # ⭐ FIX: 3→3 channel mapping (instead of repeat)
        x = self.srm_to_rgb(x)

        x = self.stem(x)
        return self.base(x)


# ---------------- TRAIN -----------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loaders, sets = get_dataloaders(args.data_dir, args.batch_size, args.img_size)

    model = ResearchModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_f1 = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n📘 EPOCH {epoch}/{args.epochs}")
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in tqdm(loaders["train"], desc="Training", ncols=100):
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"   Loss: {running_loss/len(sets['train']):.4f} | Train Acc: {correct/total:.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        preds_all, labels_all = [], []

        with torch.no_grad():
            for imgs, labels in loaders["val"]:
                imgs = imgs.to(device)
                out = model(imgs)
                probs = torch.sigmoid(out).cpu()

                preds_all.extend((probs > 0.5).int().numpy().ravel())
                labels_all.extend(labels.numpy().ravel())

        preds_t = torch.tensor(preds_all)
        labels_t = torch.tensor(labels_all)

        f1 = simple_f1(preds_t, labels_t)
        print(f"   F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_fast_model.pth")
            print("   💾 Saved best model")

    print("\n🎉 Training complete! Best F1:", best_f1)


# ---------------- MAIN -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--img_size", type=int, default=160)
    args = p.parse_args()

    train(args)
