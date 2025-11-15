import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------------------------------
# High-Pass Filter Layer (NO LAMBDA)
# -------------------------------
import torch.nn.functional as F

class HPFTransform(nn.Module):
    def __init__(self):
        super().__init__()
        hpf_kernel = torch.tensor([
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1]
        ], dtype=torch.float32)

        hpf_kernel = hpf_kernel.unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(hpf_kernel, requires_grad=False)

    def forward(self, img):
        # img: [C, H, W]
        img = img.unsqueeze(0)      # to [1,C,H,W]
        img = F.conv2d(img, self.weight.repeat(3, 1, 1, 1), padding=1, groups=3)
        return img.squeeze(0)

HPF = HPFTransform()

# -------------------------------
# DATALOADER FIXED FOR WINDOWS
# -------------------------------

def get_dataloaders(data_dir, batch_size, size=224):
    
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        HPF
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        HPF
    ])

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=eval_tf),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=eval_tf)
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size,
                      shuffle=(x == "train"), num_workers=0)
        for x in ["train", "val", "test"]
    }

    return dataloaders, image_datasets


# -------------------------------
# TRAINING LOOP
# -------------------------------

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out_dir", default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    dataloaders, datasets_obj = get_dataloaders(args.data_dir, args.batch_size)

    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all layers except layer4 + fc
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace FC
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    best_acc = 0

    print()

    # -----------------------------------
    # TRAINING
    # -----------------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"📘 EPOCH {epoch}/{args.epochs}")

        model.train()
        running_loss = 0
        running_corrects = 0

        for imgs, labels in dataloaders["train"]:
            imgs = imgs.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(datasets_obj["train"])
        epoch_acc = running_corrects.double() / len(datasets_obj["train"])

        print(f"   Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        # -----------------------------------
        # VALIDATION
        # -----------------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in dataloaders["val"]:
                imgs = imgs.to(device)
                labels = labels.float().to(device).unsqueeze(1)

                outputs = model(imgs)
                preds = (outputs > 0.5).float()

                correct += torch.sum(preds == labels)
                total += labels.size(0)

        val_acc = correct.double() / total
        print(f"   Validation Acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.out_dir}/best_model.pth")
            print("   💾 Saved new best model")

        print()

    print("🎉 TRAINING COMPLETE!")
    print("Best validation accuracy:", best_acc)


if __name__ == "__main__":
    train()
