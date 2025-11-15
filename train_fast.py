import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_dataloaders(data_dir, batch_size):

    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
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


def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out_dir", default="checkpoints_fast")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    dataloaders, datasets_obj = get_dataloaders(args.data_dir, args.batch_size)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 1),
        nn.Sigmoid()
    )
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n📘 EPOCH {epoch}/{args.epochs}")

        model.train()
        correct = 0
        total = 0

        total_batches = len(dataloaders["train"])
        batch_num = 1

        for imgs, labels in dataloaders["train"]:
            print(f"   ➤ Training batch {batch_num}/{total_batches}", end="\r")

            imgs = imgs.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            correct += torch.sum(preds == labels)
            total += labels.size(0)

            batch_num += 1

        train_acc = correct.double() / total
        print(f"\n   Train Acc: {train_acc:.4f}")

        # VALIDATION
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

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.out_dir}/best_model.pth")
            print("   💾 Saved best model")

    print("\n🎉 Training complete!")
    print("Best val accuracy:", best_acc)


if __name__ == "__main__":
    train()
