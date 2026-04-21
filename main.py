from args import get_args
import pandas as pd
from dataset import ObjDetectionDataset
import torch
from torch.utils.data import DataLoader
import os
from model import build_model
from trainer import train_model
from augmentations import build_train_transforms, build_val_transforms

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. read the dataframes
    train_df = pd.read_csv(os.path.join(args.csv_dir, "train_df.csv"))
    val_df = pd.read_csv(os.path.join(args.csv_dir, "val_df.csv"))



    # 2. prepare datasets
    train_dataset = ObjDetectionDataset(train_df, transform = build_train_transforms(args.img_size))
    val_dataset = ObjDetectionDataset(val_df, transform = build_val_transforms(args.img_size))

    # 3. create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    # 4. initialize the model
    model = build_model(args.backbone, num_classes=args.num_classes + 1)

    # 5. train the model (FIXED)
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)

    # 6. plot graph (ADD THIS)
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()

    plt.savefig("learning_curve.png")
    plt.show()


if __name__ == "__main__":
    main()