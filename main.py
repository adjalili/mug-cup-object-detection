import pandas as pd
import os
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
import torch

from model import build_model
from args import get_args

print("--- Program is working! ---")

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
from training import train_model
def main():
    args = get_args()
    #1. Read your dataframes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(os.path.join(args.csv_dir,"train_df.csv"))
    val_df = pd.read_csv(os.path.join(args.csv_dir,"val_df.csv"))

    # 2. Prepare datasets
    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    # 3. Create Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print("Data loading started. wait...")

    # 4 INITIALIZING MODEL
    print("Preparing the Model...")
    model = build_model(args.backbone)
    model.to(device)

    # 5. START TRAINING
    print("Start The Training...")
    
    train_model(model, train_loader, val_loader, device)
if __name__ == "__main__" :
    main()

