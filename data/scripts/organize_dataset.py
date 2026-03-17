import os
import shutil
import pandas as pd

# Load train and val CSV files
train_df = pd.read_csv("data/CSVs/train.csv")
val_df = pd.read_csv("data/CSVs/val.csv")

# Create folders if they don't exist
os.makedirs("data/images/train", exist_ok=True)
os.makedirs("data/images/val", exist_ok=True)
os.makedirs("data/labels/train", exist_ok=True)
os.makedirs("data/labels/val", exist_ok=True)

# Move training files
for _, row in train_df.iterrows():
    shutil.move(row["images"], "data/images/train/")
    shutil.move(row["labels"], "data/labels/train/")

# Move validation files
for _, row in val_df.iterrows():
    shutil.move(row["images"], "data/images/val/")
    shutil.move(row["labels"], "data/labels/val/")

print("Dataset organized successfully!")