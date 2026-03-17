import os
import shutil
import pandas as pd

train_df = pd.read_csv("data/CSVs/classification_train.csv")
val_df = pd.read_csv("data/CSVs/classification_val.csv")

base_path = "data/classification/images"

for split_name, df in [("train", train_df), ("val", val_df)]:
    for _, row in df.iterrows():
        label_folder = os.path.join(base_path, split_name, row["label"])
        os.makedirs(label_folder, exist_ok=True)

        shutil.copy(row["image"], label_folder)

print("Images organized for classification ✅")