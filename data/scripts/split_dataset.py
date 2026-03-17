import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("data/CSVs/dataset.csv")
print("Full dataset shape:", dataset.shape)

train_data, val_data = train_test_split(
    dataset,
    test_size=0.3,
    random_state=42
)

print("Train shape:", train_data.shape)
print("Validation shape:", val_data.shape)

train_data.to_csv("data/CSVs/train.csv", index=False)
val_data.to_csv("data/CSVs/val.csv", index=False)

print("Splitting done successfully!")
