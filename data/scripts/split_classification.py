import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("data/CSVs/classification_dataset.csv")

train_data, val_data = train_test_split(
    dataset,
    test_size=0.3,
    random_state=42,
    stratify=dataset["label"]   # very important
)

train_data.to_csv("data/CSVs/classification_train.csv", index=False)
val_data.to_csv("data/CSVs/classification_val.csv", index=False)

print("Classification splitting done ✅")