import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
'''...'''

dataset = pd.read_csv("data/CSVs/dataset.csv")

train_data, val_data = train_test_split(dataset, test_size=0.2)
train_data.to_csv("data/CSVs/train.csv", index=False)
val_data.to_csv("data/CSVs/val.csv", index=False)