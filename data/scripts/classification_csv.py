import os
import pandas as pd

images_dir = "data/images"
csv_path = "data/CSVs/classification_dataset.csv"

data = []

for img in sorted(os.listdir(images_dir)):
    if not img.endswith(".jpg"):
        continue

    img_number = int(img.split("_")[1].split(".")[0])

    if img_number <= 152:
        label = "mug"
    else:
        label = "no_mug"

    image_path = os.path.join(images_dir, img)
    data.append([image_path, label])

df = pd.DataFrame(data, columns=["image", "label"])
df.to_csv(csv_path, index=False)

print("Classification CSV created successfully ✅")