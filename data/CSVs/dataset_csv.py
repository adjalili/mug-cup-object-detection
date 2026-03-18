import os
import csv

# Paths to dataset folders
images_dir = "data/images"          # Folder containing image files
labels_dir = "data/labels"          # Folder containing label files
csv_path = "data/CSVs/dataset.csv"  # Output CSV file path

# Get all filenames in the images directory
image_files = sorted(os.listdir(images_dir))

# Open (or create) the CSV file in write mode
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    # Write CSV header
    writer.writerow(["images", "labels"])

    # Loop through image files
    for img in image_files:

        # Skip non-image files (e.g., .gitkeep)
        if not img.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Build image path
        image_path = os.path.join(images_dir, img)

        # Build corresponding label filename
        label_name = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        # Write image–label pair to CSV
        writer.writerow([image_path, label_path])

# Confirmation message
print("dataset.csv created successfully")
