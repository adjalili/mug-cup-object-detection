# Mug Classification Pipeline

## Project Description

This project implements a **modular machine learning pipeline** for classifying images into two categories:

* **Mug**
* **No Mug**

The pipeline follows a structured approach including data preparation, dataset organization, training, validation, and evaluation.

---

## Project Structure

```
mug-cup-object-detection
│
├── classification_pipeline
│   ├── args.py          # Configuration (paths, epochs, batch size)
│   ├── dataset.py       # Dataset loading and preprocessing
│   ├── model.py         # CNN model definition
│   ├── trainer.py       # Training and validation loop
│   ├── utils.py         # Helper functions
│   ├── main.py          # Main pipeline execution
│   └── evaluate.py      # Model evaluation (accuracy, confusion matrix)
│
├── data
│   ├── images/          # Original images (for detection dataset)
│   ├── labels/          # Annotation files (YOLO format)
│   └── CSVs/            # CSV files (dataset structure)
│
├── README.md
└── requirements.txt
```

---

## Dataset

The dataset is organized in a structured format:

* **images/** → contains all image files
* **labels/** → contains corresponding annotation files
* **CSVs/** → contains dataset CSV files

### CSV Format

Each row in `dataset.csv` contains:

```
images,labels
data/images/img_0.jpg,data/labels/img_0.txt
```

---

## Pipeline Overview

1. **Data Collection**
   Images collected in indoor environments.

2. **Annotation**
   Bounding boxes created in YOLO format.

3. **Dataset Structuring**
   Organized into images, labels, and CSV files.

4. **Data Splitting**
   Training and validation sets created.

5. **Training**
   CNN model trained using PyTorch.

6. **Validation**
   Model evaluated on validation dataset.

7. **Evaluation**
   Performance measured using accuracy and confusion matrix.

---

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

---

## Group Members

* Ahmadshah Djalili
* Emir Gorduk
* Nadeera Hettithanthreege Don
