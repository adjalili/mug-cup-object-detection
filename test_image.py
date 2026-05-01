import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms
import os

from model import build_model

MODEL_PATH = "sessions/best_model.pth"
FOLDER_PATH = "data/test_images"
OUTPUT_DIR = "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = build_model("fasterrcnn_mobilenet_v3_large_fpn", num_classes=2)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

print("Model loaded")

for img_name in os.listdir(FOLDER_PATH):

    IMAGE_PATH = os.path.join(FOLDER_PATH, img_name)
    print("Processing:", img_name)

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image)

    input_tensor = transform(image).to(DEVICE)

    with torch.no_grad():
        outputs = model([input_tensor])[0]

    boxes = outputs["boxes"]
    scores = outputs["scores"]

    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.3

    keep = scores > SCORE_THRESHOLD
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) > 0:
        keep_idx = nms(boxes, scores, IOU_THRESHOLD)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        best_idx = scores.argmax()
        box = boxes[best_idx]
        score = scores[best_idx]

        x1, y1, x2, y2 = box.int().cpu().numpy()

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            image_np,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    output_path = os.path.join(OUTPUT_DIR, f"output_{img_name}")
    cv2.imwrite(output_path, image_np)

    print("Saved:", output_path)

print("Done. Check results/ folder.")