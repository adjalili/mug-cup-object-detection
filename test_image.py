import torch
import cv2
import os
import torchvision.transforms as transforms
from model import SimpleCNN   # ✅ correct import

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD MODEL =====
model = SimpleCNN()
model.load_state_dict(torch.load("../best_model.pth", map_location=device))  # ✅ correct path
model.to(device)
model.eval()

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===== CLASSES =====
classes = ["No Mug", "Mug"]

# ===== TEST FOLDER =====
folder = "../test_images"   # ✅ correct path

print("📸 Testing images...")

# ===== WINDOW SETTINGS (BIG DISPLAY) =====
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 1200, 800)

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    image = cv2.imread(path)
    if image is None:
        continue

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform
    img = transform(img_rgb).unsqueeze(0).to(device)

    # ===== PREDICTION =====
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

        label = classes[pred.item()]
        conf = confidence.item() * 100

    # ===== TEXT =====
    text = f"{label} ({conf:.1f}%)"
    color = (0, 255, 0) if label == "Mug" else (0, 0, 255)

    cv2.putText(image, text, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, color, 3)

    # ===== RESIZE TO BIG SCREEN =====
    resized = cv2.resize(image, (1200, 800))

    # ===== SHOW =====
    cv2.imshow("Result", resized)

    # wait 1 second OR press key
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()