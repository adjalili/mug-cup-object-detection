import torch
import cv2
import torchvision.transforms as transforms
from model import SimpleCNN

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD MODEL =====
model = SimpleCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
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

# ===== CAMERA =====
cap = cv2.VideoCapture("http://192.168.1.104:8080/video")

print("📷 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to get frame")
        break

    # Convert BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transform
    img = transform(img).unsqueeze(0).to(device)

    # ===== PREDICTION =====
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

        label = classes[pred.item()]
        conf = confidence.item() * 100

    # ===== TEXT ONLY =====
    text = f"{label} ({conf:.1f}%)"

    color = (0, 255, 0) if label == "Mug" else (0, 0, 255)

    cv2.putText(frame, text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, color, 3)

    # ===== SHOW =====
    cv2.imshow("Mug Classification (Phone Camera)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()