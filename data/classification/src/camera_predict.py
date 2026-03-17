import torch
import cv2
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
from args import Args

# classes
classes = ["mug", "no_mug"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load trained model
model = SimpleCNN()
model.load_state_dict(torch.load(Args.model_path, map_location=device))
model.to(device)
model.eval()

# image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# open webcam
cap = cv2.VideoCapture(0)

while True:

    # 📌 THIS IS WHERE YOUR CODE GOES
    ret, frame = cap.read()

    if not ret:
        break

    # crop center area
    h, w, _ = frame.shape
    cx = w // 2
    cy = h // 2

    crop = frame[cy-100:cy+100, cx-100:cx+100]

    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output,1)

    label = classes[pred.item()]

    # show prediction
    cv2.putText(frame, f"Prediction: {label}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # draw box in center
    cv2.rectangle(frame,(cx-100,cy-100),(cx+100,cy+100),(255,0,0),2)

    cv2.imshow("Mug Classifier", frame)

    # press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()