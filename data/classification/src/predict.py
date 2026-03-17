import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
from args import Args

# classes
classes = ["mug", "no_mug"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = SimpleCNN()
model.load_state_dict(torch.load(Args.model_path, map_location=device))
model.to(device)
model.eval()

# image transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output,1)

    label = classes[pred.item()]
    print(f"Prediction: {label}")


if __name__ == "__main__":

    image_path = "sample_image/mug_sample.jpg"
    predict(image_path)