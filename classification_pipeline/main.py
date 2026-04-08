from dataset import get_dataloaders
from model import SimpleCNN
from trainer import train_model
from utils import show_batch   
import torch


def main():

    print("🚀 Starting Project...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders()

    # ✅ SHOW WHAT GOES INTO MODEL
    images, labels = next(iter(train_loader))
    classes = train_loader.dataset.classes

    print("📸 Showing sample inputs...")
    show_batch(images, labels, classes)

    # model
    model = SimpleCNN()

    # train
    train_model(model, train_loader, val_loader, device, epochs=15)


if __name__ == "__main__":
    main()