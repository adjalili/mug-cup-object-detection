from dataset import get_dataloaders
from model import SimpleCNN
from trainer import train_model
import torch


def main():

    print("🚀 Starting Project...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders()

    model = SimpleCNN()

    train_model(model, train_loader, val_loader, device, epochs=15)


if __name__ == "__main__":
    main()