from dataset import get_dataloaders
from model import SimpleCNN
from trainer import train_model
from evaluate import evaluate
from utils import show_batch
from args import get_args
import torch


def main():

    print("🚀 Starting Project...")

    # ===== LOAD ARGS =====
    args = get_args()

    # ===== DEVICE =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== LOAD DATA =====
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size
    )

    # ===== SHOW SAMPLE INPUT =====
    images, labels = next(iter(train_loader))
    classes = train_loader.dataset.classes

    print("📸 Showing sample inputs...")
    show_batch(images, labels, classes)

    # ===== MODEL =====
    model = SimpleCNN()

    # ===== TRAIN =====
    train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr
    )

    # ===== LOAD BEST MODEL =====
    print("\n📦 Loading best model...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    # ===== EVALUATE =====
    print("\n📊 Evaluating model...")
    evaluate(model, val_loader)


if __name__ == "__main__":
    main()