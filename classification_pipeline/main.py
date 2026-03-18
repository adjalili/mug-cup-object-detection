from dataset import get_dataloaders
from model import SimpleCNN
from trainer import train
from evaluate import evaluate
from args import Args


def main():

    train_loader, val_loader = get_dataloaders()

    model = SimpleCNN()

    train(
        model,
        train_loader,
        val_loader,
        epochs=Args.epochs,
        lr=Args.learning_rate
    )

    evaluate(model, val_loader)


if __name__ == "__main__":
    main()