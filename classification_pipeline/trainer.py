import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, val_loader, epochs=15, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -----------------------
    # Class Weights (handle imbalance)
    # 0 = mug
    # 1 = no_mug
    # -----------------------
    class_weights = torch.tensor([1.0, 3.0]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0.0

    print("\nStarting Training...\n")

    for epoch in range(epochs):

        # =========================
        # TRAINING
        # =========================
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Val Accuracy: {accuracy:.2f}%"
        )

        # =========================
        # SAVE BEST MODEL
        # =========================
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                model.state_dict(),
                "data/classification/best_model.pth"
            )
            print("🔥 Best model saved!")

    print("\nTraining Finished.")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")