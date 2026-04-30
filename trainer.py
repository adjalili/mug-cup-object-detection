import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, device, epochs=15, lr=0.0003):

    model.to(device)

    # ===== LOSS =====
    class_weights = torch.tensor([1.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ===== OPTIMIZER =====
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ✅ LEARNING RATE SCHEDULER (VERY IMPORTANT)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_accuracy = 0.0

    train_losses = []
    val_losses = []
    val_accuracies = []

    print("\n🚀 Starting Training...\n")

    for epoch in range(epochs):

        # ===== TRAIN =====
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

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f} "
            f"Val Acc: {accuracy:.2f}%"
        )

        # ===== SAVE BEST MODEL =====
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Best model saved!")

        # ✅ STEP SCHEDULER
        scheduler.step()

    print("\nTraining Finished.")
    print(f"Best Accuracy: {best_accuracy:.2f}%")

    # ===== PLOT =====
    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Ultra Model)")
    plt.legend()

    plt.savefig("learning_curve.png")
    plt.show()

    # ===== OPTIONAL: ACCURACY PLOT =====
    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.show()