import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def evaluate(model, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(cm)

    # ===== ACCURACY =====
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    # ===== PLOT CONFUSION MATRIX =====
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Show numbers inside cells
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()