import torch
from sklearn.metrics import confusion_matrix


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
    correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
    accuracy = correct / len(all_labels) * 100

    print(f"\nAccuracy: {accuracy:.2f}%")