from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size):

    # ===== TRANSFORMS =====
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # ===== DATASETS =====
    train_dataset = datasets.ImageFolder(
        root="../data/classification/images/train",   # ✅ FIXED PATH
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root="../data/classification/images/val",     # ✅ FIXED PATH
        transform=val_transform
    )

    # ===== DATALOADERS =====
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader