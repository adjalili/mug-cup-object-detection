from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders():

    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root="data/classification/images/train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root="data/classification/images/val",
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader