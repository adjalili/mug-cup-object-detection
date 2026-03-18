from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from args import Args


def get_dataloaders():

    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=Args.train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=Args.val_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Args.batch_size,
        shuffle=False
    )

    return train_loader, val_loader