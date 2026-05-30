from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMAGE_SIZE = 224
BATCH_SIZE = 32


train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])


val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def create_category_dataloaders(data_dir="data/processed/category_classification"):
    data_dir = Path(data_dir)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes