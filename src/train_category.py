import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from category_dataset import create_category_dataloaders
from model import create_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 8
LEARNING_RATE = 0.001
MODEL_PATH = "saved_models/category_classifier_efficientnet_b0.pth"


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(outputs, 1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(dataloader), correct / total


def main():
    train_loader, val_loader, _, classes = create_category_dataloaders()

    print(f"Classes: {classes}")
    print(f"Using device: {DEVICE}")

    model = create_model(num_classes=len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print("Best category model saved.")

    print("Category training complete.")


if __name__ == "__main__":
    main()