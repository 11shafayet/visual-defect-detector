import torch

from category_dataset import create_category_dataloaders
from model import create_model

from sklearn.metrics import classification_report, confusion_matrix


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "saved_models/category_classifier_efficientnet_b0.pth"


def main():
    _, _, test_loader, classes = create_category_dataloaders()

    model = create_model(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    actual_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            actual_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    print("Classification Report")
    print(classification_report(actual_labels, predicted_labels, target_names=classes))

    print("Confusion Matrix")
    print(confusion_matrix(actual_labels, predicted_labels))


if __name__ == "__main__":
    main()