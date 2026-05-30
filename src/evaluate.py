import torch

from dataset import create_dataloaders
from model import create_model
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    accuracy_score, 
    recall_score, 
    precision_score, 
    confusion_matrix
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "saved_models/defect_detector_efficientnet_b0.pth"

def main():
    _,_, test_loader, classes = create_dataloaders()

    print(f"Classes: {classes}")
    print(f"Using device: {DEVICE}")

    model = create_model(num_classes=len(classes)).to(DEVICE)
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
    
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average='binary', pos_label=0)
    recall = recall_score(actual_labels, predicted_labels, average='binary', pos_label=0)
    f1 = f1_score(actual_labels, predicted_labels, average='binary', pos_label=0)

    print("Evaluation Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(actual_labels, predicted_labels, target_names=classes))

    print("Confusion Matrix:")
    print(confusion_matrix(actual_labels, predicted_labels))


if __name__ == "__main__":
    main()