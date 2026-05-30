from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_models/defect_detector_efficientnet_b0.pth"
IMAGE_SIZE = 224

CLASSES = ["defective", "normal"]

transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transforms(image).unsqueeze(0).to(DEVICE)

    model = create_model(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return CLASSES[predicted_class.item()], confidence.item()

def main():
    image_path = input("Enter image path: ")

    prediction, confidence = predict_image(Path(image_path))
    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()