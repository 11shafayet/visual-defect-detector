from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.model import create_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("saved_models/defect_detector_efficientnet_b0.pth")

CLASSES = ["defective", "normal"]
IMAGE_SIZE = 224


@st.cache_resource
def load_model():
    model = create_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def predict(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return CLASSES[predicted_class.item()], confidence.item(), predicted_class.item()

def generate_gradcam(image, predicted_class_index):
    model = load_model()

    rgb_image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_array = np.array(rgb_image).astype(np.float32) / 255.0

    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(predicted_class_index)]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(
        rgb_array,
        grayscale_cam,
        use_rgb=True
    )

    return visualization

st.title("Visual Defect Detector")

st.caption("Multi-category binary defect detector")

st.write(
    "Supported categories: bottle, capsule, hazelnut, metal_nut, toothbrush, zipper"
)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    prediction, confidence, predicted_class_index = predict(image)

    st.subheader("Prediction Result")

    if prediction == "defective":
        st.error(f"Prediction: {prediction.upper()}")
    else:
        st.success(f"Prediction: {prediction.upper()}")

    st.write(f"**Confidence:** {confidence:.2%}")
    st.progress(confidence)

    st.subheader("Model Attention Heatmap")
    heatmap = generate_gradcam(image, predicted_class_index)

    st.image(
        heatmap,
        caption="Grad-CAM Heatmap",
        use_container_width=True
    )