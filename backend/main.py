import os

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from model import create_model

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import sqlite3
from datetime import datetime


IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFECT_MODEL_PATH = Path("saved_models/defect_detector_efficientnet_b0.pth")
CATEGORY_MODEL_PATH = Path("saved_models/category_classifier_efficientnet_b0.pth")

DEFECT_CLASSES = ["defective", "normal"]
CATEGORY_CLASSES = [
    "bottle",
    "capsule",
    "hazelnut",
    "metal_nut",
    "toothbrush",
    "zipper",
]

IMAGE_SIZE = 224

DB_PATH = Path(os.getenv("DB_PATH", "predictions.db"))

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                category TEXT,
                category_confidence REAL,
                prediction TEXT,
                confidence REAL,
                created_at TEXT
            )
        """)

        conn.commit()

init_db()

app = FastAPI(title="Visual Defect Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_model(model_path, num_classes):
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


defect_model = load_model(DEFECT_MODEL_PATH, len(DEFECT_CLASSES))
category_model = load_model(CATEGORY_MODEL_PATH, len(CATEGORY_CLASSES))


@app.get("/")
def home():
    return {"message": "Visual Defect Detector API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        defect_outputs = defect_model(image_tensor)
        defect_probs = torch.softmax(defect_outputs, dim=1)
        defect_confidence, defect_class = torch.max(defect_probs, 1)

        category_outputs = category_model(image_tensor)
        category_probs = torch.softmax(category_outputs, dim=1)
        category_confidence, category_class = torch.max(category_probs, 1)

    defect_class_index = defect_class.item()
    category_class_index = category_class.item()

    rgb_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_array = np.array(rgb_image).astype(np.float32) / 255.0

    target_layers = [defect_model.features[-1]]
    targets = [ClassifierOutputTarget(defect_class_index)]

    cam = GradCAM(model=defect_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    heatmap = show_cam_on_image(
        rgb_array,
        grayscale_cam,
        use_rgb=True,
    )

    heatmap_image = Image.fromarray(heatmap)
    buffer = BytesIO()
    heatmap_image.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (
                filename,
                category,
                category_confidence,
                prediction,
                confidence,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file.filename,
                CATEGORY_CLASSES[category_class_index],
                round(category_confidence.item(), 4),
                DEFECT_CLASSES[defect_class_index],
                round(defect_confidence.item(), 4),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

        cursor.execute("""
            DELETE FROM predictions
            WHERE id NOT IN (
                SELECT id FROM predictions
                ORDER BY id DESC
                LIMIT 20
            )
        """)

        conn.commit()

    return {
        "category": CATEGORY_CLASSES[category_class_index],
        "category_confidence": round(category_confidence.item(), 4),
        "prediction": DEFECT_CLASSES[defect_class_index],
        "confidence": round(defect_confidence.item(), 4),
        "heatmap": f"data:image/png;base64,{heatmap_base64}",
    }

@app.get("/history")
def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id,
                filename,
                category,
                category_confidence,
                prediction,
                confidence,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 20
        """)

        rows = cursor.fetchall()

    return [
        {
            "id": row[0],
            "filename": row[1],
            "category": row[2],
            "category_confidence": row[3],
            "prediction": row[4],
            "confidence": row[5],
            "created_at": row[6],
        }
        for row in rows
    ]
