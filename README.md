# Visual Defect Detector

A full-stack AI application that detects product defects, classifies product categories, generates Grad-CAM explanations, and stores prediction history.

Built using PyTorch, EfficientNet-B0, FastAPI, React, and SQLite.

---

## Project Overview

This project started as a simple binary defect detector using a single product category from the MVTec AD dataset. It gradually evolved into a complete computer vision application capable of:

* Detecting defective products
* Identifying product categories
* Explaining model decisions with Grad-CAM
* Serving predictions through an API
* Providing a web-based user interface
* Storing recent prediction history

Supported product categories:

* Bottle
* Capsule
* Hazelnut
* Metal Nut
* Toothbrush
* Zipper

---

## Features

### Defect Detection

Classifies products as:

* Defective
* Normal

### Product Category Classification

Identifies the product type before defect analysis.

### Explainable AI

Generates Grad-CAM heatmaps to visualize which regions influenced the model's prediction.

### Prediction History

Stores recent predictions with:

* Filename
* Product category
* Prediction result
* Confidence scores
* Timestamp

### Full-Stack Architecture

* React frontend
* FastAPI backend
* SQLite database
* PyTorch inference engine

---

## Development Journey

### Phase 1 — Single Category Detection

Started with the Bottle category from the MVTec AD dataset.

Goal:

```text
Defective vs Normal
```

Initial ResNet18 model struggled due to severe class imbalance.

After introducing weighted loss functions, performance improved significantly.

**Result**

* Accuracy: 90.32%
* Recall: 100%
* F1 Score: 88.00%

---

### Phase 2 — Multi-Category Detection

Expanded the dataset to six product categories:

* Bottle
* Capsule
* Hazelnut
* Metal Nut
* Toothbrush
* Zipper

ResNet18 performance dropped considerably, highlighting the increased complexity of multi-category defect detection.

---

### Phase 3 — EfficientNet Upgrade

Replaced ResNet18 with EfficientNet-B0.

This dramatically improved performance and became the final defect detection model.

**Final Defect Detection Results**

* Accuracy: 93.52%
* Precision: 92.71%
* Recall: 90.82%
* F1 Score: 91.75%

---

### Phase 4 — Category Classification

Built a separate EfficientNet-B0 classifier to identify product categories.

**Category Classification Result**

* Accuracy: 100%

---

### Phase 5 — Full-Stack Application

The final version combines:

* Defect detection
* Category classification
* Grad-CAM visualization
* Prediction history
* Web interface
* REST API

into a single deployable application.

---

## Model Performance

### Defect Detection Model

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 93.52% |
| Precision | 92.71% |
| Recall    | 90.82% |
| F1 Score  | 91.75% |

### Category Classification Model

| Metric   | Score |
| -------- | ----- |
| Accuracy | 100%  |

---

## Architecture

```text
React Frontend
        │
        ▼
FastAPI Backend
        │
        ├── Category Classifier
        │
        ├── Defect Detector
        │
        ├── Grad-CAM
        │
        └── SQLite History
```

---

## Tech Stack

### Machine Learning

* PyTorch
* Torchvision
* EfficientNet-B0
* Grad-CAM

### Backend

* FastAPI
* Uvicorn
* SQLite

### Frontend

* React
* Vite
* Axios

---

## Project Structure

```text
visual-defect-detector/
│
├── frontend/
│
├── backend/
│
├── src/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## Future Improvements

* Webcam-based real-time inspection
* Multi-class defect type classification
* Cloud database integration
* Batch image processing
* Production deployment on AWS
* Industrial analytics dashboard

---

## Key Learnings

This project covered the complete machine learning lifecycle:

* Dataset preparation
* Data preprocessing
* Class imbalance handling
* Transfer learning
* CNN architectures
* Model evaluation
* Explainable AI
* REST API development
* Frontend integration
* Full-stack AI deployment

The goal was not only to train a model but to understand how machine learning systems are built, integrated, and deployed as real-world applications.
