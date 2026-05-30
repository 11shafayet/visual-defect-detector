import torch.nn as nn
from torchvision import models


def create_model(num_classes=2):
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model