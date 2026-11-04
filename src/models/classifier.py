from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.data.nih import LABELS


def build_model(num_classes: int =len(LABELS), pretrained: bool = True) -> nn.Module:

    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

