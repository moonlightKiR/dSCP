import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights


class CriminalityVGG16(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(CriminalityVGG16, self).__init__()

        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.vgg = models.vgg16(weights=weights)

        for param in self.vgg.features.parameters():
            param.requires_grad = False

        for param in self.vgg.features[24:].parameters():
            param.requires_grad = True

        num_ftrs = self.vgg.classifier[0].in_features

        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paso hacia adelante de la red."""
        return self.vgg(x)
