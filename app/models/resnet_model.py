import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class CriminalityResNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(CriminalityResNet, self).__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)

        # Congelamos las capas base (Transfer Learning)
        for param in self.resnet.parameters():
            param.requires_grad = False

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)