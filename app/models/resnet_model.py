"""
Utiliza ResNet50, aprovechamos que la red ya sabe reconocer formas, colores y 
texturas y simplemente "reentrenamos" la parte final para que aprenda a 
distinguir entre "Perfil Estándar" y "Perfil de Riesgo"
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class CriminalityResNet(nn.Module):
    """
    Clase que define la arquitectura ResNet50 adaptada para clasificación binaria.
    Utiliza pesos preentrenados y modifica la capa final para la tarea específica.
    """

    def __init__(self, pretrained: bool = True):
        """
        Inicializa el modelo ResNet50.
        Args:
            pretrained (bool): Si es True, carga los pesos preentrenados de ImageNet. 
                               Por defecto es True para aprovechar el Transfer Learning.
        """
        super(CriminalityResNet, self).__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)

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
        """
        Realiza el paso hacia adelante (forward pass) de la red.

        Args:
            x (torch.Tensor): Tensor de entrada con la imagen (Batch, Canales, H, W).

        Returns:
            torch.Tensor: Probabilidad de que la imagen pertenezca a la clase de riesgo.
        """
        return self.resnet(x)