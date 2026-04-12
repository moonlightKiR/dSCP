"""
Contiene la lógica del bucle de entrenamiento y el de 
validación. 
Permitoiendo ser usado para sus diferentes modelos (VGG, etc.), 
garantizando que todos midan el Accuracy y el Loss con las mismas 
caracteristicas.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """
    Realiza un ciclo completo de entrenamiento sobre el dataset (una época).

    Args:
        model (nn.Module): El modelo a entrenar (ej. CriminalityResNet).
        loader (DataLoader): El cargador de datos de entrenamiento.
        optimizer (Optimizer): El algoritmo de optimización (ej. Adam).
        criterion (nn.Module): La función de pérdida (ej. BCELoss).
        device (torch.device): El dispositivo de ejecución (cuda o cpu).

    Returns:
        Tuple[float, float]: (Pérdida media de la época, Precisión media de 
        la época).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=" > Entrenando"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """
    Evalúa el rendimiento del modelo en un conjunto de datos.

    Args:
        model (nn.Module): El modelo a evaluar.
        loader (DataLoader): El cargador de datos de evaluación.
        criterion (nn.Module): La función de pérdida.
        device (torch.device): El dispositivo de ejecución.

    Returns:
        Tuple[float, float]: (Pérdida media, Precisión media).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    eval_loss = running_loss / total
    eval_acc = correct / total
    return eval_loss, eval_acc