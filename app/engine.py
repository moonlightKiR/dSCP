"""
FICHERO: engine.py
PROYECTO: dSCP - Framework de Entrenamiento Unificado
DESCRIPCIÓN: Contiene la lógica del bucle de entrenamiento y validación. 
             Permite ser usado para diferentes modelos (VGG, ResNet, etc.), 
             garantizando una medición estandarizada de Accuracy, Loss y Sesgo.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

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
        model: El modelo a entrenar (ej. ResNet50, VGG16).
        loader: El cargador de datos (DataLoader) de entrenamiento.
        optimizer: El algoritmo de optimización (ej. Adam).
        criterion: La función de pérdida (ej. BCELoss).
        device: Dispositivo de ejecución (cuda o cpu).

    Returns:
        Tuple[float, float]: (Pérdida media, Precisión media de la época).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=" > Entrenando"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

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
    Evalúa el rendimiento del modelo en un conjunto de datos (validación/test).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    eval_loss = running_loss / total
    eval_acc = correct / total
    return eval_loss, eval_acc

def run_bias_audit(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device,
    model_name: str = "Modelo"
):
    """
    Genera métricas avanzadas para detectar sesgos: Matriz de Confusión e Informe detallado.
    Esencial para verificar si el 100% de Accuracy es real o un 'atajo' racial.
    """
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\n🔍 Iniciando Auditoría de Resultados para {model_name}...")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Riesgo', 'Riesgo'], 
                yticklabels=['No Riesgo', 'Riesgo'])
    plt.title(f'Matriz de Confusión: {model_name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    
    os.makedirs("reports", exist_ok=True)
    plt.savefig(f"reports/confusion_matrix_{model_name.lower()}.png")
    plt.show()

    print(f"\nInforme de Clasificación Detallado - {model_name}:")
    print(classification_report(all_labels, all_preds, target_names=['No Riesgo', 'Riesgo']))
    
    print("-" * 60)
    print("CONSEJO TÉCNICO: Si el F1-Score en 'Riesgo' es mucho más alto que en 'No Riesgo',")
    print("el modelo sigue teniendo sesgo a pesar del balanceo.")
    print("-" * 60)