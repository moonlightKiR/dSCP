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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="Training"):
        images, labels = batch[0].to(device), batch[1].to(device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels.float())

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def run_bias_audit(model, loader, device, model_name="Modelo"):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\nIniciando Auditoría de Resultados para {model_name}...")

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Riesgo", "Riesgo"],
        yticklabels=["No Riesgo", "Riesgo"],
    )
    plt.title(f"Matriz de Confusión: {model_name}")
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Predicción del Modelo")

    os.makedirs("reports", exist_ok=True)
    plt.savefig(f"reports/confusion_matrix_{model_name.lower()}.png")
    plt.show()

    print(f"\nInforme de Clasificación Detallado:")
    print(
        classification_report(
            all_labels, all_preds, target_names=["No Riesgo", "Riesgo"]
        )
    )
