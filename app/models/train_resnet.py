import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from models.resnet_model import CriminalityResNet
from dataset import CriminalityDataset, get_default_transforms
from engine import train_one_epoch, evaluate
from database.config import ILLINOIS_PATH, LFW_PATH, ILLINOIS_CSV_PATH

# Importaciones relativas al sistema de carpetas del proyecto
# Asumimos que 'app' está en el PYTHONPATH al ejecutar desde el notebook
from models.resnet_model import CriminalityResNet
from dataset import CriminalityDataset, get_default_transforms
from engine import train_one_epoch, evaluate
from database.config import ILLINOIS_PATH, LFW_PATH #

def run_resnet_training(epochs: int = 5, n_trials: int = 3):
    """
    Orquesta el entrenamiento completo, la optimización y la evaluación.
    Imprime todos los parámetros necesarios para la explicación de resultados.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nConfiguración de Entrenamiento ResNet")
    print(f"Dispositivo: {device}")

    transform = get_default_transforms()
    full_dataset = CriminalityDataset(transform=transform, balance=True)
    
    total_imgs = len(full_dataset)
    train_size = int(0.8 * total_imgs)
    val_size = total_imgs - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    print(f"Cantidad total de imágenes (balanceadas): {total_imgs}")
    print(f"Distribución: {train_size} (Train) / {val_size} (Val/Test)")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        model = CriminalityResNet().to(device)
        optimizer = optim.Adam(model.resnet.fc.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        for _ in range(2): 
            train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, val_acc = evaluate(model, val_loader, criterion, device)
        return val_acc

    print("\nBuscando el mejor Learning Rate con Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_lr = study.best_params["lr"]
    
    print(f"\nIniciando Entrenamiento Final con LR: {best_lr}")
    model = CriminalityResNet().to(device)
    optimizer = optim.Adam(model.resnet.fc.parameters(), lr=best_lr)
    criterion = nn.BCELoss()

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Época {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Guardamos en la carpeta models del repositorio
            torch.save(model.state_dict(), "app/models/resnet_criminality.pth")

    print("\nRESULTADOS FINALES:")
    print(f"Modelo: ResNet50 (Transfer Learning)")
    print(f"Hiperparámetros finales: Adam Optimizer, LR={best_lr}")
    print(f"Imágenes utilizadas: {total_imgs} (50% Riesgo / 50% Estándar)")
    print(f"Accuracy de Entrenamiento final: {train_acc*100:.2f}%")
    print(f"Accuracy de Validación (Best): {best_acc*100:.2f}%")

def classify_from_url(url: str, model_path: str = "app/models/resnet_criminality.pth"):
    """
    Descarga una imagen, la muestra y la clasifica.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CriminalityResNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_default_transforms()
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob_risk = model(input_tensor).item()
    
    prob_standard = 1.0 - prob_risk
    clase = "RIESGO (Delincuente)" if prob_risk > 0.5 else "ESTÁNDAR (No Delincuente)"

    plt.imshow(img)
    plt.title(f"Predicción: {clase}")
    plt.axis('off')
    plt.show()

    print(f"\nAnálisis de Probabilidad")
    print(f"Clasificado como: {clase}")
    print(f"Probabilidad de Riesgo: {prob_risk*100:.2f}%")
    print(f"Probabilidad Estándar: {prob_standard*100:.2f}%")