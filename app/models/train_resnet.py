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

# --- IMPORTACIONES UNIFICADAS ---
from models.resnet_model import CriminalityResNet
from dataset import FaceDataset, train_transforms, val_transforms
from engine import train_one_epoch, evaluate, run_bias_audit
from database.config import ILLINOIS_PATH, LFW_PATH, ILLINOIS_CSV_PATH

def run_resnet_training(epochs: int = 5, n_trials: int = 3, lfw_csv_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Iniciando Pipeline de Entrenamiento")
    print(f"Dispositivo detectado: {device}")

    # 1. Preparación del Dataset balanceado y auditado
    full_dataset = FaceDataset(
        illinois_path=ILLINOIS_PATH,
        lfw_path=LFW_PATH,
        illinois_csv=ILLINOIS_CSV_PATH,
        lfw_csv=lfw_csv_path,
        transform=train_transforms,
        samples_per_class=500
    )
    
    total_imgs = len(full_dataset)
    train_size = int(0.8 * total_imgs)
    val_size = total_imgs - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # 2. Optimización del Learning Rate
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        model = CriminalityResNet().to(device)
        optimizer = optim.Adam(model.resnet.fc.parameters(), lr=lr)
        criterion = nn.BCELoss()
        for _ in range(2): 
            train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, val_acc = evaluate(model, val_loader, criterion, device)
        return val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_lr = study.best_params["lr"]
    
    # 3. Entrenamiento Final
    model = CriminalityResNet().to(device)
    optimizer = optim.Adam(model.resnet.fc.parameters(), lr=best_lr)
    criterion = nn.BCELoss()

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Época {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("app/models", exist_ok=True)
            torch.save(model.state_dict(), "app/models/resnet_criminality.pth")

    # 4. Auditoría de Sesgo Final (Matriz de Confusión)
    run_bias_audit(model, val_loader, device, model_name="ResNet50_Final")

    return model, val_loader, device

def classify_from_url(url: str, model_path: str = "app/models/resnet_criminality.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CriminalityResNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        input_tensor = val_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob_risk = model(input_tensor).item()
        
        clase = "RIESGO" if prob_risk > 0.5 else "ESTÁNDAR"
        plt.imshow(img)
        plt.title(f"Predicción: {clase} ({prob_risk*100:.2f}%)")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")