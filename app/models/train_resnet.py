import os
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
from models.resnet_model import CriminalityResNet
from engine import train_one_epoch, evaluate, run_bias_audit
from dataset import BalancedFaceDataset, train_transforms, val_transforms
from database.config import ILLINOIS_CSV_PATH, LFW_CSV_PATH
import requests
from io import BytesIO

def run_full_experiment(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset
    ds = BalancedFaceDataset(ILLINOIS_CSV_PATH, LFW_CSV_PATH, transform=train_transforms)
    train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    
    # Custom collate para manejar los paths
    def collate_fn(batch):
        return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch]), [x[2] for x in batch], [x[3] for x in batch]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)

    model = CriminalityResNet().to(device)
    optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    # Entrenamiento (simplificado para el main)
    for epoch in range(epochs):
        model.train()
        for imgs, labels, _, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad(); loss = criterion(model(imgs), labels); loss.backward(); optimizer.step()
        print(f"Época {epoch+1} completada.")

    # 2. Matriz de Confusión (vía Engine)
    # Adaptamos temporalmente el loader para engine.py
    simple_val_loader = DataLoader(val_ds, batch_size=32) 
    run_bias_audit(model, simple_val_loader, device, "ResNet50_Final")

    return model, val_ds, device

def visual_explanation(model, img_tensor, orig_path, label_real, dataset_name, device):
    model.eval()
    target_layers = [model.resnet.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    input_tensor = img_tensor.unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    img_pil = Image.open(orig_path).convert('RGB').resize((224, 224))
    visualization = show_cam_on_image(np.array(img_pil)/255.0, grayscale_cam, use_rgb=True)
    
    prob = model(input_tensor).item()
    pred = "Riesgo" if prob > 0.5 else "No Riesgo"
    
    plt.figure(figsize=(6,6))
    plt.imshow(visualization)
    plt.title(f"Real: {dataset_name} | Pred: {pred} ({prob:.2%})\nArchivo: {os.path.basename(orig_path)}")
    plt.axis('off')
    
    os.makedirs("reports", exist_ok=True)
    nombre_archivo = f"reports/heatmap_{os.path.basename(orig_path)}"
    plt.savefig(nombre_archivo, bbox_inches='tight')
    print(f"Mapa de calor guardado con éxito en: {nombre_archivo}")
    
    plt.close() 


def classify_from_url(model, url: str, device, person_name: str):
    """Descarga una imagen de internet, predice el riesgo y guarda el Grad-CAM."""
    model.eval()
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        orig_img = Image.open(BytesIO(response.content)).convert('RGB')
        
        from dataset import val_transforms 
        input_tensor = val_transforms(orig_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob = model(input_tensor).item()
        pred_label = "Riesgo" if prob > 0.5 else "No Riesgo"
        
        target_layers = [model.resnet.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        
        img_pil_resized = orig_img.resize((224, 224))
        visualization = show_cam_on_image(np.array(img_pil_resized)/255.0, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(6,6))
        plt.imshow(visualization)
        plt.title(f"Sujeto: {person_name}\nPredicción: {pred_label} ({prob:.2%})")
        plt.axis('off')
        
        os.makedirs("reports", exist_ok=True)
        nombre_archivo = f"reports/url_{person_name.replace(' ', '_').lower()}.png"
        plt.savefig(nombre_archivo, bbox_inches='tight')
        print(f"Análisis URL guardado con éxito en: {nombre_archivo}")
        plt.close()
        
    except Exception as e:
        print(f"Error procesando la URL de {person_name}: {e}")