import os
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import requests
from io import BytesIO

# Importes de tu estructura
from models.resnet_model import CriminalityResNet
from models.vgg_model import CriminalityVGG16
from engine import train_one_epoch, evaluate, run_bias_audit
from dataset import BalancedFaceDataset, train_transforms, val_transforms
from database.config import ILLINOIS_CSV_PATH, LFW_CSV_PATH

def run_full_experiment(model_type="resnet", epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds = BalancedFaceDataset(ILLINOIS_CSV_PATH, LFW_CSV_PATH, transform=train_transforms)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    
    def collate_fn(batch):
        return (torch.stack([x[0] for x in batch]), 
                torch.stack([x[1] for x in batch]), 
                [x[2] for x in batch], 
                [x[3] for x in batch])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    if model_type.lower() == "vgg":
        model = CriminalityVGG16().to(device)
        model_name = "VGG16_Final"
    else:
        model = CriminalityResNet().to(device)
        model_name = "ResNet50_Final"

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.BCELoss()

    print(f"\nIniciando entrenamiento del modelo: {model_name}")
    for epoch in range(epochs):
        model.train()
        for imgs, labels, _, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        print(f"Época {epoch+1}/{epochs} completada.")

    audit_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn) 
    run_bias_audit(model, audit_loader, device, model_name)

    return model, val_ds, device

def visual_explanation(model, img_tensor, orig_path, label_real, dataset_name, device):
    model.eval()
    
    if hasattr(model, 'resnet'):
        target_layers = [model.resnet.layer4[-1]]
        m_type = "resnet"
    elif hasattr(model, 'vgg'):
        target_layers = [model.vgg.features[28]] 
        m_type = "vgg"
    else:
        raise AttributeError("Modelo no reconocido")

    cam = GradCAM(model=model, target_layers=target_layers)
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    img_pil = Image.open(orig_path).convert('RGB').resize((224, 224))
    img_array = np.array(img_pil) / 255.0
    
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    
    with torch.no_grad():
        prob = model(input_tensor).item()
    pred = "Riesgo" if prob > 0.5 else "No Riesgo"
    
    plt.figure(figsize=(6,6))
    plt.imshow(visualization)
    plt.title(f"Modelo: {m_type.upper()} | Real: {dataset_name}\nPred: {pred} ({prob:.2%})")
    plt.axis('off')
    
    os.makedirs("reports", exist_ok=True)
    nombre_archivo = f"reports/heatmap_{m_type}_{os.path.basename(orig_path)}"
    plt.savefig(nombre_archivo, bbox_inches='tight')
    print(f"[{m_type.upper()}] Mapa de calor guardado: {nombre_archivo}")
    plt.close()

def classify_from_url(model, url: str, device, person_name: str):
    model.eval()
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        orig_img = Image.open(BytesIO(response.content)).convert('RGB')
        
        input_tensor = val_transforms(orig_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob = model(input_tensor).item()
        pred_label = "Riesgo" if prob > 0.5 else "No Riesgo"
        
        if hasattr(model, 'resnet'):
            target_layers = [model.resnet.layer4[-1]]
            m_type = "resnet"
        elif hasattr(model, 'vgg'):
            target_layers = [model.vgg.features[28]]
            m_type = "vgg"
        
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        
        img_pil_resized = orig_img.resize((224, 224))
        visualization = show_cam_on_image(np.array(img_pil_resized)/255.0, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(6,6))
        plt.imshow(visualization)
        plt.title(f"Sujeto: {person_name}\n{m_type.upper()} | Pred: {pred_label} ({prob:.2%})")
        plt.axis('off')
        
        os.makedirs("reports", exist_ok=True)
        nombre_archivo = f"reports/url_{person_name.replace(' ', '_').lower()}_{m_type}.png"
        plt.savefig(nombre_archivo, bbox_inches='tight')
        print(f"[{m_type.upper()}] Análisis URL guardado: {nombre_archivo}")
        plt.close()
        
    except Exception as e:
        print(f"Error en URL de {person_name}: {e}")