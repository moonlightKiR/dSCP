import os
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
from models.resnet_model import CriminalityResNet
from models.vgg_model import CriminalityVGG16
from engine import train_one_epoch, evaluate, run_bias_audit
from dataset import BalancedFaceDataset, train_transforms, val_transforms
from database.config import ILLINOIS_CSV_PATH, LFW_CSV_PATH
import requests
from io import BytesIO


# Definimos collate_fn fuera para que sea serializable por multiprocessing
def collate_fn(batch):
    return (
        torch.stack([x[0] for x in batch]),
        torch.stack([x[1] for x in batch]),
        [x[2] for x in batch],
        [x[3] for x in batch],
    )


def run_full_experiment(model_type="resnet", epochs=5):
    # Soporte para MPS (Apple Silicon), CUDA o CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    nombre_bonito = "ResNet50" if model_type == "resnet" else "VGG-16"
    print(f"Entrenamiento {nombre_bonito} usando dispositivo: {device}")

    # 1. Dataset
    ds = BalancedFaceDataset(
        ILLINOIS_CSV_PATH, LFW_CSV_PATH, transform=train_transforms
    )
    train_ds, val_ds = random_split(
        ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))]
    )

    # Usamos num_workers > 0 para carga paralela y aumentamos batch_size a 64
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
    )

    if model_type == "vgg":
        model = CriminalityVGG16().to(device)
    else:
        model = CriminalityResNet().to(device)

    # Entrenamos solo los parámetros que requieren gradiente (finetuning)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    criterion = torch.nn.BCELoss()

    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        for imgs, labels, _, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        print(f"Época {epoch + 1}/{epochs} completada.")

    # 2. Matriz de Confusión y Auditoría
    # Usamos un loader simple para la auditoría (compatible con engine.py)
    audit_loader = DataLoader(val_ds, batch_size=32)
    run_bias_audit(model, audit_loader, device, f"{nombre_bonito}_Final")

    return model, val_ds, device


def get_target_layer(model):
    """Devuelve la capa objetivo para Grad-CAM dependiendo del modelo."""
    if hasattr(model, "resnet"):
        # Última capa convolucional de ResNet50
        return [model.resnet.layer4[-1]]
    elif hasattr(model, "vgg"):
        # Última capa convolucional de VGG16 (antes del classifier)
        return [model.vgg.features[-1]]
    return []


def visual_explanation(
    model, img_tensor, orig_path, label_real, dataset_name, device
):
    model.eval()

    # Identificar modelo para el título y capa
    m_name = "ResNet50" if hasattr(model, "resnet") else "VGG-16"
    m_slug = "resnet" if hasattr(model, "resnet") else "vgg"

    target_layers = get_target_layer(model)
    cam = GradCAM(model=model, target_layers=target_layers)

    input_tensor = img_tensor.unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]

    img_pil = Image.open(orig_path).convert("RGB").resize((224, 224))
    visualization = show_cam_on_image(
        np.array(img_pil) / 255.0, grayscale_cam, use_rgb=True
    )

    with torch.no_grad():
        prob = model(input_tensor).item()
    pred = "Riesgo" if prob > 0.5 else "No Riesgo"

    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.title(
        f"Modelo: {m_name} | Real: {dataset_name}\nPred: {pred} ({prob:.2%})"
    )
    plt.axis("off")

    os.makedirs("reports", exist_ok=True)
    nombre_archivo = f"reports/heatmap_{m_slug}_{os.path.basename(orig_path)}"
    plt.savefig(nombre_archivo, bbox_inches="tight")
    print(f"[{m_name}] Mapa de calor guardado en: {nombre_archivo}")

    plt.close()


def classify_from_url(model, url: str, device, person_name: str):
    """Descarga una imagen de internet, predice el riesgo y guarda el Grad-CAM."""
    model.eval()
    m_name = "ResNet50" if hasattr(model, "resnet") else "VGG-16"
    m_slug = "resnet" if hasattr(model, "resnet") else "vgg"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        orig_img = Image.open(BytesIO(response.content)).convert("RGB")

        from dataset import val_transforms

        input_tensor = val_transforms(orig_img).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = model(input_tensor).item()
        pred_label = "Riesgo" if prob > 0.5 else "No Riesgo"

        target_layers = get_target_layer(model)
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]

        img_pil_resized = orig_img.resize((224, 224))
        visualization = show_cam_on_image(
            np.array(img_pil_resized) / 255.0, grayscale_cam, use_rgb=True
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(visualization)
        plt.title(
            f"Modelo: {m_name} | Sujeto: {person_name}\nPredicción: {pred_label} ({prob:.2%})"
        )
        plt.axis("off")

        os.makedirs("reports", exist_ok=True)
        nombre_archivo = (
            f"reports/url_{person_name.replace(' ', '_').lower()}_{m_slug}.png"
        )
        plt.savefig(nombre_archivo, bbox_inches="tight")
        print(f"[{m_name}] Análisis URL guardado en: {nombre_archivo}")
        plt.close()

    except Exception as e:
        print(f"Error procesando la URL de {person_name} con {m_name}: {e}")
