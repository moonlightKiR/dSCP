"""
Unificación: Mezcla el dataset de Illinois DOC (clase 1) y LFW (clase 0) en una 
sola lista maestra.

Balanceo: Como el dataset de Illinois es mucho más grande (68k fotos), esta 
clase selecciona una muestra aleatoria para que el modelo no se sesgue hacia 
una clase solo por cantidad.

Preparación Técnica: Aplica las transformaciones necesarias para que todas las
imágenes sean de 224x224 píxeles y tengan los colores normalizados, requisito
indispensable para ResNet y VGG.
"""
import os
from typing import List, Tuple, Optional, Callable
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from database.config import ILLINOIS_PATH, LFW_PATH, ILLINOIS_CSV_PATH

class CriminalityDataset(Dataset):
    """
    Clase personalizada de PyTorch para cargar y balancear los datasets
    de Illinois DOC y LFW.
    """

    def __init__(self, transform: Optional[Callable] = None, balance: bool = True):
        """
        Inicializa el dataset cargando rutas y etiquetas.

        Args:
            transform (Callable, optional): Transformaciones de torchvision a aplicar.
            balance (bool): Si es True, reduce el dataset de Illinois para igualar al de LFW.
        """
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        lfw_list: List[Tuple[str, int]] = []
        for root, _, files in os.walk(LFW_PATH):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    lfw_list.append((os.path.join(root, file), 0))
        
        num_lfw = len(lfw_list)
        self.samples.extend(lfw_list)

        if os.path.exists(ILLINOIS_CSV_PATH):
            df_illinois = pd.read_csv(ILLINOIS_CSV_PATH, sep=';', engine='python')
            
            if balance:
                df_illinois = df_illinois.sample(n=min(num_lfw, len(df_illinois)), random_state=42)
            
            for _, row in df_illinois.iterrows():
                img_path = os.path.join(ILLINOIS_PATH, "faces", f"{row['ID']}.jpg")
                if os.path.exists(img_path):
                    self.samples.append((img_path, 1))

        print(f"Dataset cargado: {len(self.samples)} imágenes totales ({num_lfw} por clase si balance=True)")

    def __len__(self) -> int:
        """Devuelve el número total de muestras en el dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Carga una imagen, la procesa y devuelve el tensor junto a su etiqueta.

        Args:
            idx (int): Índice de la muestra a recuperar.

        Returns:
            Tuple[torch.Tensor, int]: (Imagen procesada, Etiqueta)
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_default_transforms() -> transforms.Compose:
    """
    Define las transformaciones estándar para modelos de visión tipo ResNet/VGG.

    Returns:
        transforms.Compose: Pipeline de transformaciones (Resize, ToTensor, Normalize).
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])