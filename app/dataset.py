"""
FICHERO: dataset.py
PROYECTO: Detección de Riesgo Biométrico (ResNet/VGG)
DESCRIPCIÓN: Implementa la clase FaceDataset para la carga de imágenes, 
             gestionando la unificación de fuentes (Illinois/LFW), el 
             balanceo de clases y la auditoría de diversidad étnica.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from database.config import PROCESSED_ILL_PATH, PROCESSED_LFW_PATH

class BalancedFaceDataset(Dataset):
    def __init__(self, illinois_csv, lfw_csv, transform=None):
        self.transform = transform
        
        # 1. Cargar y normalizar metadatos
        df_ill = pd.read_csv(illinois_csv, sep=';', on_bad_lines='skip', engine='python')
        df_ill['race'] = df_ill['race'].astype(str).str.lower().str.strip()
        df_ill['filename'] = df_ill['id'].astype(str) + '.jpg'
        
        df_lfw = pd.read_csv(lfw_csv)
        df_lfw['race'] = df_lfw['race'].astype(str).str.lower().str.strip()
        df_lfw['filename'] = df_lfw['image_path'].apply(
            lambda x: os.path.basename(x)
        )

        # 2. Filtrar solo los que realmente se procesaron bien en MTCNN
        ill_proc = set(os.listdir(PROCESSED_ILL_PATH))
        lfw_proc = set(os.listdir(PROCESSED_LFW_PATH))
        
        df_ill = df_ill[df_ill['filename'].isin(ill_proc)]
        df_lfw = df_lfw[df_lfw['filename'].isin(lfw_proc)]

        # 3. Muestreo Estratificado (Balanceo 4-Way)
        counts = [
            len(df_ill[df_ill['race'] == 'white']), len(df_ill[df_ill['race'] == 'black']),
            len(df_lfw[df_lfw['race'] == 'white']), len(df_lfw[df_lfw['race'] == 'black'])
        ]
        min_size = min(counts)
        print(f"⚖️ Balanceando dataset a {min_size} muestras por subgrupo (Total: {min_size*4})")

        g1 = df_ill[df_ill['race'] == 'white'].sample(n=min_size, random_state=42)
        g2 = df_ill[df_ill['race'] == 'black'].sample(n=min_size, random_state=42)
        g3 = df_lfw[df_lfw['race'] == 'white'].sample(n=min_size, random_state=42)
        g4 = df_lfw[df_lfw['race'] == 'black'].sample(n=min_size, random_state=42)

        g1['path'] = g1['filename'].apply(lambda x: os.path.join(PROCESSED_ILL_PATH, x))
        g2['path'] = g2['filename'].apply(lambda x: os.path.join(PROCESSED_ILL_PATH, x))
        g3['path'] = g3['filename'].apply(lambda x: os.path.join(PROCESSED_LFW_PATH, x))
        g4['path'] = g4['filename'].apply(lambda x: os.path.join(PROCESSED_LFW_PATH, x))
        
        g1['label'], g2['label'] = 1, 1 # Riesgo
        g3['label'], g4['label'] = 0, 0 # No Riesgo

        self.data = pd.concat([g1, g2, g3, g4]).sample(frac=1).reset_index(drop=True)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        origen = "Illinois" if row['label'] == 1 else "LFW"
        return img, torch.tensor(row['label'], dtype=torch.float32), row['path'], origen

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])