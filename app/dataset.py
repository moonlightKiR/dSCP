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

class FaceDataset(Dataset):
    def __init__(self, illinois_path, lfw_path, illinois_csv, lfw_csv=None, transform=None, samples_per_class=500):
        self.transform = transform
        
        df_ill = pd.read_csv(illinois_csv, sep=';', on_bad_lines='skip')
        df_ill['label'] = 1
        df_ill['full_path'] = df_ill['id'].apply(lambda x: os.path.join(illinois_path, f"{x}.jpg"))

        if lfw_csv and os.path.exists(lfw_csv):
            df_lfw = pd.read_csv(lfw_csv)
            df_lfw['label'] = 0
            df_lfw['full_path'] = df_lfw['image_path']
        else:
            lfw_images = []
            for root, _, files in os.walk(lfw_path):
                for f in files:
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        lfw_images.append(os.path.join(root, f))
            df_lfw = pd.DataFrame({'full_path': lfw_images, 'label': 0, 'race': 'Unknown'})

        df_risk_sample = df_ill.sample(n=min(samples_per_class, len(df_ill)), random_state=42)
        df_std_sample = df_lfw.sample(n=min(samples_per_class, len(df_lfw)), random_state=42)
        self.data = pd.concat([df_risk_sample, df_std_sample]).reset_index(drop=True)
        
        print(f"\nDataset balanceado: {len(self.data)} imágenes totales.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['full_path']
        label = self.data.iloc[idx]['label']
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color='black')
        if self.transform: img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])