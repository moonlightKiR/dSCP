import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from deepface import DeepFace
from .eda_base import EDABase

class LFWEDA(EDABase):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.df = None

    def build_dataframe(self):
        """Scans the LFW directory and builds a DataFrame."""
        print(f"Scanning {self.data_dir}...")
        data = []
        with os.scandir(self.data_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    images = [f for f in os.listdir(entry.path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    for img in images:
                        data.append({
                            'Name': entry.name.replace('_', ' '),
                            'Image_Path': os.path.join(entry.path, img)
                        })

        self.df = pd.DataFrame(data)
        if not self.df.empty:
            photo_counts = self.df['Name'].value_counts().reset_index()
            photo_counts.columns = ['Name', 'Num_Photos_per_Person']
            self.df = self.df.merge(photo_counts, on='Name')
            print(f"Total images: {len(self.df)} | Unique identities: {self.df['Name'].nunique()}")
        return self.df

    def plot_top_identities(self, n=10):
        """Plots the Top N people with most photographs."""
        if self.df is None: return
        top_n = self.df['Name'].value_counts().head(n)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_n.values, y=top_n.index, palette="mako")
        plt.title(f'Top {n} People with most images in LFW')
        plt.xlabel('Number of Photographs')
        plt.tight_layout()
        self.save_plot(plt, 'lfw_top_identities.png')
        plt.close()

    def analyze_quality(self, sample_size=500):
        """Analyzes brightness and contrast on a sample of images."""
        if self.df is None: return
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        brightness, contrast = [], []

        print(f"Analyzing quality for {len(sample_df)} images...")
        for path in tqdm(sample_df['Image_Path']):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                brightness.append(np.mean(img))
                contrast.append(np.std(img))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(brightness, bins=30, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title('Mean Brightness Distribution (Grayscale)')
        sns.histplot(contrast, bins=30, kde=True, ax=axes[1], color='salmon')
        axes[1].set_title('Contrast Distribution (Std Dev)')
        plt.tight_layout()
        self.save_plot(plt, 'lfw_image_quality.png')
        plt.close()

    def generate_average_face(self, sample_size=500):
        """Generates an average face image from the dataset."""
        if self.df is None: return
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        first_img = cv2.imread(sample_df.iloc[0]['Image_Path'])
        if first_img is None: return
        
        avg_img = np.zeros_like(first_img, dtype=np.float32)
        valid_images = 0
        
        print("Generating average face...")
        for path in tqdm(sample_df['Image_Path']):
            img = cv2.imread(path)
            if img is not None and img.shape == first_img.shape:
                avg_img += img
                valid_images += 1

        if valid_images > 0:
            avg_img = np.uint8(avg_img / valid_images)
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(avg_img, cv2.COLOR_BGR2RGB))
            plt.title('Average Image (LFW)')
            plt.axis('off')
            self.save_plot(plt, 'lfw_average_face.png')
            plt.close()

    def analyze_emotions(self, sample_size=50):
        """Analyzes emotions using DeepFace."""
        if self.df is None: return
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        emotions = []

        print(f"Analyzing emotions for {len(sample_df)} images (using DeepFace en {self.device})...")
        for path in tqdm(sample_df['Image_Path']):
            try:
                # Usamos detector_backend='skip' si las caras ya están alineadas/recortadas para máxima velocidad
                analysis = DeepFace.analyze(
                    path, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv', # Opencv es el más rápido para CPU/GPU básico
                    silent=True
                )
                emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
                emotions.append(emotion)
            except Exception:
                emotions.append('Unknown')

        emotion_counts = pd.Series(emotions).value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="rocket")
        plt.title('Emotion Distribution in LFW')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.save_plot(plt, 'lfw_emotions.png')
        plt.close()

    def run_all(self):
        """Runs the full EDA pipeline for LFW."""
        if self.build_dataframe() is not None:
            self.plot_top_identities()
            self.analyze_quality()
            self.generate_average_face()
            # Emotions are slow, maybe make it optional or keep sample small
            self.analyze_emotions()

    def generate_ethnicity_csv(self, output_csv_path="lfw_race_metadata.csv"):
        """
        Usa DeepFace para predecir la etnia de cada imagen en LFW y genera un CSV.
        Optimizado para usar aceleración por hardware.
        """
        if self.df is None: 
            self.build_dataframe()

        print(f"🚀 Iniciando escaneo de etnias para {len(self.df)} imágenes en {self.device}...")
        results = []
        
        # Procesamiento secuencial pero optimizado por el backend de TensorFlow
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            img_path = row['Image_Path']
            try:
                # detector_backend='opencv' es el equilibrio perfecto entre velocidad y precisión
                analysis = DeepFace.analyze(
                    img_path, 
                    actions=['race'], 
                    enforce_detection=False, 
                    detector_backend='opencv',
                    silent=True
                )
                dominant_race = analysis[0]['dominant_race']
                
                results.append({
                    'image_path': img_path,
                    'race': dominant_race,
                    'label': 0 # No Riesgo
                })
            except Exception:
                continue

        df_race = pd.DataFrame(results)
        df_race.to_csv(output_csv_path, index=False)
        print(f"✅ CSV de etnias generado en: {output_csv_path}")
        return df_race
