import os
import glob
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from deepface import DeepFace
from .eda_base import EDABase

class IllinoisEDA(EDABase):
    def __init__(self, data_dir, csv_path):
        super().__init__(data_dir)
        self.csv_path = csv_path
        self.df_metadata = None
        self.front_images = []

    def load_metadata(self):
        """Loads and cleans the Illinois DOC metadata from CSV."""
        if not os.path.exists(self.csv_path):
            print(f"Metadata file not found at {self.csv_path}")
            return None
        
        print(f"Loading metadata from {self.csv_path}...")
        df = pd.read_csv(self.csv_path, sep=';', na_values=['N/A', ' N/A', 'N/A '], engine='python')
        df = df.dropna(axis=1, how='all')
        
        df['race'] = df['race'].astype(str).str.strip().str.title()
        df['sex'] = df['sex'].astype(str).str.strip().str.title()
        
        current_year = 2026
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        df['age'] = current_year - df['date_of_birth'].dt.year
        
        self.df_metadata = df
        return df

    def filter_front_images(self):
        """Finds all images in the 'faces' subdirectory."""
        search_pattern = os.path.join(self.data_dir, 'faces', '*.[jJ][pP][gG]')
        self.front_images = glob.glob(search_pattern)
        print(f"Total front images found: {len(self.front_images)}")
        return self.front_images

    def analyze_quality(self, sample_size=500):
        """Analyzes brightness and contrast distribution for Illinois."""
        if not self.front_images: return
        
        print(f"Analyzing quality for {min(sample_size, len(self.front_images))} images...")
        brightness_list = []
        contrast_list = []
        
        sample = random.sample(self.front_images, min(sample_size, len(self.front_images)))
        for img_path in tqdm(sample):
            img = cv2.imread(img_path)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                v_channel = hsv[:, :, 2]
                brightness_list.append(np.mean(v_channel))
                contrast_list.append(np.std(v_channel))
                
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(brightness_list, kde=True, color='gold')
        plt.title('Brightness Distribution (Illinois)')
        
        plt.subplot(1, 2, 2)
        sns.histplot(contrast_list, kde=True, color='teal')
        plt.title('Contrast Distribution (Illinois)')
        
        self.save_plot(plt, 'illinois_quality.png')
        plt.close()

    def generate_average_face(self, sample_size=500, target_size=(250, 250)):
        """Generates an average face from Illinois images."""
        if not self.front_images: return
        
        print(f"Generating average face for Illinois...")
        avg_img = np.zeros((target_size[0], target_size[1], 3), np.float32)
        sample = random.sample(self.front_images, min(sample_size, len(self.front_images)))
        
        count = 0
        for img_path in tqdm(sample):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                avg_img += img.astype(np.float32)
                count += 1
        
        if count > 0:
            avg_img /= count
            avg_img = cv2.cvtColor(avg_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(avg_img)
            plt.axis('off')
            plt.title(f'Average Face (Illinois, n={count})')
            self.save_plot(plt, 'illinois_average_face.png')
            plt.close()

    def analyze_emotions(self, sample_size=50):
        """Uses DeepFace to analyze emotion distribution in Illinois."""
        if not self.front_images: return
        
        print(f"Analyzing emotions for {min(sample_size, len(self.front_images))} images (DeepFace)...")
        emotions_list = []
        sample = random.sample(self.front_images, min(sample_size, len(self.front_images)))
        
        for img_path in tqdm(sample):
            try:
                results = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False, silent=True)
                emotions_list.append(results[0]['dominant_emotion'])
            except Exception:
                continue
        
        if emotions_list:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=emotions_list, palette="coolwarm")
            plt.title('Dominant Emotion Distribution (Illinois Sample)')
            plt.tick_params(axis='x', rotation=45)
            self.save_plot(plt, 'illinois_emotions.png')
            plt.close()

    def plot_demographics(self):
        """Plots demographic distributions from CSV."""
        if self.df_metadata is None: return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Race Distribution
        sns.countplot(data=self.df_metadata, x='race', ax=axes[0], palette="viridis")
        axes[0].set_title('Race Distribution in Illinois DOC')
        axes[0].tick_params(axis='x', rotation=45)

        # 2. Age and Sex Distribution
        sns.histplot(data=self.df_metadata, x='age', hue='sex', multiple="stack", bins=30, ax=axes[1], palette="muted")
        axes[1].set_title('Demographic Distribution: Age & Sex')

        plt.tight_layout()
        self.save_plot(plt, 'illinois_demographics.png')
        plt.close()

    def run_all(self):
        """Runs the full EDA pipeline for Illinois DOC."""
        if self.load_metadata() is not None:
            self.filter_front_images()
            self.plot_demographics()
            self.analyze_quality()
            self.generate_average_face()
            self.analyze_emotions()
