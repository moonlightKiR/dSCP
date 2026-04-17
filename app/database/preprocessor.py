import os
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from database.config import ILLINOIS_PATH, LFW_PATH, PROCESSED_ILL_PATH, PROCESSED_LFW_PATH

class DataPreprocessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(image_size=224, margin=30, post_process=False, device=self.device)
        
        os.makedirs(PROCESSED_ILL_PATH, exist_ok=True)
        os.makedirs(PROCESSED_LFW_PATH, exist_ok=True)

    def process_single_image(self, input_path, output_path):
        """Procesa una imagen: detecta cara, quita fondo y guarda."""
        if os.path.exists(output_path):
            return True # Saltar si ya existe

        try:
            img = Image.open(input_path).convert('RGB')
            face_tensor = self.detector(img)
            
            if face_tensor is not None:
                # Convertir tensor a imagen PIL y guardar
                face_img = Image.fromarray(face_tensor.permute(1, 2, 0).numpy().astype('uint8'))
                face_img.save(output_path)
                return True
            else:
                return False # No se detectó cara
        except Exception as e:
            print(f"Error procesando {input_path}: {e}")
            return False

    def run_full_preprocessing(self, illinois_limit=500, lfw_limit=500):
        """Recorre los datasets originales y crea las versiones sin fondo."""
        
        # El Reconstructor mueve las fotos a una carpeta /faces
        ill_faces_path = os.path.join(ILLINOIS_PATH, "faces")
        lfw_faces_path = os.path.join(LFW_PATH, "faces")

        print("\nProcesando Illinois (Criminality)...")
        if os.path.exists(ill_faces_path):
            ill_files = [f for f in os.listdir(ill_faces_path) if f.endswith('.jpg')][:illinois_limit]
            for f in tqdm(ill_files):
                self.process_single_image(
                    os.path.join(ill_faces_path, f),
                    os.path.join(PROCESSED_ILL_PATH, f)
                )
        else:
            print(f"ERROR: No se encontró la carpeta {ill_faces_path}")

        print("\nProcesando LFW (Standard)...")
        if os.path.exists(lfw_faces_path):
            lfw_files = [f for f in os.listdir(lfw_faces_path) if f.endswith('.jpg')][:lfw_limit]
            for f in tqdm(lfw_files):
                self.process_single_image(
                    os.path.join(lfw_faces_path, f),
                    os.path.join(PROCESSED_LFW_PATH, f)
                )
        else:
            print(f"ERROR: No se encontró la carpeta {lfw_faces_path}")

    def show_example(self, original_path, processed_path):
        """Muestra la comparativa para el informe."""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(original_path))
        plt.title("Original (Con Fondo)")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        if os.path.exists(processed_path):
            plt.imshow(Image.open(processed_path))
            plt.title("Procesada (MTCNN - Solo Cara)")
        else:
            plt.text(0.5, 0.5, "Error en proceso", ha='center')
        plt.axis('off')
        plt.show()