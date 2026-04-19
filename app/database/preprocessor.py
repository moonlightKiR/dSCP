import os
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from database.config import (
    ILLINOIS_PATH,
    LFW_PATH,
    PROCESSED_ILL_PATH,
    PROCESSED_LFW_PATH,
)


class DataPreprocessor:
    def __init__(self):
        # Prioridad estricta: MPS (Mac) > CUDA (Nvidia) > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Preprocesador (MTCNN) iniciando en: {self.device}")
        # Detector principal (GPU/MPS)
        self.detector = MTCNN(
            image_size=224, margin=30, post_process=False, device=self.device
        )
        # Detector de respaldo (CPU)
        self.cpu_detector = MTCNN(
            image_size=224,
            margin=30,
            post_process=False,
            device=torch.device("cpu"),
        )

        os.makedirs(PROCESSED_ILL_PATH, exist_ok=True)
        os.makedirs(PROCESSED_LFW_PATH, exist_ok=True)

    def process_single_image_pil(self, img_pil, output_path):
        """Procesa una sola imagen PIL con fallback robusto a CPU."""
        try:
            # Intentar primero con el dispositivo principal
            face_tensor = None
            try:
                face_tensor = self.detector(img_pil)
            except Exception:
                # Fallback a CPU si falla el principal (ej. error de pooling en MPS)
                face_tensor = self.cpu_detector(img_pil)

            if face_tensor is not None:
                # El tensor devuelto por MTCNN con post_process=False tiene shape (3, 224, 224)
                # y valores en [0, 255] pero en float.
                face_img = Image.fromarray(
                    face_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8")
                )
                face_img.save(output_path)
                return True
        except Exception as e:
            print(f" Error crítico guardando {output_path}: {e}")
        return False

    def run_full_preprocessing(
        self, illinois_limit=500, lfw_limit=500, batch_size=32
    ):
        """Recorre los datasets y crea las versiones sin fondo de forma unificada."""

        configs = [
            {
                "name": "Illinois",
                "src": os.path.join(ILLINOIS_PATH, "faces"),
                "dst": PROCESSED_ILL_PATH,
                "limit": illinois_limit,
                "prefix": "",
            },
            {
                "name": "LFW",
                "src": os.path.join(LFW_PATH, "faces"),
                "dst": PROCESSED_LFW_PATH,
                "limit": lfw_limit,
                "prefix": "lfw_",
            },
        ]

        for config in configs:
            print(f"\nProcesando {config['name']} en {config['src']}...")
            if not os.path.exists(config["src"]):
                print(f" Error: No se encontró {config['src']}")
                continue

            # Recolección unificada de archivos
            all_files = []
            for root, _, files in os.walk(config["src"]):
                for f in files:
                    if f.endswith((".jpg", ".jpeg", ".png")):
                        all_files.append(os.path.join(root, f))

            files_to_process = sorted(all_files)[: config["limit"]]

            # Procesamiento individual para mayor fiabilidad (MTCNN batching es caprichoso en MPS)
            # Pero mantenemos la apertura de archivos eficiente
            saved_count = 0
            for p in tqdm(files_to_process):
                out_name = f"{config['prefix']}{os.path.basename(p)}"
                out_path = os.path.join(config["dst"], out_name)

                if not os.path.exists(out_path):
                    try:
                        img = Image.open(p).convert("RGB")
                        if self.process_single_image_pil(img, out_path):
                            saved_count += 1
                    except Exception:
                        continue
            print(
                f" Finalizado {config['name']}: {saved_count} imágenes nuevas guardadas en {config['dst']}"
            )

    def show_example(self, original_path, processed_path):
        """Muestra la comparativa para el informe."""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(original_path))
        plt.title("Original (Con Fondo)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        if os.path.exists(processed_path):
            plt.imshow(Image.open(processed_path))
            plt.title("Procesada (MTCNN - Solo Cara)")
        else:
            plt.text(0.5, 0.5, "Error en proceso", ha="center")
        plt.axis("off")
        plt.show()
