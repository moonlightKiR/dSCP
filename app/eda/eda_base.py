import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
import platform


class EDABase:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = self._detect_device()
        self._configure_tensorflow()
        sns.set_theme(style="whitegrid")

    def _detect_device(self):
        """Detects the best available device: CUDA, MPS (Metal), or CPU."""
        system = platform.system()

        # 1. Comprobar GPUs físicas visibles para TensorFlow
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            if system == "Darwin":
                print(f"Apple Silicon / Metal (MPS) detectado: {gpus}")
                return "MPS/Metal"
            else:
                print(f"NVIDIA CUDA detectado: {gpus}")
                return "CUDA"

        print("No se detectó GPU acelerada (ni CUDA ni Metal), usando CPU.")
        return "CPU"

    def _configure_tensorflow(self):
        """Configures TF to use the available accelerator."""
        try:
            # En Mac con tensorflow-metal, a veces es necesario asegurar
            # que no intentamos configurar memory_growth si no es CUDA
            gpus = tf.config.list_physical_devices("GPU")
            if gpus and platform.system() != "Darwin":
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Configuración de memoria dinámica (CUDA) activada.")
            elif gpus and platform.system() == "Darwin":
                # Para Metal, la gestión de memoria la hace el plugin automáticamente
                print(
                    "TensorFlow configurado para usar Metal Performance Shaders."
                )
        except Exception as e:
            print(f" Nota sobre configuración de dispositivo: {e}")

    def calculate_brightness(self, image_path):
        """Calculates the average brightness of an image using the V channel of HSV."""
        img = cv2.imread(image_path)
        if img is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return hsv[:, :, 2].mean()
        return None

    def save_plot(self, plt_obj, filename):
        """Saves the current plot to the reports directory."""
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, filename)
        plt_obj.savefig(path)
        print(f"Plot saved to {path}")
