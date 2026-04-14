"""
FICHERO: main.py
UBICACIÓN: /app/main.py
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importamos las rutas y clases desde la estructura de carpetas
from database.config import ILLINOIS_PATH, LFW_PATH, ILLINOIS_CSV_PATH, LFW_CSV_PATH
from database.checker import Checker
from eda.eda_lfw import LFWEDA
from eda.eda_illinois import IllinoisEDA
from models.train_resnet import run_resnet_training, classify_from_url

def main():
    imagen_trump = "https://upload.wikimedia.org/wikipedia/commons/1/19/January_2025_Official_Presidential_Portrait_of_Donald_J._Trump.jpg"
    imagen_manson = "https://ca-times.brightspotcdn.com/dims4/default/d6d33bf/2147483647/strip/true/crop/3240x1824+0+168/resize/840x473!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F86%2Fbd%2F839200e6438c84f2d9793273c4f4%2Fmanson-doc-chaos-latt.png"

    print("\n" + "="*60)
    print("SISTEMA DE CLASIFICACIÓN BIOMÉTRICA - START")
    print("="*60)

    print("\n[Paso 1] Verificando metadatos étnicos de LFW...")
    lfw_eda = LFWEDA(LFW_PATH)

    if os.path.exists(LFW_CSV_PATH) and os.path.getsize(LFW_CSV_PATH) == 0:
      os.remove(LFW_CSV_PATH)

    if not os.path.exists(LFW_CSV_PATH):
        print("Metadatos no encontrados. Generando auditoría con DeepFace...")
        lfw_eda.generate_ethnicity_csv(LFW_CSV_PATH)
    else:
        print(f"Metadatos detectados en: {LFW_CSV_PATH}")

    print("\n[Paso 2] Iniciando Entrenamiento ResNet50 (Transfer Learning)...")
    model, val_loader, device = run_resnet_training(
        epochs=5, 
        n_trials=3, 
        lfw_csv_path=LFW_CSV_PATH
    )

    print("\n[Paso 3] Ejecutando pruebas de detección facial...")
    print("\n> Analizando Perfil 1:")
    classify_from_url(imagen_trump)
    
    print("\n> Analizando Perfil 2:")
    classify_from_url(imagen_manson)

    print("\n" + "="*60)
    print("PROCESO COMPLETADO CON ÉXITO")
    print("="*60)

if __name__ == "__main__":
    main()