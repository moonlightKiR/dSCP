import os
import sys
import torch
import random

sys.path.append(os.path.join(os.getcwd(), 'app'))

from database.config import (
    ILLINOIS_PATH,
    LFW_PATH,
    ILLINOIS_CSV_PATH,
    LFW_CSV_PATH,
    PROCESSED_ILL_PATH,
    PROCESSED_LFW_PATH
)
from database.checker import Checker
from database.preprocessor import DataPreprocessor
from eda.eda_lfw import LFWEDA
from eda.eda_illinois import IllinoisEDA
from models.train_resnet import run_full_experiment, visual_explanation, classify_from_url

def main():
    os.makedirs("reports", exist_ok=True)
    print("====================================================")
    print("   INICIANDO PIPELINE DE DETECCIÓN BIOMÉTRICA")
    print("====================================================\n")

    # [PASO 0] Verificando datos originales...
    print("[PASO 0] Verificando datos originales...")
    checker = Checker()
    checker.full_check()

    # [PASO 1] Ejecutando EDA sobre imágenes originales...
    print("\n[PASO 1] Ejecutando EDA sobre imágenes originales...")
    ill_eda = IllinoisEDA(ILLINOIS_PATH, ILLINOIS_CSV_PATH)
    lfw_eda = LFWEDA(LFW_PATH)
    
    print(">> Ejecutando análisis para Illinois...")
    ill_eda.run_all() 
    
    print(">> Ejecutando análisis para LFW...")
    lfw_eda.run_all()

    if not os.path.exists(LFW_CSV_PATH) or os.path.getsize(LFW_CSV_PATH) == 0:
        print(">> Generando metadatos étnicos de LFW (DeepFace)...")
        lfw_eda.generate_ethnicity_csv(LFW_CSV_PATH)
    else:
        print(">> Metadatos de LFW ya existen. Saltando...")

    # [PASO 2] Extrayendo rostros (Eliminando fondo)...
    print("\n[PASO 2] Extrayendo rostros (MTCNN)...")
    
    # Diagnóstico de Conteo
    ill_procesadas = len(os.listdir(PROCESSED_ILL_PATH)) if os.path.exists(PROCESSED_ILL_PATH) else 0
    lfw_procesadas = len(os.listdir(PROCESSED_LFW_PATH)) if os.path.exists(PROCESSED_LFW_PATH) else 0

    print(f"-> Conteo actual: {ill_procesadas} fotos en Illinois, {lfw_procesadas} fotos en LFW.")

    # Lógica de salto (si hay suficientes)
    if ill_procesadas > 5000 and lfw_procesadas > 5000:
        print(f">> Dataset procesado detectado ({ill_procesadas} Illinois, {lfw_procesadas} LFW).")
        print(">> Saltando extracción facial para ahorrar tiempo...")
    else:
        print(f">>Faltan imágenes o el dataset está incompleto. Iniciando MTCNN masivo...")
        prep = DataPreprocessor()
        prep.run_full_preprocessing(illinois_limit=20000, lfw_limit=20000)

    # [PASO 3, 4 y 5] Entrenamiento Balanceado y Comparativo
    modelos_a_entrenar = ["resnet", "vgg"]

    for tipo in modelos_a_entrenar:
        nombre_label = "ResNet50" if tipo == "resnet" else "VGG-16"
        print(f"\n\n" + "#"*60)
        print(f"### EJECUTANDO EXPERIMENTO: {nombre_label} ###")
        print("#"*60)

        # [PASO 3] Entrenamiento y Auditoría (Matriz de Confusión)
        model, val_ds, device = run_full_experiment(model_type=tipo, epochs=3)

        # [PASO 4] Generando Ejemplos de Explicabilidad...
        print(f"\n[PASO 4] Generando Mapas de Calor ({nombre_label})...")
        for _ in range(2):
            idx = random.randint(0, len(val_ds)-1)
            img_t, label, path, origen = val_ds[idx]
            visual_explanation(model, img_t, path, label, origen, device)

        # [PASO 5] Analizando sujetos externos (URLs)...
        print(f"\n[PASO 5] Analizando sujetos externos con {nombre_label}...")
        urls = {
            "Donald Trump": "https://upload.wikimedia.org/wikipedia/commons/1/19/January_2025_Official_Presidential_Portrait_of_Donald_J._Trump.jpg",
            "Charles Manson": "https://ca-times.brightspotcdn.com/dims4/default/d6d33bf/2147483647/strip/true/crop/3240x1824+0+168/resize/840x473!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F86%2Fbd%2F839200e6438c84f2d9793273c4f4%2Fmanson-doc-chaos-latt.png",
            "Gandhi": "https://www.shutterstock.com/image-photo/mahatma-gandhi-indian-independence-activist-260nw-2479250571.jpg",
            "Mussolini": "https://www.nationalww2museum.org/sites/default/files/2020-04/Primary_%20Benito_Mussolini_colored%20photograph%20wearing%20commander%20in%20chief%20uniform%20c%201940%20courtesy%20wikipedia%20-%20Robert%20Citino.jpg",
            "Maria Corina Machado": "https://www.larepublica.ec/wp-content/uploads/2013/07/maria-corina-machado-rostro.jpg"
        }

        for nombre, url in urls.items():
            classify_from_url(model, url, device, nombre)

    print("\n" + "="*60)
    print("   TODOS LOS EXPERIMENTOS HAN FINALIZADO CON ÉXITO")
    print("   Revisa la carpeta /reports para ver los resultados comparativos.")
    print("="*60)


if __name__ == "__main__":
    main()
