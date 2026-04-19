import os
import sys
import torch

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
from models.train_resnet import run_full_experiment, visual_explanation
from models.train_resnet import run_full_experiment, visual_explanation, classify_from_url

def main():
    print("INICIANDO PIPELINE DE DETECCIÓN BIOMÉTRICA...")
# 1. CHECKER: Comprobación rápida (Solo cuenta si hay archivos, ignora MD5)
    print("\n[PASO 0] Verificando datos originales...")

    checker = Checker()
    checker.full_check()

    # 2. EDA: Con imágenes originales (con fondo)
    print("\n[PASO 1] Ejecutando EDA sobre imágenes originales...")
    ill_eda = IllinoisEDA(ILLINOIS_PATH, ILLINOIS_CSV_PATH)
    lfw_eda = LFWEDA(LFW_PATH)
    ill_eda.run_all() # Descomentar para generar informes .png

    if not os.path.exists(LFW_CSV_PATH) or os.path.getsize(LFW_CSV_PATH) == 0:
        print(">> Generando CSV de etnias para LFW (DeepFace)...")
        lfw_eda.generate_ethnicity_csv(LFW_CSV_PATH)
    else:
        print(">> Metadatos de LFW ya existen. Saltando...")

    print("\n[PASO 2] Extrayendo rostros (Eliminando fondo)...")
    
    # 1. Diagnóstico de Rutas (La prueba del algodón)
    print(f"-> ¿Existe la ruta {PROCESSED_ILL_PATH}? : {os.path.exists(PROCESSED_ILL_PATH)}")
    print(f"-> ¿Existe la ruta {PROCESSED_LFW_PATH}? : {os.path.exists(PROCESSED_LFW_PATH)}")

    # 2. Conteo
    ill_procesadas = len(os.listdir(PROCESSED_ILL_PATH)) if os.path.exists(PROCESSED_ILL_PATH) else 0
    lfw_procesadas = len(os.listdir(PROCESSED_LFW_PATH)) if os.path.exists(PROCESSED_LFW_PATH) else 0

    print(f"-> Conteo actual: {ill_procesadas} fotos en Illinois, {lfw_procesadas} fotos en LFW.")

    # 3. Lógica de salto
    if ill_procesadas > 500 and lfw_procesadas > 500:
        print(f">> Detectado dataset limpio ({ill_procesadas} Illinois, {lfw_procesadas} LFW).")
        print(">> Saltando extracción facial para ahorrar tiempo...")
    else:
        print(f">>Faltan imágenes. Iniciando MTCNN masivo...")
        # prep = DataPreprocessor()
        # prep.run_full_preprocessing(illinois_limit=20000, lfw_limit=20000)
        print("STOP DE SEGURIDAD: He comentado el preprocesador para no gastar GPU hasta que arreglemos las rutas.")

    print("\n[PASO 3] Iniciando Entrenamiento Balanceado...")
    model, val_ds, device = run_full_experiment(epochs=3)

    print("\n[PASO 4] Generando Ejemplos de Explicabilidad...")
    import random
    for _ in range(2):
        idx = random.randint(0, len(val_ds)-1)
        img_t, label, path, origen = val_ds[idx]
        visual_explanation(model, img_t, path, label, origen, device)

    print("\n[PASO 5] Analizando sujetos externos (URLs)...")
    url_trump = "https://upload.wikimedia.org/wikipedia/commons/1/19/January_2025_Official_Presidential_Portrait_of_Donald_J._Trump.jpg"
    url_manson = "https://ca-times.brightspotcdn.com/dims4/default/d6d33bf/2147483647/strip/true/crop/3240x1824+0+168/resize/840x473!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F86%2Fbd%2F839200e6438c84f2d9793273c4f4%2Fmanson-doc-chaos-latt.png"
    url_gandhi = "https://www.shutterstock.com/image-photo/mahatma-gandhi-indian-independence-activist-260nw-2479250571.jpg"
    url_mussolini = "https://www.nationalww2museum.org/sites/default/files/2020-04/Primary_%20Benito_Mussolini_colored%20photograph%20wearing%20commander%20in%20chief%20uniform%20c%201940%20courtesy%20wikipedia%20-%20Robert%20Citino.jpg"
    url_mariaCorina = "https://www.larepublica.ec/wp-content/uploads/2013/07/maria-corina-machado-rostro.jpg"
    classify_from_url(model, url_trump, device, "Donald Trump")
    classify_from_url(model, url_manson, device, "Charles Manson")
    classify_from_url(model, url_gandhi, device, "Gandhi")
    classify_from_url(model, url_mussolini, device, "Mussolini")
    classify_from_url(model, url_mariaCorina, device, "Maria Corina Machado")

    print("\nPROCESO FINALIZADO.")

if __name__ == "__main__":
    main()