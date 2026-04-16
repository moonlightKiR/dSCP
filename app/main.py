"""
import os
import sys
import torch
import random

# Añadir carpetas al path para que los imports funcionen
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
    print("====================================================")
    print("   INICIANDO PIPELINE DE DETECCIÓN BIOMÉTRICA")
    print("====================================================\n")
  
    print("[PASO 0] Verificando datos originales...")
    print(">> Datos detectados. Saltando verificación profunda.")

    # [PASO 1] EDA (Solo se hace una vez)
    print("\n[PASO 1] Ejecutando EDA sobre imágenes originales...")
    ill_eda = IllinoisEDA(ILLINOIS_PATH, ILLINOIS_CSV_PATH)
    lfw_eda = LFWEDA(LFW_PATH)
    ill_eda.run_all() 

    if not os.path.exists(LFW_CSV_PATH) or os.path.getsize(LFW_CSV_PATH) == 0:
        print(">> Generando metadatos étnicos...")
        lfw_eda.generate_ethnicity_csv(LFW_CSV_PATH)
    else:
        print(">> Metadatos de LFW detectados.")
    
    # [PASO 2] Preprocesamiento (Solo se hace una vez)
    print("\n[PASO 2] Verificando extracción facial (MTCNN)...")
    ill_procesadas = len(os.listdir(PROCESSED_ILL_PATH)) if os.path.exists(PROCESSED_ILL_PATH) else 0
    lfw_procesadas = len(os.listdir(PROCESSED_LFW_PATH)) if os.path.exists(PROCESSED_LFW_PATH) else 0

    if ill_procesadas > 500 and lfw_procesadas > 500:
        print(f">> Dataset limpio detectado ({ill_procesadas} Illinois, {lfw_procesadas} LFW).")
    else:
        print(">> Faltan imágenes procesadas. Iniciando preprocesamiento...")
        # prep = DataPreprocessor()
        # prep.run_full_preprocessing()

    # --- LISTA DE MODELOS A EVALUAR ---
    modelos_a_entrenar = ["resnet", "vgg"]

    for tipo in modelos_a_entrenar:
        nombre_bonito = "ResNet50" if tipo == "resnet" else "VGG-16"
        print(f"\n\n" + "#"*60)
        print(f"### EJECUTANDO EXPERIMENTO COMPLETO: {nombre_bonito} ###")
        print("#"*60)

        # [PASO 3] Entrenamiento y Auditoría (Matriz de Confusión)
        model, val_ds, device = run_full_experiment(model_type=tipo, epochs=3)

        # [PASO 4] Explicabilidad (Mapas de calor del Dataset)
        print(f"\n[PASO 4] Generando Mapas de Calor ({nombre_bonito})...")
        for i in range(2):
            idx = random.randint(0, len(val_ds)-1)
            img_t, label, path, origen = val_ds[idx]
            visual_explanation(model, img_t, path, label, origen, device)

        # [PASO 5] Pruebas externas (URLs)
        print(f"\n[PASO 5] Analizando sujetos externos con {nombre_bonito}...")
        urls = {
            "Donald Trump": "https://upload.wikimedia.org/wikipedia/commons/1/19/January_2025_Official_Presidential_Portrait_of_Donald_J._Trump.jpg",
            "Charles Manson": "https://ca-times.brightspotcdn.com/dims4/default/d6d33bf/2147483647/strip/true/crop/3240x1824+0+168/resize/840x473!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F86%2Fbd%2F839200e6438c84f2d9793273c4f4%2Fmanson-doc-chaos-latt.png",
            "Gandhi": "https://www.shutterstock.com/image-photo/mahatma-gandhi-indian-independence-activist-260nw-2479250571.jpg",
            "Mussolini": "https://www.nationalww2museum.org/sites/default/files/2020-04/Primary_%20Benito_Mussolini_colored%20photograph%20wearing%20commander%20in%20chief%20uniform%20c%201940%20courtesy%20wikipedia%20-%20Robert%20Citino.jpg",
            "Maria Corina Machado": "https://www.larepublica.ec/wp-content/uploads/2013/07/maria-corina-machado-rostro.jpg"
        }

        for nombre, url in urls.items():
            # Añadimos el tipo de modelo al nombre para que no se sobreescriban los archivos
            id_sujeto = f"{nombre}_{tipo}"
            classify_from_url(model, url, device, id_sujeto)

    print("\n" + "="*60)
    print("   TODOS LOS EXPERIMENTOS HAN FINALIZADO CON ÉXITO")
    print("   Revisa la carpeta /reports para ver los resultados.")
    print("="*60)

if __name__ == "__main__":
    main()

"""


import os
import sys
import torch
import random

# Añadir carpetas al path para que los imports funcionen
sys.path.append(os.path.join(os.getcwd(), 'app'))

from database.config import (
    ILLINOIS_CSV_PATH, 
    LFW_CSV_PATH,
    PROCESSED_ILL_PATH,  
    PROCESSED_LFW_PATH    
)
# Nota: No importamos el preprocesador ni el checker para evitar cargar librerías innecesarias
from models.train_resnet import run_full_experiment, visual_explanation, classify_from_url

def main():
    print("====================================================")
    print("   MODO PRUEBA: EVALUACIÓN EXCLUSIVA VGG-16")
    print("====================================================\n")

    # [PASO 0, 1 y 2] COMENTADOS: Ya tienes la data procesada
    print("[INFO] Saltando Verificación, EDA y Extracción Facial...")
    print(f">> Usando datos de: {PROCESSED_ILL_PATH}")

    # --- CONFIGURACIÓN DE PRUEBA: SOLO VGG ---
    modelos_a_entrenar = ["vgg"] 

    for tipo in modelos_a_entrenar:
        print(f"\n" + "#"*60)
        print(f"### INICIANDO EXPERIMENTO: VGG-16 ###")
        print("#"*60)

        # [PASO 3] Entrenamiento y Auditoría (Matriz de Confusión)
        # Se guarda en reports/confusion_matrix_vgg16_final.png
        model, val_ds, device = run_full_experiment(model_type=tipo, epochs=3)

        # [PASO 4] Explicabilidad (Mapas de calor del Dataset)
        print(f"\n[PASO 4] Generando Mapas de Calor (VGG-16)...")
        for i in range(2):
            idx = random.randint(0, len(val_ds)-1)
            img_t, label, path, origen = val_ds[idx]
            visual_explanation(model, img_t, path, label, origen, device)

        # [PASO 5] Pruebas externas (URLs)
        print(f"\n[PASO 5] Analizando sujetos externos con VGG-16...")
        urls = {
            "Donald Trump": "https://upload.wikimedia.org/wikipedia/commons/1/19/January_2025_Official_Presidential_Portrait_of_Donald_J._Trump.jpg",
            "Charles Manson": "https://ca-times.brightspotcdn.com/dims4/default/d6d33bf/2147483647/strip/true/crop/3240x1824+0+168/resize/840x473!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F86%2Fbd%2F839200e6438c84f2d9793273c4f4%2Fmanson-doc-chaos-latt.png",
            "Gandhi": "https://www.shutterstock.com/image-photo/mahatma-gandhi-indian-independence-activist-260nw-2479250571.jpg",
            "Mussolini": "https://www.nationalww2museum.org/sites/default/files/2020-04/Primary_%20Benito_Mussolini_colored%20photograph%20wearing%20commander%20in%20chief%20uniform%20c%201940%20courtesy%20wikipedia%20-%20Robert%20Citino.jpg",
            "Maria Corina Machado": "https://www.larepublica.ec/wp-content/uploads/2013/07/maria-corina-machado-rostro.jpg"
        }

        for nombre, url in urls.items():
            # El nombre del archivo incluirá '_vgg' para diferenciarlo
            id_sujeto = f"{nombre}_{tipo}"
            classify_from_url(model, url, device, id_sujeto)

    print("\n" + "="*60)
    print("   PRUEBA DE VGG-16 FINALIZADA")
    print("   Resultados disponibles en la carpeta /reports")
    print("="*60)

if __name__ == "__main__":
    main()