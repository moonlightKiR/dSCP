import os

# 1. Ruta base del proyecto
BASE_PATH = os.getcwd()

# 2. Rutas de IMÁGENES ORIGINALES (Entrada)
ILLINOIS_PATH = os.path.join(BASE_PATH, "data/illinois")
LFW_PATH = os.path.join(BASE_PATH, "data/lfw")

# Asegurar que existan las carpetas de datos originales
os.makedirs(ILLINOIS_PATH, exist_ok=True)
os.makedirs(LFW_PATH, exist_ok=True)

# 3. Rutas de IMÁGENES PROCESADAS (Salida del MTCNN)
PROCESSED_ROOT = os.path.join(BASE_PATH, "app/data_processed")
PROCESSED_ILL_PATH = os.path.join(PROCESSED_ROOT, "illinois")
PROCESSED_LFW_PATH = os.path.join(PROCESSED_ROOT, "lfw")

os.makedirs(PROCESSED_ILL_PATH, exist_ok=True)
os.makedirs(PROCESSED_LFW_PATH, exist_ok=True)

# 4. Rutas de METADATOS (CSVs)
ILLINOIS_CSV_PATH = os.path.join(ILLINOIS_PATH, "person.csv")
LFW_CSV_PATH = os.path.join(BASE_PATH, "app/database/lfw_race_metadata.csv")

# 5. Configuración Kaggle (Se mantiene igual)
ILLINOIS_DB = "davidjfisher/illinois-doc-labeled-faces-dataset"
LFW_DB = "jessicali9530/lfw-dataset"

# Hashes MD5 actualizados para el estado actual (reorganizado)
ILLINOIS_MD5 = "ae4d7341cae9eeb6794057ff738c1432"
LFW_MD5 = "7a844a32b6414adaec516a92879f89be"