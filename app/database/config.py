import os

# 1. Ruta base del proyecto
BASE_PATH = os.getcwd()

# 2. Rutas de IMÁGENES ORIGINALES (Entrada)
# Apuntamos a la raíz del dataset para que el Reconstructor gestione las subcarpetas
ILLINOIS_PATH = os.path.join(BASE_PATH, "data/illinois")
LFW_PATH = os.path.join(BASE_PATH, "data/lfw")

# 3. Rutas de IMÁGENES PROCESADAS (Salida del MTCNN)
PROCESSED_ROOT = os.path.join(BASE_PATH, "data_processed")
PROCESSED_ILL_PATH = os.path.join(PROCESSED_ROOT, "illinois")
PROCESSED_LFW_PATH = os.path.join(PROCESSED_ROOT, "lfw")

# 4. Rutas de METADATOS (CSVs)
ILLINOIS_CSV_PATH = os.path.join(BASE_PATH, "data/illinois/person.csv")
METADATA_ROOT = os.path.join(BASE_PATH, "metadata")
LFW_CSV_PATH = os.path.join(METADATA_ROOT, "lfw_race_metadata.csv")

# 5. Configuración Kaggle (Se mantiene igual)
ILLINOIS_DB = "davidjfisher/illinois-doc-labeled-faces-dataset"
LFW_DB = "jessicali9530/lfw-dataset"
ILLINOIS_MD5 = "bbba276ae0449a0ee932cbc35057bce5"
LFW_MD5 = "71be1d36f5c0c0dd0fd4e0e1c660eebf"