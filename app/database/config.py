import os

BASE_PATH = "/content/drive/MyDrive/dsCP/dSCP"

ILLINOIS_PATH = os.path.join(BASE_PATH, "data/illinois/front/front")
LFW_PATH = os.path.join(BASE_PATH, "data/lfw/lfw-deepfunneled/lfw-deepfunneled")

ILLINOIS_CSV_PATH = os.path.join(BASE_PATH, "data/illinois/person.csv")
LFW_CSV_PATH = os.path.join(BASE_PATH, "app/database/lfw_race_metadata.csv")

ILLINOIS_DB = "davidjfisher/illinois-doc-labeled-faces-dataset"
LFW_DB = "jessicali9530/lfw-dataset"
ILLINOIS_MD5 = "bbba276ae0449a0ee932cbc35057bce5"
LFW_MD5 = "71be1d36f5c0c0dd0fd4e0e1c660eebf"