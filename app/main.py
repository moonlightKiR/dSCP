from database import Checker, ILLINOIS_PATH, LFW_PATH, ILLINOIS_CSV_PATH
from eda.eda_lfw import LFWEDA
from eda.eda_illinois import IllinoisEDA
from models.train_resnet import run_resnet_training, classify_from_url

def main():
    imagen_trump = "https://upload.wikimedia.org/wikipedia/commons/1/19/January_2025_Official_Presidential_Portrait_of_Donald_J._Trump.jpg"
    imagen_charlesManson = "https://ca-times.brightspotcdn.com/dims4/default/d6d33bf/2147483647/strip/true/crop/3240x1824+0+168/resize/840x473!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F86%2Fbd%2F839200e6438c84f2d9793273c4f4%2Fmanson-doc-chaos-latt.png"
    # 1. Data Integrity Check
    checker = Checker()
    # checker.full_check()
    """
    # 2. Exploratory Data Analysis (EDA)
    print("\n--- Starting EDA for Illinois DOC ---")
    illinois_eda = IllinoisEDA(ILLINOIS_PATH, ILLINOIS_CSV_PATH)
    illinois_eda.run_all()

    print("\n--- Starting EDA for LFW ---")
    lfw_eda = LFWEDA(LFW_PATH)
    lfw_eda.run_all()
    """
    run_resnet_training(epochs=5, n_trials=3)
    classify_from_url(imagen_trump)
    classify_from_url(imagen_charlesManson)


if __name__ == "__main__":
    main()
