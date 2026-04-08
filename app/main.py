from database import Checker, ILLINOIS_PATH, LFW_PATH, ILLINOIS_CSV_PATH
from eda.eda_lfw import LFWEDA
from eda.eda_illinois import IllinoisEDA


def main():
    # 1. Data Integrity Check
    checker = Checker()
    # checker.full_check()

    # 2. Exploratory Data Analysis (EDA)
    print("\n--- Starting EDA for Illinois DOC ---")
    illinois_eda = IllinoisEDA(ILLINOIS_PATH, ILLINOIS_CSV_PATH)
    illinois_eda.run_all()

    print("\n--- Starting EDA for LFW ---")
    lfw_eda = LFWEDA(LFW_PATH)
    lfw_eda.run_all()


if __name__ == "__main__":
    main()
