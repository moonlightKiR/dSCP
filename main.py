from app.database import kaggle_retreiver

def main():
    database = kaggle_retreiver.KaggleRetreiver()
    database.download_data()

if __name__ == "__main__":
    main()
