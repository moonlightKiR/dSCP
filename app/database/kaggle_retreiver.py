from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleRetreiver:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def download_data(self, database: str, path: str):
        self.api.dataset_download_files(
            dataset=database,
            path=path,
            unzip=True,
            quiet=False,
        )
