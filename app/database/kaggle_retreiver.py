import os
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleRetreiver:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def _check_data(self):
        if os.path.exists("data") and any(os.scandir("data")):
            print("Data already available in 'data' directory. Skipping download.")
            return True
        return False

    def _purge_unnecesary_data(self):
        if not os.path.exists("data"):
            return

        for filename in os.listdir("data"):
            filepath = os.path.join("data", filename)
            if os.path.isfile(filepath):
                if filename.lower().startswith("readme") or filename.endswith(
                    (".py", ".torrent")
                ):
                    try:
                        os.remove(filepath)
                        print(f"  - Deleted unnecessary file: {filename}")
                    except OSError:
                        pass

    def download_data(self):
        if self._check_data():
            self._purge_unnecesary_data()
            return

        self.api.dataset_download_files(
            "davidjfisher/illinois-doc-labeled-faces-dataset", path="data", unzip=True
        )
        print("Download complete!")
        self._purge_unnecesary_data()
