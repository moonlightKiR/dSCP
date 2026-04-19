import shutil
from pathlib import Path
from .kaggle_retreiver import KaggleRetreiver
from .reconstructor import Reconstructor
from .config import (
    ILLINOIS_DB,
    ILLINOIS_PATH,
    LFW_DB,
    LFW_PATH,
)


class Checker:
    def __init__(self):
        self.path_illinois = Path(ILLINOIS_PATH)
        self.path_lfw = Path(LFW_PATH)

    def _is_structure_correct(
        self, dataset_path: Path, expected_min_count: int
    ) -> bool:
        """Verifica simplemente si la carpeta del dataset existe."""
        return dataset_path.exists() and dataset_path.is_dir()

    def full_check(self):
        print("--- Verificando Dataset ILLINOIS ---")
        if not self._is_structure_correct(self.path_illinois, 0):
            print(
                f"Error: No se encontró la carpeta {self.path_illinois}. Se requiere descarga."
            )
            database = KaggleRetreiver()
            reconstructor = Reconstructor()
            database.download_data(ILLINOIS_DB, ILLINOIS_PATH)
            reconstructor.clean_illinois()
            reconstructor.reorganize_illinois()
        else:
            print(f"Dataset ILLINOIS detectado en {self.path_illinois}")

        print("\n--- Verificando Dataset LFW ---")
        if not self._is_structure_correct(self.path_lfw, 0):
            print(
                f"Error: No se encontró la carpeta {self.path_lfw}. Se requiere descarga."
            )
            database = KaggleRetreiver()
            reconstructor = Reconstructor()
            database.download_data(LFW_DB, LFW_PATH)
            reconstructor.clean_lfw()
            reconstructor.reorganize_lfw()
        else:
            print(f"Dataset LFW detectado en {self.path_lfw}")
