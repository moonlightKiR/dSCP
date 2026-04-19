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

    def _is_structure_correct(self, dataset_path: Path, last_filename: str) -> bool:
        """Verifica si existe la carpeta 'faces', el último archivo y la cantidad total."""
        faces_dir = dataset_path / "faces"
        if not faces_dir.exists() or not faces_dir.is_dir():
            return False
        
        # Verificar existencia del último archivo
        last_file = faces_dir / last_filename
        if not last_file.exists():
            return False

        # Verificar cantidad total de archivos (asumiendo que last_filename es el número total)
        try:
            expected_count = int(Path(last_filename).stem)
            actual_count = len(list(faces_dir.glob("*.jpg")))
            
            if actual_count != expected_count:
                print(f"Error de estructura: Se esperaban {expected_count} fotos, pero se encontraron {actual_count}.")
                return False
        except ValueError:
            pass
            
        return True

    def full_check(self):
        database = KaggleRetreiver()
        reconstructor = Reconstructor()

        print("--- Verificando Dataset ILLINOIS ---")
        # Se espera que Illinois tenga hasta la foto 68491.jpg
        if not self._is_structure_correct(self.path_illinois, "68491.jpg"):
            print("Estructura de Illinois incorrecta o incompleta. Reconstruyendo...")
            shutil.rmtree(self.path_illinois, ignore_errors=True)
            database.download_data(ILLINOIS_DB, ILLINOIS_PATH)
            reconstructor.clean_illinois()
            reconstructor.reorganize_illinois()
        else:
            print("Estructura de ILLINOIS correcta (última foto 68491.jpg encontrada).")

        print("\n--- Verificando Dataset LFW ---")
        # Se espera que LFW tenga hasta la foto 13233.jpg
        if not self._is_structure_correct(self.path_lfw, "13233.jpg"):
            print("Estructura de LFW incorrecta o incompleta. Reconstruyendo...")
            shutil.rmtree(self.path_lfw, ignore_errors=True)
            database.download_data(LFW_DB, LFW_PATH)
            reconstructor.clean_lfw()
            reconstructor.reorganize_lfw()
        else:
            print("Estructura de LFW correcta (última foto 13233.jpg encontrada).")
