import shutil
import hashlib
from pathlib import Path
from .kaggle_retreiver import KaggleRetreiver
from .reconstructor import Reconstructor
from .config import (
    ILLINOIS_DB,
    ILLINOIS_PATH,
    LFW_DB,
    LFW_PATH,
    ILLINOIS_MD5,
    LFW_MD5,
)


class Checker:
    def __init__(self):
        self.path_illinois = Path(ILLINOIS_PATH)
        self.path_lfw = Path(LFW_PATH)

    def _check_illinois(self) -> bool:
        if not self.path_illinois.exists():
            print(
                f"ERROR: Directorio ILLINOIS no existe en {self.path_illinois}"
            )
            return False

        has_data = any(self.path_illinois.iterdir())
        if has_data:
            print("Directorio ILLINOIS correcto (no vacío).")
        else:
            print("ERROR: Directorio ILLINOIS está vacío.")
        return has_data

    def _check_lfw(self) -> bool:
        if not self.path_lfw.exists():
            print(f"ERROR: Directorio LFW no existe en {self.path_lfw}")
            return False

        has_data = any(self.path_lfw.iterdir())
        if has_data:
            print("Directorio LFW correcto (no vacío).")
        else:
            print("ERROR: Directorio LFW está vacío.")
        return has_data

    def _calculate_data_md5(self, path: str):

        data_path = Path(path)
        if not data_path.exists():
            print(f"ERROR: El directorio {path} no existe.")
            return None

        print(
            f"Calculando MD5 de la carpeta {path}"
            "(esto puede tardar un poco)..."
        )
        md5_hash = hashlib.md5()

        files = sorted(
            [
                f
                for f in data_path.rglob("*")
                if f.is_file()
                and f.name != ".DS_Store"
                and not f.name.startswith("._")
            ]
        )

        for file_path in files:
            md5_hash.update(str(file_path.relative_to(data_path)).encode())
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)

        result = md5_hash.hexdigest()
        print(f"MD5 de la carpeta data: {result}")
        return result

    def full_check(self):
        database = KaggleRetreiver()
        reconstructor = Reconstructor()

        illinois_md5 = self._calculate_data_md5(ILLINOIS_PATH)
        if illinois_md5 != ILLINOIS_MD5:
            print("Illinois data corrupted or missing, redownloading...")
            shutil.rmtree(ILLINOIS_PATH, ignore_errors=True)
            database.download_data(ILLINOIS_DB, ILLINOIS_PATH)
            reconstructor.clean_illinois()
            reconstructor.reorganize_illinois()
        else:
            print("All ILLINOIS data ok!")
            # Aseguramos que las carpetas extras estén borradas aunque el MD5 sea correcto
            reconstructor.clean_illinois()

        lfw_md5 = self._calculate_data_md5(LFW_PATH)
        if lfw_md5 != LFW_MD5:
            print("LFW data corrupted or missing, redownloading...")
            shutil.rmtree(LFW_PATH, ignore_errors=True)
            database.download_data(LFW_DB, LFW_PATH)
            reconstructor.clean_lfw()
            reconstructor.reorganize_lfw()
        else:
            print("All LFW data ok!")
            # Aseguramos limpieza de LFW también
            reconstructor.clean_lfw()
