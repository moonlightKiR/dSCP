import shutil
from pathlib import Path
from .config import ILLINOIS_PATH, LFW_PATH


class Reconstructor:
    def __init__(self):
        self.path_illinois = Path(ILLINOIS_PATH)
        self.path_lfw = Path(LFW_PATH)
        self.clean_illinois()
        self.clean_lfw()

    def clean_illinois(self):
        if not self.path_illinois.exists():
            return

        folders = ["inmates", "side"]
        for folder in folders:
            folder_path = self.path_illinois / folder
            if folder_path.is_dir():
                shutil.rmtree(folder_path)
                print(f"Directorio ILLINOIS eliminado: {folder_path}")

        patterns = ["readme*", "README*", "*.py", "*.torrent"]
        for pattern in patterns:
            for file_path in self.path_illinois.rglob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    print(f"Fichero ILLINOIS eliminado: {file_path}")

    def reorganize_illinois(self):
        source_dir = self.path_illinois / "front" / "front"
        target_dir = self.path_illinois / "faces"

        if not source_dir.exists():
            print(f"El directorio de origen {source_dir} no existe.")
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in source_dir.iterdir() if f.is_file()])

        for i, file_path in enumerate(files, start=1):
            new_name = f"{i}.jpg"
            target_path = target_dir / new_name
            shutil.move(str(file_path), str(target_path))
            print(f"Movido y renombrado: {file_path.name} -> {new_name}")

        print(
            f"Reorganización de ILLINOIS completada: "
            f"{len(files)} fotos en {target_dir}"
        )

        if (self.path_illinois / "front").exists():
            shutil.rmtree(self.path_illinois / "front")
            print(f"Directorio eliminado: {self.path_illinois / 'front'}")

    def clean_lfw(self):
        pass

    def reorganize_lfw(self):
        source_dir = self.path_lfw / "lfw-deepfunneled" / "lfw-deepfunneled"
        target_dir = self.path_lfw / "faces"

        if not source_dir.exists():
            print(f"El directorio de origen {source_dir} no existe.")
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        # Buscar recursivamente todos los ficheros .jpg
        image_files = sorted(list(source_dir.rglob("*.jpg")))

        for i, file_path in enumerate(image_files, start=1):
            new_name = f"{i}.jpg"
            target_path = target_dir / new_name
            shutil.move(str(file_path), str(target_path))
            print(f"Movido y renombrado: {file_path.name} -> {new_name}")

        print(
            f"Reorganización de LFW completada: "
            f"{len(image_files)} fotos en {target_dir}"
        )

        lfw_extra_folder = self.path_lfw / "lfw-deepfunneled"
        if lfw_extra_folder.exists():
            shutil.rmtree(lfw_extra_folder)
            print(f"Directorio eliminado: {lfw_extra_folder}")
