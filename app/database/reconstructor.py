import shutil
from pathlib import Path
from .config import ILLINOIS_PATH, LFW_PATH


class Reconstructor:
    def __init__(self):
        self.path_illinois = Path(ILLINOIS_PATH)
        self.path_lfw = Path(LFW_PATH)

    def clean_illinois(self):
        if not self.path_illinois.exists():
            return

        # 1. Eliminar archivos específicos y patrones molestos primero
        # Incluimos .DS_Store para evitar "Directory not empty" en Mac
        patterns = [
            "readme*",
            "README*",
            "*.py",
            "*.torrent",
            ".DS_Store",
            "._*",
        ]
        for pattern in patterns:
            for file_path in self.path_illinois.rglob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"Fichero eliminado: {file_path}")
                except Exception:
                    pass

        # 2. Intentar borrar las carpetas no deseadas de forma más agresiva
        folders = ["inmates", "side"]
        for folder in folders:
            folder_path = self.path_illinois / folder
            if folder_path.is_dir():
                try:
                    # Intentamos borrar recursivamente
                    shutil.rmtree(folder_path)
                    print(f"Directorio ILLINOIS eliminado: {folder_path}")
                except OSError:
                    # Si falla (ej. por .DS_Store recién creado),
                    # forzamos borrado interno e intentamos de nuevo
                    for item in folder_path.rglob("*"):
                        try:
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                        except Exception:
                            pass
                    shutil.rmtree(folder_path, ignore_errors=True)
                    if not folder_path.exists():
                        print(
                            f"Directorio ILLINOIS eliminado (segundo intento): {folder_path}"
                        )

    def reorganize_illinois(self):
        source_dir = self.path_illinois / "front" / "front"
        target_dir = self.path_illinois / "faces"

        if not source_dir.exists():
            print(f"El directorio de origen {source_dir} no existe.")
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in source_dir.iterdir() if f.is_file()])

        for file_path in files:
            target_path = target_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            print(f"Movido: {file_path.name}")

        print(
            f"Reorganización de ILLINOIS completada: "
            f"{len(files)} fotos en {target_dir}"
        )

        if (self.path_illinois / "front").exists():
            shutil.rmtree(self.path_illinois / "front", ignore_errors=True)
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

        for file_path in image_files:
            # Mantener nombre original pero aplanado: AJ_Cook_0001.jpg
            target_path = target_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            print(f"Movido: {file_path.name}")

        print(
            f"Reorganización de LFW completada: "
            f"{len(image_files)} fotos en {target_dir}"
        )

        lfw_extra_folder = self.path_lfw / "lfw-deepfunneled"
        if lfw_extra_folder.exists():
            shutil.rmtree(lfw_extra_folder, ignore_errors=True)
            print(f"Directorio eliminado: {lfw_extra_folder}")
