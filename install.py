import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# 1. Définition des chemins universels (~/venv/ATOMOD et ~/src)
home = Path.home()
venv_dir = home / "venv" / "ATOMOD"
src_dir = home / "src"

# Détection de l'exécutable Python selon l'OS (Windows vs Linux/WSL)
if os.name == "nt":  # Windows
    venv_python = venv_dir / "Scripts" / "python.exe"
else:  # Linux / WSL / Mac
    venv_python = venv_dir / "bin" / "python"

print("Démarrage de l'installation d'ATOMOD...")

# 2. Création de l'environnement virtuel
if not venv_dir.exists():
    print(f"Création de l'environnement virtuel dans : {venv_dir}")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
else:
    print("L'environnement virtuel existe déjà.")

# 3. Mise à jour de PIP dans l'environnement virtuel
print(" Mise à jour de pip...")
subprocess.run(
    [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True
)

# 4. Installation des paquets requis
packages = [
    "jupyterlab",
    "numpy",
    "scipy<1.17",
    "matplotlib",
    "mace-torch",
    "dask",
    "tabulate",
    "numba",
    "threadpoolctl",
    "zarr",
    "ipywidgets",
    "pyfftw",
]

print(" Installation des bibliothèques Python (cette étape peut prendre du temps)...")
subprocess.run([str(venv_python), "-m", "pip", "install"] + packages, check=True)


# Fonction d'aide pour télécharger, extraire et renommer les dépôts GitHub
def download_and_extract_github(url, zip_name, final_folder_name):
    zip_path = src_dir / zip_name
    final_path = src_dir / final_folder_name

    # Téléchargement (remplace wget)
    print(f" Téléchargement de {final_folder_name}...")
    urllib.request.urlretrieve(url, zip_path)

    # Extraction (remplace unzip)
    print(f" Extraction de {zip_name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Récupère le nom du dossier racine généré par GitHub (ex: site-packages-main)
        root_extracted_folder = zip_ref.namelist()[0].split("/")[0]
        zip_ref.extractall(src_dir)

    # Nettoyage du ZIP (remplace rm)
    zip_path.unlink()

    # Renommage (remplace mv)
    extracted_path = src_dir / root_extracted_folder
    if final_path.exists():
        shutil.rmtree(final_path)  # Nettoie si une ancienne installation existe
    extracted_path.rename(final_path)

    return final_path


# 5. Création du dossier ~/src
src_dir.mkdir(parents=True, exist_ok=True)

# 6. Téléchargement et installation de HBPy
hbpy_path = download_and_extract_github(
    url="https://github.com/hbulou/site-packages/archive/refs/heads/main.zip",
    zip_name="HBPy.zip",
    final_folder_name="HBPy",
)

print("Installation de HBPy en mode éditable (-e)...")
subprocess.run(
    [str(venv_python), "-m", "pip", "install", "-e", "."],
    cwd=str(hbpy_path),
    check=True,
)

# 7. Téléchargement de ATOMOD
download_and_extract_github(
    url="https://github.com/hbulou/ATOMOD/archive/refs/heads/main.zip",
    zip_name="ATOMOD-main.zip",
    final_folder_name="ATOMOD",
)

print("\nInstallation d'ATOMOD complétée avec succès sur votre machine ! ")
print(f"Le code source se trouve dans : {src_dir / 'ATOMOD'}")
