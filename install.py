import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


if os.name == "nt":  # Windows

    # 🛡️ Vérification de la version : maximum 3.12 autorisé
    # sys.version_info renvoie un tuple, par exemple (3, 14, 0) pour Python 3.14
    if sys.version_info > (3, 13):
        print(f"❌ Erreur : Vous utilisez Python {sys.version_info.major}.{sys.version_info.minor}.")
        print("Ce script nécessite Python 3.13 ou une version antérieure (ex: 3.11, 3.10) pour éviter les erreurs de compilation.")
        sys.exit(1)  # Arrête immédiatement le script avec un code d'erreur

    print(f"✅ Version de Python compatible : {sys.version_info.major}.{sys.version_info.minor}") 




#  Defining installation paths
root        = Path.cwd()
atomod_dir  = root / "ATOMOD"
venv_dir    = atomod_dir / "venv" / "ATOMOD" 

# cloning the GitHub repository
url_depot = "https://github.com/hbulou/ATOMOD.git"

# Security: Check if the folder does not already exist.
if not atomod_dir.exists():
    print(f"📥 Cloning the repository {url_depot}...")
    
    try:
        subprocess.run(["git", "clone", url_depot], check=True)
        print("✅ Repository successfully cloned!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during git clone: {e}")
    except FileNotFoundError:
        print("❌ Error: The 'git' command is not installed or accessible on this system.")
else:
    print(f"ℹ️ The directory '{atomod_dir}' already exists. Cloning cancelled.")

# Creation of the virtual environment
if not venv_dir.exists():
    print(f"Creation of the virtual environment in: {venv_dir}")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
else:
    print("The virtual environment already exists.")

if os.name == "nt":  # Windows
    pip_venv = venv_dir / "Scripts" / "pip.exe"
    pip_sys = sys.executable
    subprocess.run([pip_sys, "-m","pip","install", "--no-cache-dir", "--upgrade", "pip"], check=True)

else:  # Linux / macOS
    pip_venv = venv_dir / "bin" / "pip"
    pip_sys = venv_dir / "bin" / "pip"
    # pip update
    subprocess.run([pip_sys, "install", "--no-cache-dir", "--upgrade", "pip"], check=True)


pip_venv = str(pip_venv)  # subprocess préfère une chaîne, surtout sous Windows



# Installation of specific packages
print("📦 Installing packages...")
packages = [
    "jupyterlab",
    "numpy",
    "scipy<1.17",
    "matplotlib",
    "torch==2.12.1",
    "setuptools==81.0.0",
    "pyfftw==0.15.0",
    "mace-torch",
    "dask",
    "tabulate",
    "numba",
    "threadpoolctl",
    "zarr",
    "ipywidgets",
    "py3Dmol",
    "ovito",
    "pyarrow",
    "tensorflow",
    "scikit-learn",
    "umpa-learn",
    "hdbscan",
]
for pack in packages:
    subprocess.run([pip_venv, "install", "--no-cache-dir", pack], check=True)

print("✅ Installation complete!")
