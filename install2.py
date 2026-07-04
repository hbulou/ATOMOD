# import os
# import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path



# 1. Définition des chemins universels (~/venv/ATOMOD et ~/src)
root = Path.cwd()
venv_dir = root / "ATOMOD" / "venv" 
src_dir  = root / "ATOMOD"

url_depot = "https://github.com/hbulou/ATOMOD.git"
dossier_cible = Path("./ATOMOD")

# 🛡️ Sécurité : On vérifie si le dossier n'existe pas déjà
if not dossier_cible.exists():
    print(f"📥 Clonage du dépôt {url_depot}...")
    
    try:
        # Exécute la commande sous forme de liste pour éviter les failles d'injection
        # check=True force Python à lever une erreur si le clone échoue (ex: pas d'internet)
        subprocess.run(["git", "clone", url_depot], check=True)
        print("✅ Dépôt cloné avec succès !")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du git clone : {e}")
    except FileNotFoundError:
        print("❌ Erreur : La commande 'git' n'est pas installée ou accessible sur ce système.")
else:
    print(f"ℹ️ Le dossier '{dossier_cible}' existe déjà. Clonage annulé.")



    
# 2. Création de l'environnement virtuel
if not venv_dir.exists():
    print(f"Création de l'environnement virtuel dans : {venv_dir}")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
else:
    print("L'environnement virtuel existe déjà.")



# 1. Définir le chemin absolu vers le pip de ton venv
pip_du_venv = venv_dir/"bin/pip"

# 2. Installer un ou plusieurs packages spécifiques
print("📦 Installation des packages en cours...")
subprocess.run([pip_du_venv, "install", "--no-cache-dir", "--upgrade", "pip"])
#subprocess.run([pip_du_venv, "install", "--no-cache-dir", "jupyterlab", "numpy"])

# 4. Installation des paquets requis
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
]
for pack in packages:
    subprocess.run([pip_du_venv, "install", "--no-cache-dir", pack])

print("✅ Installation terminée !")
exit()

# Détection de l'exécutable Python selon l'OS (Windows vs Linux/WSL)
if os.name == "nt":  # Windows
    venv_python = venv_dir / "Scripts" / "python.exe"
else:  # Linux / WSL / Mac
    venv_python = venv_dir / "bin" / "python"

print("Démarrage de l'installation d'ATOMOD...")


# 3. Mise à jour de PIP dans l'environnement virtuel
print(" Mise à jour de pip...")
subprocess.run(
    [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True
)


print(" Installation des bibliothèques Python (cette étape peut prendre du temps)...")
subprocess.run([str(venv_python), "-m", "pip", "install"] + packages, check=True)




# 5. Création du dossier ~/src
src_dir.mkdir(parents=True, exist_ok=True)


