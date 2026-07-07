import os
import sys
import subprocess
from pathlib import Path

# Définition des chemins de manière portable avec pathlib
root = Path.cwd()
tuto_dir = root / "doc" / "tutorials"

# 1. Détection du système d'exploitation
if sys.platform.startswith("win"):
    # Configuration pour Windows
    print("[ATOMOD] Système détecté : Windows")
    python_executable = root / "venv" / "ATOMOD" / "Scripts" / "python.exe"
else:
    # Configuration pour Linux / MacOS
    print("[ATOMOD] Système détecté : Linux / Unix")
    python_executable = root / "venv" / "ATOMOD" / "bin" / "python"

# 2. Vérifications de sécurité
if not python_executable.exists():
    sys.exit(f"❌ Erreur : L'environnement virtuel est introuvable à l'emplacement : {python_executable}\n"
             "Veuillez relancer l'installation.")

if not tuto_dir.exists():
    sys.exit(f"❌ Erreur : Le dossier des tutoriels est introuvable : {tuto_dir}")

# 3. Changement de répertoire vers les tutoriels
os.chdir(tuto_dir)

# 4. Lancement de Jupyter Lab en utilisant le Python de l'environnement virtuel
# Utiliser "python -m jupyterlab" évite d'avoir à appeler le script d'activation de l'environnement
print(f"[ATOMOD] Lancement de Jupyter Lab depuis {tuto_dir}...")
try:
    subprocess.run([str(python_executable), "-m", "jupyterlab"], check=True)
except KeyboardInterrupt:
    print("\n👋 Jupyter Lab arrêté proprement.")
except Exception as e:
    print(f"❌ Erreur lors du lancement de Jupyter Lab : {e}")
