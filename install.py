import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

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

pip_du_venv = venv_dir/"bin/pip"
# pip update
subprocess.run([pip_du_venv, "install", "--no-cache-dir", "--upgrade", "pip"])

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
]
for pack in packages:
    subprocess.run([pip_du_venv, "install", "--no-cache-dir", pack])

print("✅ Installation complete!")


