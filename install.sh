cd
python3 -m venv venv/ATOMOD   # création de l'environnement - répertoire ~/venv/ATOMOD
source venv/ATOMOD/bin/activate  # activation de l'environnement
pip install --upgrade pip                   # màj de pip


pip install jupyterlab
pip install numpy
pip install "scipy<1.17"
pip install matplotlib
pip install mace-torch
pip install dask            #  Dask est une bibliothèque de calcul parallèle et de gestion des grandes masses de données (Big Data) en Python.
pip install tabulate
pip install numba
pip install threadpoolctl
pip install zarr
pip install ipywidgets
pip install pyfftw

cd ; mkdir -p src ; cd src
wget https://github.com/hbulou/site-packages/archive/refs/heads/main.zip -O HBPy.zip
unzip HBPy.zip ; rm HBPy.zip ; mv site-packages-main HBPy
cd HBPy ; pip install -e .
cd ..
wget https://github.com/hbulou/ATOMOD/archive/refs/heads/main.zip -O ATOMOD-main.zip
unzip ATOMOD-main.zip ; rm ATOMOD-main.zip ; mv ATOMOD-main ATOMOD
