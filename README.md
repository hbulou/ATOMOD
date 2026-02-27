# Projet ATOMOD

## Installation
* choisir un répertoire où stocker ATOMOD et s'y placer
* cloner ATOMOD
```sh
git clone git@github.com:hbulou/ATOMOD.git
```
* en principe un répertoire ATOMOD a été créé. Aller dans ATOMOD
```sh
cd HBPy
```
* charger l'environnement ATOMOD. 
```sh
source ~/venv/ATOMOD/bin/activate
# ou
source ~/venv/ATOMOD_gpu/bin/activate
```
RQ : sur les serveurs de calcul, il peut être nécessaire de charger le module python au préalable
```sh
module load python
```

-------------------
**sur hpc**
```bash
cd
module load python
python3 -m venv venv/ATOMOD
source venv/ATOMOD/bin/activate
cd workdir/ATOMOD
pip install torch
pip install tensorflow[and-cuda]
pip install opencv-python
```

```bash
cd /home/bulou/ownCloud/Notebooks/M2P2_HEA/Modelisation/ATOMOD/GUI
source /home/bulou/venv/ATOMOD/bin/activate
python Py_ATOMODv0.5.py
```

```bash
pyuic6 HB_ATOMOD_GUI.ui -o HB_ATOMOD_GUI.py
```
