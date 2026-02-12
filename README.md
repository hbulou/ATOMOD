# Projet ATOMOD

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
