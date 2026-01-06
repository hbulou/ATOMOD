# Projet ATOMOD

**sur hpc**

cd

module load python

python3 -m venv venv/ATOMOD

source venv/ATOMOD/bin/activate

cd workdir/ATOMOD

pip install torch

pip install tensorflow[and-cuda]

pip install opencv-python
