#!/bin/bash
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpudp
#SBATCH -t 1-00:00:00

module load python

# On active simplement l'environnement créé précédemment
cd
source venv/ATOMOD_gpu/bin/activate
cd workdir/ATOMOD

echo "Environnement chargé : $VIRTUAL_ENV"

# Force TF à ne pas tester tous les algos (évite le crash 5003)
export TF_CUDNN_USE_AUTOTUNE=0

# Gestion dynamique de la mémoire (évite les conflits d'allocation)
export TF_FORCE_GPU_ALLOW_GROWTH=true
python Stand_Alone_ATOMOD.py
