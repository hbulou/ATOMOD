#!/bin/bash
#Cluster HPC

#SBATCH -p grant -A g2025a370c
#SBATCH -N 1                # nombre de noeuds alloués. Par  ex -N 1-4 permet d'allouer de 1 à 4 noeuds (1 à 4 machines physiques)
#SBATCH -n 4                # equivalent à #SBATCH --ntasks=4 : Spécifie le nombre total de tâches parallèles que SLURM doit lancer.
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
python gen_data.py

