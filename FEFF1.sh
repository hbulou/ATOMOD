#!/bin/bash
#Cluster HPC

#SBATCH -p grant -A g2025a370c
#SBATCH -N 1                # nombre de noeuds alloués. Par  ex -N 1-4 permet d'allouer de 1 à 4 noeuds (1 à 4 machines physiques)
#SBATCH -n 1                # equivalent à #SBATCH --ntasks=4 : Spécifie le nombre total de tâches parallèles que SLURM doit lancer.
#SBATCH -t 00:30:00     

cd

module load python

#python3 -m venv venv/ATOMOD

source venv/ATOMOD_gpu/bin/activate

cd workdir/ATOMOD

python3 Stand_Alone_FEFF_simulation.py
