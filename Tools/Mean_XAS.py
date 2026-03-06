import argparse
import numpy as np
import glob
import re
from scipy.interpolate import interp1d
from pathlib import Path

def mk_mean_exafs(series_list):
    """
    Moyenne une liste de spectres en les interpolant sur une grille commune.
    Inspiré de mk_mean2 du projet M2P2_HEA.
    """
    # 1. Déterminer la plage d'énergie (omega) commune (intersection)
    # Dans xmu.dat : col 0 = omega (énergie absolue)
    omega_min = max(s[:, 0].min() for s in series_list)
    omega_max = min(s[:, 0].max() for s in series_list)
    
    # Créer une grille uniforme (on prend le nb de points du premier fichier)
    n_points = len(series_list[0])
    omega_common = np.linspace(omega_min, omega_max, n_points)
    
    # 2. Préparer les listes pour chaque colonne à moyenner (e, k, mu, mu0, chi)
    # colonnes 1 à 5 de xmu.dat
    interpolated_data = [[] for _ in range(5)] 
    
    for data in series_list:
        omega_orig = data[:, 0]
        for col_idx in range(1, 6):
            f = interp1d(omega_orig, data[:, col_idx], kind='linear', fill_value='extrapolate')
            interpolated_data[col_idx-1].append(f(omega_common))
    
    # 3. Calculer les moyennes
    means = [np.mean(np.array(col_list), axis=0) for col_list in interpolated_data]
    
    return omega_common, *means

def main():
    # Pattern pour trouver les fichiers : xmu_ELEMENT_INDEX.dat
    file_pattern = "xmu_Ru_*.dat"
    files = glob.glob(file_pattern)
    
    if not files:
        print("Aucun fichier trouvé avec le pattern xmu_elt_idx.dat")
        return

    # Regrouper les fichiers par élément (Rh, Ir, etc.)
    elements = {}
    for f in files:
        # Extrait l'élément entre 'xmu_' et le dernier '_'
        match = re.search(r'xmu_([a-zA-Z]+)_\d+\.dat', f)
        if match:
            elt = match.group(1)
            if elt not in elements:
                elements[elt] = []
            elements[elt].append(f)

    # Traiter chaque élément
    for elt, file_list in elements.items():
        print(f"Traitement de l'élément {elt} ({len(file_list)} fichiers)...")
        
        all_data = []
        for f in file_list:
            # Charge les 6 colonnes standard de xmu.dat
            try:
                data = np.loadtxt(f, comments='#', usecols=(0, 1, 2, 3, 4, 5))
                all_data.append(data)
            except Exception as e:
                print(f"Erreur lors de la lecture de {f}: {e}")

        if all_data:
            # Calcul de la moyenne
            omega, e, k, mu, mu0, chi = mk_mean_exafs(all_data)
            
            # Sauvegarde du fichier moyen
            output_name = f"xmu_{elt}_mean.dat"
            header = f"Moyenne de {len(file_list)} spectres pour {elt}\nomega          e          k          mu          mu0          chi"
            
            np.savetxt(
                output_name,
                np.column_stack((omega, e, k, mu, mu0, chi)),
                fmt='%.6f',
                delimiter='    ',
                header=header,
                comments='# '
            )
            print(f"✅ Fichier moyen enregistré : {output_name}")

if __name__ == "__main__":
    main()

