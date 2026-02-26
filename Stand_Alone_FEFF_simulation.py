import sys
from pathlib import Path

from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d


sys.path.append('/home/bulou/src/lib/site-packages/')
from HBPy.Crystal import Crystal
#from HBPy.ForceField import ForceField

from PyFEFF.FEFF import FEFF


class EXAFS:
    def __init__(self):
        self.energy=[]
        self.chi=[]
        self.FEFF=FEFF()

# ============================================================================================
# TOOLS
# ============================================================================================
def mk_mean(series_list):
    """
    Méthode  : Interpolation sur une grille commune.
    
    Stratégie :
    1. Trouver la plage d'énergie commune à toutes les séries
    2. Créer une grille uniforme sur cette plage
    3. Interpoler chaque série sur cette grille
    4. Moyenner
    
    Args:
        series_list: Liste de tuples (energy, intensity)
    
    Returns:
        energy_common, intensity_mean, intensity_std
    """
    # Trouver la plage commune (intersection de toutes les séries)
    energy_min = max(serie[0].min() for serie in series_list)
    energy_max = min(serie[0].max() for serie in series_list)
    
    print(f"Plage commune: [{energy_min:.2f}, {energy_max:.2f}]")
    
    # Créer une grille uniforme
    n_points = len(series_list[0][0])  # Utilise le nombre de points de la première série
    energy_common = np.linspace(energy_min, energy_max, n_points)
    
    # Interpoler chaque série sur la grille commune
    interpolated_intensities = []
    
    for energy, intensity in series_list:
        # Interpolation linéaire (ou 'cubic' pour plus de lissage)
        f = interp1d(energy, intensity, kind='linear', fill_value='extrapolate')
        intensity_interp = f(energy_common)
        interpolated_intensities.append(intensity_interp)
    
    # Convertir en array pour calculs vectorisés
    interpolated_intensities = np.array(interpolated_intensities)
    
    # Calculer moyenne et écart-type
    intensity_mean = np.mean(interpolated_intensities, axis=0)
    intensity_std = np.std(interpolated_intensities, axis=0)
    
    return energy_common, intensity_mean, intensity_std
def mk_mean2(series_list):
    """
    Méthode  : Interpolation sur une grille commune.
    
    Stratégie :
    1. Trouver la plage d'énergie commune à toutes les séries
    2. Créer une grille uniforme sur cette plage
    3. Interpoler chaque série sur cette grille
    4. Moyenner
    
    Args:
        series_list: Liste de tuples (energy, intensity)
    
    Returns:
        energy_common, intensity_mean, intensity_std
    """
    # Trouver la plage commune (intersection de toutes les séries)
    energy_min = max(serie[0].min() for serie in series_list)
    energy_max = min(serie[0].max() for serie in series_list)
    
    print(f"Plage commune: [{energy_min:.2f}, {energy_max:.2f}]")
    
    # Créer une grille uniforme
    n_points = len(series_list[0][0])  # Utilise le nombre de points de la première série
    energy_common = np.linspace(energy_min, energy_max, n_points)
    
    # Interpoler chaque série sur la grille commune
    interpolated_f1 = []
    interpolated_f2 = []
    interpolated_f3 = []
    interpolated_f4 = []
    interpolated_f5 = []
    
    for energy, f1,f2,f3,f4,f5 in series_list:
        # Interpolation linéaire (ou 'cubic' pour plus de lissage)
        f1 = interp1d(energy, f1, kind='linear', fill_value='extrapolate')
        f1_interp = f1(energy_common)
        interpolated_f1.append(f1_interp)
        
        f2 = interp1d(energy, f2, kind='linear', fill_value='extrapolate')
        f2_interp = f2(energy_common)
        interpolated_f2.append(f2_interp)

        f3 = interp1d(energy, f3, kind='linear', fill_value='extrapolate')
        f3_interp = f3(energy_common)
        interpolated_f3.append(f3_interp)

        f4 = interp1d(energy, f4, kind='linear', fill_value='extrapolate')
        f4_interp = f4(energy_common)
        interpolated_f4.append(f4_interp)

        f5 = interp1d(energy, f5, kind='linear', fill_value='extrapolate')
        f5_interp = f5(energy_common)
        interpolated_f5.append(f5_interp)

    # Convertir en array pour calculs vectorisés
    interpolated_f1 = np.array(interpolated_f1)
    interpolated_f2 = np.array(interpolated_f2)
    interpolated_f3 = np.array(interpolated_f3)
    interpolated_f4 = np.array(interpolated_f4)
    interpolated_f5 = np.array(interpolated_f5)
    
    # Calculer moyenne et écart-type
    f1_mean = np.mean(interpolated_f1, axis=0)
    f1_std = np.std(interpolated_f1, axis=0)
    
    f2_mean = np.mean(interpolated_f2, axis=0)
    f2_std = np.std(interpolated_f2, axis=0)

    f3_mean = np.mean(interpolated_f3, axis=0)
    f3_std = np.std(interpolated_f3, axis=0)
    
    f4_mean = np.mean(interpolated_f4, axis=0)
    f4_std = np.std(interpolated_f4, axis=0)
    
    f5_mean = np.mean(interpolated_f5, axis=0)
    f5_std = np.std(interpolated_f5, axis=0)

    return energy_common, f1_mean, f2_mean, f3_mean, f4_mean, f5_mean
# ##########################################################################################
def main():
    config={
        'xyzfile': "data/NP/RuRhPdIrPt_wulff807D_eq_10K_299.xyz",  #"./GUI/NP.xyz",
        'absorbers': "all",
        'rpath':5.0,
        'edge':{'Ru':'K','Rh':'K','Pd':'K','Ir':'L3','Pt':'L3'}
        }
    molecule=Crystal()
    molecule.load_file(config['xyzfile'])
    molecule.MassCenter()
    molecule.get_element_distribution()
    molecule.get_structure()
    list_exafs=[]
    # Initialise automatiquement avec une liste vide
    
    if 'all' in config['absorbers']:
        list_absorbers=np.arange(len(molecule.atoms))
    if '-' in config['absorbers']:
        list_absorbers=np.arange(int(config['absorbers'].split("-")[0]),
                                 int(config['absorbers'].split("-")[1]))
    for idx in list_absorbers:
        elt=molecule.atoms[idx].elt
        print(f"absorber {idx}: {elt}")
        filename=f"feff_{elt}_{idx}.inp"
        list_exafs.append(EXAFS())
        list_exafs[-1].FEFF.config['TITLE']= f"Atome absorbeur : {elt}(idx={idx})"
        list_exafs[-1].FEFF.config['EDGE']=config['edge'][elt]
        list_exafs[-1].FEFF.create_input_file(molecule=molecule,
                                              absorber_idx=idx,
                                              filename=filename)

        list_exafs[-1].FEFF.run(filename,config,list_pgm=['rdinp',
                                                          'atomic',
                                                          'dmdw',
                                                          'pot',
                                                          'xsph',
                                                          'path',
                                                          'genfmt',
                                                          'ff2x',
                                                          'sfconv',
                                                          'compton'])
        Path("xmu.dat").rename(Path(f"xmu_{elt}_{idx}.dat"))


    E0=defaultdict(list)
    series = defaultdict(list)
    for idx in list_absorbers:
        elt=molecule.atoms[idx].elt
        omega,e,k,mu,mu0,chi = np.loadtxt(f"xmu_{elt}_{idx}.dat", comments='#', usecols=(0,1,2,3,4,5), unpack=True)
        series[elt].append((e,omega,k,mu,mu0,chi))
        E0[elt].append(omega[0])

    for i,elt in enumerate(series.keys()):
        e,omega,k,mu,mu0,chi=mk_mean2(series[elt])
        np.savetxt(
            f'xmu_{elt}_mean.dat',               # Le nom du fichier
            np.column_stack((omega,e,k,mu,mu0,chi)),        # Le tableau 2D créé juste au-dessus
            fmt='%.6f',               # Le format (ici : 6 chiffres après la virgule)
            delimiter='    ',         # Le séparateur entre les colonnes (ici : 4 espaces)
            header='#omega e k mu mu0 chi', # (Optionnel) Ajoute un en-tête
            comments='# '             # (Optionnel) Le caractère pour commenter l'en-tête
        )
    for elt in E0.keys():
        print(f"{elt} omega0={E0[elt]}")

# ##########################################################################################
# Point d’entrée du programme
if __name__ == "__main__":
    main()
