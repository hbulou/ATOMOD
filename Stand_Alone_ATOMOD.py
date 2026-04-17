import sys
#import os
import random



import HBPy
from HBPy.Molecule.Crystal import Crystal,Atom

import matplotlib.pyplot as plt
sys.path.append('./lib/')
import abtem

import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # _______________________________________
    #
    # Etape 1 : construire la nanoparticules
    # _______________________________________
    #   Etape 1.1 : la structure
    NP=Crystal()
    NP.build(radius=6.0,materials='NP')
    NP.origin_at_mass_center()
    logger.info(f"min={NP.qmin} max={NP.qmax}")
    logger.info(f"Mass center={NP.MC}")
    logger.info(f"Number of atoms={len(NP.atoms)}")
    #   Etape 1.2 : la distribution chimique
    composition=['Pt','Ni','Ir']
    NP.set_composition(composition)
    NP.save(prefix="NP",fmt='xyz')
    #   Etape 1.3 : (optionnelle) l'optimisation structurale et/ou chimique

    # _______________________________________
    #
    # Etape 2 : construire les cartes de probabilité de présence atomique
    # _______________________________________
    config={
        'train':{
            'TEM_img_dir':"data/train/images",  # répertoire de stockage des images TEM
            'prob_maps_img_dir':"data/train/prob_maps"  # répertoire de stockage des images TEM
        },
        'abtem':{
            'dx':0.04,
            'dy':0.04,
            'dz':4.08/2,
            'energy':300e3,
            'focal spread':40,
            'semiangle cutoff':20,
            'defocus':200,
            'cell scale':1.1
            }
    }


    NP.xyz2slice(config)

    # _______________________________________
    # Etape 3 : construire l'image TEM
    #   abTEM : https://github.com/abTEM/abTEM
    #   https://abtem.readthedocs.io/en/latest/intro.html#
    #NP.abTEM(config)
    # _______________________________________
    
if __name__ == "__main__":
    main()
