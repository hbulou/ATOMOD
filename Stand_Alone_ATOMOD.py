import sys
import os
import random



import HBPy
from HBPy.Molecule.Crystal import Crystal,Atom

from PyFEFF.FEFF import (FEFF)

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
    config={
        'root_dir':'simul',
        'train':{
            'TEM_img_dir':"train/images",  # répertoire de stockage des images TEM
            'prob_maps_img_dir':"train/prob_maps"  # répertoire de stockage des images TEM
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
            },
        'atomic presence probability map':{
            'ninter':{ # nombre d'intervalles entre deux positions atomiques
                'x':20,
                'y':20,
                'z':2
            },
            'sigma': .6  # en Å, largeur de la gaussienne ~ rayon atomique ou un peu moins
        },
        'image':{
            'xmin':0.0,
            'xmax':0.0,
            'ymin':0.0,
            'ymax':0.0
            },
        'nvaccum':2.0,
        'structure':{
            'optimization':False,
            'composition':['Pt','Co'],
            'radius':5.0,
            'a':0.5*(3.55+3.92),
        },
        'feff':{
            'TITLE':'FEFF INPUT FILE',
            'DEBYE_TEMP': 190.0,
            'SCF_RADIUS': 5.0,
            'RPATH': 5.0,     # typique 2.2xdistance plus proches voisins. changer pour étudier la cvg des spectres
            'EXAFS' : 20.0,   # xkmax - default 20 ang.^-1
            'EDGE':  "K",
            'RMAX':8.0,
            'feff_dir':  '/home/bulou/ownCloud/Notebooks/M2P2_HEA/Home/Modelisation/ATOMOD/JFEFF/feff90/unix/',
            'input_save_dir':'./',
            'filename':'feff.inp',
            'list_pgm':['rdinp','atomic','dmdw','pot',
                        'opconsat', 
                        'screen',
                        'xsph',
                        'fms',
                        'mkgtr',
                        'path', 
                        'genfmt',
                        'ff2x',
                        'sfconv',
                        'compton',
                        'eels',
                        'ldos'
                        ]
        }
        
    }
    # _______________________________________
    #
    # Etape 1 : construire la nanoparticules
    # _______________________________________
    #   Etape 1.1 : la structure
    NP=Crystal()
    NP.build(a=config['structure']['a'],
             radius=config['structure']['radius'],
             materials='NP')
    NP.origin_at_mass_center()
    logger.info(f"min={NP.qmin} max={NP.qmax}")
    logger.info(f"Mass center={NP.MC}")
    logger.info(f"Number of atoms={len(NP.atoms)}")
    #   Etape 1.2 : la distribution chimique

    NP.set_composition(config['structure']['composition'])
    NP.save(prefix="NP",fmt='xyz')
    #   Etape 1.3 : (optionnelle) l'optimisation structurale et/ou chimique
    if config['structure']['optimization']:
        NP.optimize_ase()

    # ___________________________________________________________________
    #
    # Etape 2 : construire les cartes de probabilité de présence atomique
    # ___________________________________________________________________
    NP.xyz2slice(config)
    logger.info(f"{NP.qmin[0]} {NP.qmax[0]} {NP.qmin[1]} {NP.qmax[1]}")
    logger.info(f"{config['image']['xmin']} {config['image']['xmax']} {config['image']['ymin']} {config['image']['ymax']}")
    
    # ____________________________________________________
    # Etape 3 : construire l'image TEM
    #   abTEM : https://github.com/abTEM/abTEM
    #   https://abtem.readthedocs.io/en/latest/intro.html#
    # ____________________________________________________
    NP.abTEM(config)

    # ____________________________________________________
    # Etape 4 : construire les spectres EXAFS
    # ____________________________________________________
    NP.feff=FEFF()
    base_dir=os.getcwd()

    for atm in NP.atoms:
        config['feff']['input_save_dir']=f"{config['root_dir']}/feff_input_files/{atm.elt}_{atm.idx}"
        os.makedirs(config['feff']['input_save_dir'], exist_ok=True)
        os.chdir(f"{base_dir}/{config['feff']['input_save_dir']}")
        NP.FEFF_create_input_file(config['feff'],absorber_idx=atm.idx)
        NP.FEFF_run(config['feff'])
        os.chdir(base_dir)

    
if __name__ == "__main__":
    main()
