import sys
import os
import random
import numpy
from collections import defaultdict

import HBPy
from HBPy.Molecule.Crystal import Crystal,Atom

import matplotlib.pyplot as plt
sys.path.append('./lib/')
import abtem
from pathlib import Path
import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ##################################################################################
def mk_in_silico_data():
    status={
        'NP':True,
        'abtem':True,
        'feff':True,
        'atomic probability map':True,
        'optimization':False,
    }
    config={
        'root_dir':'simul',
        'train':{
            'TEM_img_dir'      : "train/TEM",        # répertoire de stockage des images TEM
            'EXAFS_dir'        : "train/EXAFS",      # répertoire de stockage des spectres EXAFS
            'prob_maps_img_dir': "train/prob_maps",  # répertoire de stockage des images TEM
            'nfo_dir'          : "train/nfo",  # répertoire de stockage des images TEM
        },
        'abtem':{
            'status':status['abtem'],
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
        'atomic probability map':{
            'status':status['atomic probability map']
        },
        'NP':{
            'status':status['NP'],
            'seed':2,
            'structure':{
                'optimization':status['optimization'],
                #  'composition':['Pt','Co','Au'],
                # 'radius':3.5,
                # 'a':0.5*(3.55+3.92),
                'composition':['Pt','Co','Au','Pd','Rh'],
                'radius':9.0,
                'a':3.92,
            },
            'nvaccum':2.0,
        },
        'image':{
            'xmin':0.0,
            'xmax':0.0,
            'ymin':0.0,
            'ymax':0.0
            },
        'feff':{
            'status':status['feff'],
            'parameters':{
                'TITLE':'FEFF INPUT FILE',
                'DEBYE_TEMP': 190.0,
                'SCF_RADIUS': 5.0,
                'RPATH': 5.0,     # typique 2.2xdistance plus proches voisins. changer pour étudier la cvg des spectres
                'EXAFS' : 20.0,   # xkmax - default 20 ang.^-1
                'EDGE': {'Co':'K','Ni':'K','Ru':'K','Rh':'K','Pd':'K','Ir':'L3','Pt':'L3','Au':'L3'},
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
    }
    ##############################################################################################################################"
    config['root_dir']=f"{config['root_dir']}/{config['NP']['seed']}"
    # _______________________________________
    #
    # Etape 1 : construire la nanoparticules
    # _______________________________________
    #   Etape 1.1 : la structure
    if config['NP']['status']:
        NP=Crystal()
        NP.build(a=config['NP']['structure']['a'],
                 radius=config['NP']['structure']['radius'],
                 materials='NP')
        NP.origin_at_mass_center()
        logger.info(f"min={NP.qmin} max={NP.qmax}")
        logger.info(f"Mass center={NP.MC}")
        logger.info(f"Number of atoms={len(NP.atoms)}")
        #   Etape 1.2 : la distribution chimique

        NP.set_composition(config['NP']['structure']['composition'],seed=config['NP']['seed'])
        directory=Path.cwd()/config['root_dir']/config['train']['nfo_dir']/"XYZ"
        directory.mkdir(parents=True, exist_ok=True)

        NP.save(prefix="NP",fmt='xyz',directory=directory)
    
        #   Etape 1.3 : (optionnelle) l'optimisation structurale et/ou chimique
        if config['NP']['structure']['optimization']:
            NP.optimize_ase()

    # ___________________________________________________________________
    #
    # Etape 2 : construire les cartes de probabilité de présence atomique
    # ___________________________________________________________________
    if config['atomic probability map']['status']:
        NP.xyz2slice(config)
        logger.info(f"{NP.qmin[0]} {NP.qmax[0]} {NP.qmin[1]} {NP.qmax[1]}")
        logger.info(f"{config['image']['xmin']} {config['image']['xmax']} {config['image']['ymin']} {config['image']['ymax']}")
    
    # ___________________________________________________________________
    # Etape 3 : construire l'image TEM
    #   abTEM : https://github.com/abTEM/abTEM
    #   https://abtem.readthedocs.io/en/latest/intro.html#
    # ___________________________________________________________________
    if config['abtem']['status']:
        NP.abTEM(config)

    # ____________________________________________________
    # Etape 4 : construire les spectres EXAFS
    # ____________________________________________________
    feff_dir=Path.cwd()/config['root_dir']/config['train']['nfo_dir']/"feff_input_files"
    if config['feff']['status']:
        feff_dir.mkdir(parents=True, exist_ok=True)
        base_dir=os.getcwd()
        if config['feff']['status']:
            for atm in NP.atoms:
                config['feff']['parameters']['input_save_dir']=f"{feff_dir}/{atm.elt}_{atm.idx}"
                os.makedirs(config['feff']['parameters']['input_save_dir'], exist_ok=True)
                
                os.chdir(f"{config['feff']['parameters']['input_save_dir']}")
                NP.FEFF_create_input_file(config['feff']['parameters'],absorber_idx=atm.idx)
                NP.FEFF_run(config['feff']['parameters'])
                os.chdir(base_dir)
    
    # Initialise automatiquement avec une liste vide
    series = defaultdict(list)
    logger.info(NP.list_elt)
    base_dir=os.getcwd()
    logger.info(f"{base_dir}")
    for atm in NP.atoms:
        xmu_dir=f"{feff_dir}/{atm.elt}_{atm.idx}"
        try:
            k, chi = numpy.loadtxt(f"{xmu_dir}/xmu.dat", comments='#', usecols=(2, 5), unpack=True)
            series[atm.elt].append((k,chi))
        except:
            logger.error(f"file {xmu_dir}/xmu.dat not found!")
    exafs_dir=Path.cwd()/config['root_dir']/config['train']['EXAFS_dir']
    exafs_dir.mkdir(parents=True, exist_ok=True)


    for kexpo in range(4):
        if kexpo==0:
            ylbl=r"$\chi(E)$"
            savename="chi(k)"
            header="# k (A^-1)   chi(k)"
        elif kexpo==1:
            ylbl=r"$k\cdot\chi(E)$"
            savename="kchi(k)"
            header="# k (A^-1)   k.chi(k)"
        elif kexpo==2:
            ylbl=r"$k^2\cdot\chi(E)$"
            savename="k2chi(k)"
            header="# k (A^-1)   k^2.chi(k)"
        elif kexpo==3:
            ylbl=r"$k^3\cdot\chi(E)$"
            savename="k3chi(k)"
            header="# k (A^-1)   k^3.chi(k)"

        for elt in NP.list_elt:
            logger.info(f"{elt} : {len(series[elt])}")
            
            k,chiglob,dev=HBPy.Molecule.Tools.mk_mean(series[elt],expo=kexpo)
            logger.info(f"Saving {exafs_dir}/{savename}_{elt}.dat")
            numpy.savetxt(
                f"{exafs_dir}/{savename}_{elt}.dat",                       # Le nom du fichier
                numpy.column_stack((k,chiglob)),               # Le tableau 2D créé juste au-dessus
                fmt='%.6f',                                         # Le format (ici : 6 chiffres après la virgule)
                delimiter='    ',                                   # Le séparateur entre les colonnes (ici : 4 espaces)
                header=header,
                comments='# '                                       # (Optionnel) Le caractère pour commenter l'en-tête
            )
# #########################################################################################
if __name__ == "__main__":
    mk_in_silico_data()

