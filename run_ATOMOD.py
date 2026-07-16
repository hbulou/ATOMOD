import os
import argparse

from pathlib import Path
# Configuration du logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Exécution de la détection automatique
#auto_detect_gpu_compatibility()

# --- Reste de ton code (main, train, etc.) ---

# On traverse le dossier ATOMOD, on ouvre le fichier ATOMOD, et on importe la fonction
#from ATOMOD.data_generation import mk_in_silico_data



def parse_args():
    parser = argparse.ArgumentParser(description="Génération de données in silico ATOMOD")
    parser.add_argument('--seed_start', type=int, default=0,
                         help="Valeur de départ pour la plage de seeds (incluse)")
    parser.add_argument('--seed_end', type=int, default=1,
                         help="Valeur de fin pour la plage de seeds (exclue)")
    parser.add_argument('--radius', type=float, default=5.0,
                         help="Rayon, en angstroem, de la nanoparticule")
    parser.add_argument('--mk_in_silico_data', action='store_true',
                        help="Générer des data in silico")
    parser.add_argument('--alloy_stability', action='store_true',
                        help="Stabilité des alliages")
    parser.add_argument('--clustering', action='store_true',
                        help="Clustering")
    parser.add_argument('--exafs_model', action='store_true',
                        help="EXAFS model training")
    return parser.parse_args()


def main():

    args = parse_args()
    
    gen_data=True
    train_model=False

    status={
        'NP':True,
        'abtem':True,
        'feff':True,
        'atomic probability map':True,
        'optimization':False,
    }
    
    config={
        'run_dir':'run_dir',                          # répertoire de lancement
        'simul_dir':'simul',                          # répertoire de base de la simulation
        'train':{
            'TEM_img_dir'      : "train/TEM",        # répertoire de stockage des images TEM
            'EXAFS_dir'        : "train/EXAFS",      # répertoire de stockage des spectres EXAFS
            'prob_maps_img_dir': "train/prob_maps",  # répertoire de stockage des images TEM
            'nfo_dir'          : "train/nfo",        # répertoire des infos sur la gen. in silico
            'optimizer'        : "adam",
            'BATCH_SIZE'       :  4,
        },
        'NP':{
            'status':status['NP'],
            'seed':1,
            'structure':{
                'optimization':status['optimization'],
                'composition':['Pt','Co','Au','Pd','Rh'],
                'radius':args.radius,
                'a':3.92,
            },
            'nvaccum':2.0,
        },
        'Alloy stability':{
            'MC':{
                'nstep':200,
                'seed':2,
            },
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
            'status':status['atomic probability map'],
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
            'ymax':0.0,
            'H':128,
            'W':128,
            },
        'exafs':{
            'N_POINTS_EXAFS': 200,   # Nombre de points par spectre
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
                'feff_exec_dir': Path('/home/bulou/ownCloud/Notebooks/M2P2_HEA/Home/Modelisation/ATOMOD/JFEFF/feff90/unix/'),
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
    

    config['run_dir']=Path.cwd()
    config['feff']['parameters']['feff_dir']=config['run_dir']/'JFEFF/feff90/unix/'

    if args.alloy_stability:
        from ATOMOD.data_generation import alloy_stability
        config['simul_dir']='Alloy_Stability3'
        alloy_stability(config)
    if args.mk_in_silico_data:
        logger.info(f'{20*"#"} Make In Silico Data {20*"#"}')
        from ATOMOD.data_generation import mk_in_silico_data_v2
        config['simul_dir']=Path('simulv2')
        savedir=config['run_dir']/config['simul_dir']
        if not savedir.exists():
            print(f"❌ Le répertoire n'existe pas")
            # Créer le répertoire
            savedir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Répertoire créé: {savedir}")
            idx=0
        else:
            print(f"✅ Le répertoire existe")
            # Lister les répertoires qu'il contient
            subdirs = [d.name for d in savedir.iterdir() if d.is_dir()]
            print(f"\nRépertoires contenus ({len(subdirs)}):")
            for subdir in sorted(subdirs):
                print(f"  - {subdir}")
            idx=len(subdirs)
        for seed in range(args.seed_start, args.seed_end):
            config['NP']['seed']=seed
            mk_in_silico_data_v2(config,idx)
            idx+=1
    if args.clustering:
        from ATOMOD.MachineLearning import clustering
        clustering(config)
    if args.exafs_model:
        logger.info(f'{20*"#"} EXAFS Modeling {20*"#"}')
        from ATOMOD.XAS.ML import EXAFS_model
        EXAFS_model(config)
# #########################################################################################
if __name__ == "__main__":
    main()


    # J'ai une méthode mk_in_silico_data() définie dans un fichier ATOMOD.py qui est dans le répertoire /home/bulou/src/ATOMOD/ATOMOD.  je veux appeler cette méthode dans le script Stand_Alone_ATOMOD.py qui est dans le répertoire /home/bulou/src/ATOMOD. Que me conseilles tu ?
