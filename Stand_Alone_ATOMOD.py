import os
import tensorflow as tf


# Configuration du logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def auto_detect_gpu_compatibility():
    """
    Détecte automatiquement si le GPU est obsolète pour TensorFlow (Compute Capability < 6.0)
    ou s'il n'est pas disponible, et bascule sur le CPU le cas échéant.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        logger.info("[ATOMOD] Aucun GPU physique détecté. Mode CPU par défaut.")
        return

    try:
        # Récupère les détails matériels du premier GPU
        details = tf.config.experimental.get_device_details(gpus[0])
        compute_cap = details.get('compute_capability', (0, 0))
        gpu_name = details.get('device_name', 'GPU Inconnu')
        
        logger.info(f"[ATOMOD] GPU détecté : {gpu_name} (Compute Capability: {compute_cap[0]}.{compute_cap[1]})")
        
        # Les versions récentes de TensorFlow ont de graves problèmes de compilation
        # JIT en dessous de la version 6.0 (Pascal). Ta Quadro M1000M est en 5.0.
        if compute_cap[0] < 6:
            logger.info(f"[ATOMOD] ATTENTION : L'architecture de votre {gpu_name} est trop ancienne (CC < 6.0).")
            logger.info("[ATOMOD] Désactivation automatique du GPU pour éviter un crash 'CUDA_ERROR_NO_BINARY_FOR_GPU'.")
            
            # Action corrective : On masque le GPU pour le processus courant
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            # On force TensorFlow à réévaluer les périphériques disponibles immédiatement
            tf.config.set_visible_devices([], 'GPU')
            logger.info("[ATOMOD] Bascule sur le CPU réussie.")
            
    except Exception as e:
        logger.info(f"[ATOMOD] Erreur lors de l'analyse du GPU ({e}). Par sécurité, bascule sur le CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tf.config.set_visible_devices([], 'GPU')

# Exécution de la détection automatique
auto_detect_gpu_compatibility()

# --- Reste de ton code (main, train, etc.) ---

# On traverse le dossier ATOMOD, on ouvre le fichier ATOMOD, et on importe la fonction
#from ATOMOD.data_generation import mk_in_silico_data
from ATOMOD.MachineLearning import train
from ATOMOD.data_generation import mk_in_silico_data


def main():
    gen_data=False
    train_model=True

    status={
        'NP':True,
        'abtem':True,
        'feff':True,
        'atomic probability map':True,
        'optimization':False,
    }
    config={
        'root_dir':'simul',                          # répertoire de base
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
                'radius':5.0,
                'a':3.92,
            },
            'nvaccum':2.0,
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
    

    
    if gen_data:
        mk_in_silico_data(config)
    if train_model:
        train(config)
# #########################################################################################
if __name__ == "__main__":
    main()


    # J'ai une méthode mk_in_silico_data() définie dans un fichier ATOMOD.py qui est dans le répertoire /home/bulou/src/ATOMOD/ATOMOD.  je veux appeler cette méthode dans le script Stand_Alone_ATOMOD.py qui est dans le répertoire /home/bulou/src/ATOMOD. Que me conseilles tu ?
