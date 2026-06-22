import os
import tensorflow as tf

# Configuration du logging
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
        print("[ATOMOD] Aucun GPU physique détecté. Mode CPU par défaut.")
        return

    try:
        # Récupère les détails matériels du premier GPU
        details = tf.config.experimental.get_device_details(gpus[0])
        compute_cap = details.get('compute_capability', (0, 0))
        gpu_name = details.get('device_name', 'GPU Inconnu')
        
        print(f"[ATOMOD] GPU détecté : {gpu_name} (Compute Capability: {compute_cap[0]}.{compute_cap[1]})")
        
        # Les versions récentes de TensorFlow ont de graves problèmes de compilation
        # JIT en dessous de la version 6.0 (Pascal). Ta Quadro M1000M est en 5.0.
        if compute_cap[0] < 6:
            print(f"[ATOMOD] ATTENTION : L'architecture de votre {gpu_name} est trop ancienne (CC < 6.0).")
            print("[ATOMOD] Désactivation automatique du GPU pour éviter un crash 'CUDA_ERROR_NO_BINARY_FOR_GPU'.")
            
            # Action corrective : On masque le GPU pour le processus courant
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            # On force TensorFlow à réévaluer les périphériques disponibles immédiatement
            tf.config.set_visible_devices([], 'GPU')
            print("[ATOMOD] Bascule sur le CPU réussie.")
            
    except Exception as e:
        print(f"[ATOMOD] Erreur lors de l'analyse du GPU ({e}). Par sécurité, bascule sur le CPU.")
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
    #mk_in_silico_data()

    
    train()
# #########################################################################################
if __name__ == "__main__":
    main()


    # J'ai une méthode mk_in_silico_data() définie dans un fichier ATOMOD.py qui est dans le répertoire /home/bulou/src/ATOMOD/ATOMOD.  je veux appeler cette méthode dans le script Stand_Alone_ATOMOD.py qui est dans le répertoire /home/bulou/src/ATOMOD. Que me conseilles tu ?
