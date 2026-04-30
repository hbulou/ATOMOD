import numpy
import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_exafs_inputs(config):
    """
    file_paths: list de 3 chemins vers les fichiers .dat [A.dat, B.dat, C.dat]
    On suppose que les fichiers ont deux colonnes : [k, chi(k)]
    """
    spectra = []
    data={}
    N=[]
    for filename in config['exafs']:
        
        name=f"{config['root_dir']}/{filename}"
        try:
            data[filename] = numpy.loadtxt(name, comments='#', usecols=(0,1))
        except Exception as e:
            logger.error(f"⚠️ Erreur sur {name} : {e}")
        # On extrait la colonne du signal chi(k)
        # Assurez-vous que tous les fichiers ont la même longueur N
        N.append(len(data[filename][:,1]))
        logger.info(f"file {name} {len(data[filename][:,1])}")
    for filename in config['exafs']:
        chi_k = data[filename][0:min(N), 1]
        spectra.append(chi_k)
    # On empile les spectres pour obtenir une matrice (N, len(config['exafs']))
    # np.stack avec axis=-1 crée la forme (N, Nsp)
    exafs_matrix = numpy.stack(spectra, axis=-1)
    
    return exafs_matrix



def train():
    # --- Configuration ---
    H, W = 128, 128
    N_POINTS_EXAFS = 200 # Nombre de points par spectre
    N_ESPECES = 5        # CoCrFeMnNi par exemple
    N_PLANS = 10         # Nombre de plans en Z
    N_CHANNELS_OUT = N_ESPECES * N_PLANS


    config={
        'root_dir':'simul_save',
        'exafs':["xmu_Au.dat", "xmu_Co.dat", "xmu_Pt.dat"]
    }


    exafs_data = load_exafs_inputs(config)
    # Définition de la forme pour le réseau neuronal
    n_points = exafs_data.shape[0] # Nombre de lignes
    n_species = exafs_data.shape[1] # Devrait être 3

    input_shape_exafs = (n_points, n_species)
    logger.info(f"Forme de l'entrée EXAFS : {input_shape_exafs}")

    logger.info("DONE!")
# #########################################################################################
if __name__ == "__main__":
    train()


    # comment constuire input_shape_exafs pour 3 espèce chimique ayant chacune une série de 10 fichiers XY EXAFS ?
