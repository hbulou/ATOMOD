import numpy
import logging
import matplotlib.pyplot as plt
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import tensorflow as tf
from tensorflow.keras import layers, models,callbacks

def build_exafs_k_space_encoder(input_shape_exafs, latent_dim=128):
    """
    Encodeur optimisé pour le signal EXAFS dans l'espace k.
    input_shape_exafs : (N_points, 3) - ex: (200, 3) pour 3 espèces chimiques.
    """
    inputs = layers.Input(shape=input_shape_exafs, name="EXAFS_k_input")
    
    # 1er Bloc : Filtre large pour capturer les oscillations basses fréquences (premières couches de coordination)
    x = layers.Conv1D(filters=32, kernel_size=11, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x) # Stabilise l'entraînement sur les amplitudes
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # 2ème Bloc : Filtre intermédiaire pour les détails des battements d'oscillations
    x = layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # 3ème Bloc : Filtre plus court pour les détails haute fréquence (désordre structurel du HEA)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Condensation du signal : extrait les caractéristiques globales du spectre
    x = layers.GlobalAveragePooling1D()(x)
    
    # Projection vers l'espace latent pour alimenter les blocs FiLM du UNet
    latent_vector = layers.Dense(latent_dim, activation='relu', name="EXAFS_latent_features")(x)
    
    return models.Model(inputs, latent_vector, name="EXAFS_k_Encoder")
def resampling(filename,xmin=2.0,xmax=8.0,N=100):
    # 2. Chargement du fichier
    x_original, y_original = numpy.loadtxt(filename, comments='#', usecols=(0,1),unpack=True)
    # 3. LE TEST DE SÉCURITÉ
    # On vérifie si xmin est trop petit OU si xmax est trop grand
    if xmin < x_original.min() or xmax > x_original.max():
        raise ValueError(
            f"🚨 Erreur de bornes ! Vous demandez un intervalle [{xmin}, {xmax}] "
            f"qui déborde du fichier original.\n"
            f"👉 Plage réelle du fichier : [{x_original.min():.3f}, {x_original.max():.3f}]"
    )

    # 4. Si le test passe, le script continue en toute sécurité
    x_nouveau = numpy.linspace(xmin, xmax, N)
    y_nouveau = numpy.interp(x_nouveau, x_original, y_original)
    return x_nouveau,y_nouveau
    


def load_exafs_inputs(config):
    """
    file_paths: list de 3 chemins vers les fichiers .dat [A.dat, B.dat, C.dat]
    On suppose que les fichiers ont deux colonnes : [k, chi(k)]
    """
    spectra = []
    data={}
    k=[]
    N=[]
    for filename in config['exafs']:
        
        name=f"{config['DATA_ROOT']}/{filename}"
        try:
            #data[filename] = numpy.loadtxt(name, comments='#', usecols=(0,1))
            data[filename]=resampling(name,xmin=2.0,xmax=8.0,N=config['N_POINTS_EXAFS'])
        except Exception as e:
            logger.error(f"⚠️ Erreur sur {name} : {e}")
        # On extrait la colonne du signal chi(k)
        # Assurez-vous que tous les fichiers ont la même longueur N
        #print(data[filename])
        #N.append(len(data[filename][:,1]))
        #k.append(data[filename][:,0])
        logger.info(f"file {name} {len(data[filename][0])} ")
        plt.plot(data[filename][0],data[filename][1],label=filename)
        plt.legend()
    plt.show()

    for filename in config['exafs']:
        chi_k = data[filename][1]
        spectra.append(chi_k)
        k.append(data[filename][0])

    k_identiques = all(numpy.allclose(k[0], tab) for tab in k[1:])
    print(f"len(k)={len(k)} {k_identiques}")
    
        
    # On empile les spectres pour obtenir une matrice (N, len(config['exafs']))
    # np.stack avec axis=-1 crée la forme (N, Nsp)
    exafs_matrix = numpy.stack(spectra, axis=-1)
    
    return exafs_matrix

def film_block(feature_map, conditioner):
    """Injection de l'information EXAFS dans l'image."""
    # Le conditionneur est un vecteur latent issu de l'EXAFS
    gamma = layers.Dense(feature_map.shape[-1], activation='linear')(conditioner)
    beta = layers.Dense(feature_map.shape[-1], activation='linear')(conditioner)
    
    # Reshaping pour permettre l'opération sur (H, W, C)
    gamma = layers.Reshape((1, 1, feature_map.shape[-1]))(gamma)
    beta = layers.Reshape((1, 1, feature_map.shape[-1]))(beta)

    return layers.Add()([layers.Multiply()([feature_map, gamma]), beta])



def build_atomod_v2(config):
    # --- BRANCHE EXAFS ---
    exafs_input = layers.Input(shape=(config['N_POINTS_EXAFS'],config['N_ESPECES']), name="EXAFS_Input")
    x_ex = layers.Conv1D(64, 3, activation='relu')(exafs_input)
    x_ex = layers.GlobalAveragePooling1D()(x_ex)
    latent_exafs = layers.Dense(128, activation='relu')(x_ex)

    # --- BRANCHE TEM (UNet) ---
    tem_input = layers.Input(shape=(config['H'],config['W'],1), name="TEM_Input")
    
    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(tem_input)
    c1 = film_block(c1, latent_exafs) # Conditionnement
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = film_block(c2, latent_exafs) # Conditionnement
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    b = film_block(b, latent_exafs)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    
    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)

    # Sortie (Volume de probabilités : Espèces x Plans)
    output = layers.Conv2D(config['N_CHANNELS_OUT'], (1, 1), activation='sigmoid', name="Output")(c4)

    return models.Model(inputs=[tem_input, exafs_input], outputs=output)

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - tf.reduce_mean(numerator / (denominator + 1e-7))

def composition_constraint(y_true, y_pred):
    """Pénalise si la proportion d'atomes prédits s'écarte de la vérité terrain."""
    # Somme des probas par canal (espèce_plan)
    total_true = tf.reduce_sum(y_true, axis=(1, 2))
    total_pred = tf.reduce_sum(y_pred, axis=(1, 2))
    return tf.reduce_mean(tf.square(total_true - total_pred))

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + 0.1 * composition_constraint(y_true, y_pred)

class CustomMultimodalGenerator(tf.keras.utils.Sequence):
    """
    Générateur de données à la volée pour le modèle multimodal ATOMOD.
    Fournit ([batch_tem, batch_exafs], batch_volume_target) à chaque itération.
    """
    def __init__(self,config):
        """
        Args:
            data_ids (list): Liste des identifiants uniques des nanoparticules (ex: ['part_001', 'part_002', ...])
            data_root (str): Répertoire racine contenant toutes les simulations.
            batch_size (int): Taille des lots (BATCH_SIZE).
            global_max_abs (numpy.ndarray): Vecteur de taille (N_ESPECES,) pour normaliser l'EXAFS.
            shuffle (bool): Mélanger les données à la fin de chaque époque.
        """
        self.data_root = config['DATA_ROOT']
        self.batch_size = config['BATCH_SIZE']
        self.img_shape = (config['W'],config['H'])
        self.n_points_exafs = config['N_POINTS_EXAFS']
        self.n_especes = config['N_ESPECES']
        self.n_z_plans = config['N_PLANS']
    def __getitem__(self,index):
        """ Génère un lot (batch) de données.
        En Python, les méthodes entourées de doubles underscores (appelées méthodes magiques ou dunder methods)
        sont conçues pour être déclenchées automatiquement par des opérateurs ou des fonctions du langage.
        Pour le générateur train_generator(), __getitem__ est appelé en coulisses de deux façons différentes :
            1. Par TensorFlow/Keras (Pendant l'entraînement) : lorsqu'on lances model.fit(train_generator), Keras sait que train_generator est une instance de tf.keras.utils.Sequence.
               À chaque itération (chaque step), Keras va demander le lot suivant en utilisant la syntaxe des crochets de Python (le slicing).
               C'est précisément l'utilisation des crochets [] qui déclenche l'exécution de __getitem__.
            2. Pour tester ou déboguer le code
               Appel manuel indirect de __getitem__ en demandant le lot 0 :
                   [batch_x, batch_y] = train_generator[0]
        """
        
        # 1. Sélectionner les indices du lot courant
        start_idx = index * self.batch_size
        end_idx   = (index + 1) * self.batch_size
        print(f"{start_idx} {end_idx}")
        # 4. Renvoi du couple strict exigé par model.fit()
        #return [batch_tem, batch_exafs], batch_volume_target

def train():
    # --- Configuration ---
    config={
        'DATA_ROOT':'doc/tutorials',
        'exafs':["k2chi(k)_Au.dat", "k2chi(k)_Co.dat", "k2chi(k)_Pt.dat"],
        'NPARTICULES':1,         # nombre de particules utilisées pour l'entrainement
        'BATCH_SIZE':4,          # Taille des lots 
        'N_POINTS_EXAFS': 200,   # Nombre de points par spectre
        'N_ESPECES' : 3,         # CoCrFeMnNi par exemple
        'N_PLANS' : 10,          # Nombre de plans en Z
        'H':128,
        'W':128,
        'optimizer':'adam',
    }
    config['N_CHANNELS_OUT']= config['N_ESPECES']*config['N_PLANS']
    # 1. Instanciation du modèle
    model = build_atomod_v2(config)
    model.compile(optimizer=config['optimizer'], loss=combined_loss, metrics=['accuracy'])


    # 2. Préparation des données (Simulation de générateurs)
    # Note: Tes générateurs doivent renvoyer ([batch_tem, batch_exafs], batch_volume_3d)
    # train_gen = CustomMultimodalGenerator(...) 

    # 3. Callbacks pour monitorer l'évolution
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_atomod.keras", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]

    # Instanciation du générateur
    train_generator = CustomMultimodalGenerator(config)
    train_generator[0]
    #print(f"steps_per_epoch={len(train_generator),}")
    # 4. Lancement de l'instruction
    print("Début de l'entraînement multimodal...")
    #history = model.fit(
    #     x=train_generator,                        # données d'entrées
    #     validation_data=validation_generator,   #
    #     epochs=100,   # Le nombre total de fois où le modèle va parcourir l'intégralité du jeu de données d'entraînement.
    #     callbacks=callbacks
    #)
    
    print("Modèle prêt pour la reconstruction 3D.")

    
def dvl():
    # --- Configuration ---
    H, W = 128, 128
    N_ESPECES = 5        # CoCrFeMnNi par exemple
    N_PLANS = 10         # Nombre de plans en Z
    N_CHANNELS_OUT = N_ESPECES * N_PLANS


    #config={
    #    'DATA_ROOT':'simul_save',
    #    'exafs':["xmu_Au.dat", "xmu_Co.dat", "xmu_Pt.dat"]
    #}
    config={
        'DATA_ROOT':'doc/tutorials',
        'exafs':["k2chi(k)_Au.dat", "k2chi(k)_Co.dat", "k2chi(k)_Pt.dat"],
        'N_POINTS_EXAFS': 200 # Nombre de points par spectre
    }

    
    exafs_data = load_exafs_inputs(config)
    
    print(exafs_data.shape)
    for i in range(exafs_data.shape[1]):
        #print(exafs_data)
        valmax=max(numpy.abs(exafs_data[:,i]))
        exafs_data[:,i]=exafs_data[:,i]/valmax
        print(max(numpy.abs(exafs_data[:,i])))
        plt.plot(exafs_data[:,i])
    plt.show()
    # Définition de la forme pour le réseau neuronal
    n_points = exafs_data.shape[0] # Nombre de lignes
    n_species = exafs_data.shape[1] # Devrait être 3

    input_shape_exafs = (n_points, n_species)
    logger.info(f"Forme de l'entrée EXAFS : {input_shape_exafs}")

    # 1. On instancie l'encodeur EXAFS
    exafs_encoder = build_exafs_k_space_encoder(input_shape_exafs=input_shape_exafs, latent_dim=128)
    # 2. On définit les entrées globales du grand réseau
    tem_input = layers.Input(shape=(128, 128, 1), name="TEM_Input")
    exafs_input = layers.Input(shape=input_shape_exafs, name="EXAFS_Input")
    # 3. On connecte l'EXAFS à son encodeur pour obtenir le vecteur latent
    # C'est cette ligne qui lie l'encodeur à l'entraînement global !
    latent_exafs = exafs_encoder(exafs_input)
    # 4. On construit le UNet en lui passant ce latent_exafs dans les blocs FiLM
    # (Comme vu dans le script précédent)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(tem_input)
    c1 = film_block(c1, latent_exafs) # Utilise le vecteur pour se calibrer

    logger.info("DONE!")
# #########################################################################################
if __name__ == "__main__":
    train()


    # comment constuire input_shape_exafs pour 3 espèce chimique ayant chacune une série de 10 fichiers XY EXAFS ?
