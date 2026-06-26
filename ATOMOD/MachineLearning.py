
import numpy
from PIL import Image
import logging
import matplotlib.pyplot as plt
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#import os ; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from pathlib import Path


import tensorflow as tf
from tensorflow.keras import layers, models,callbacks

##########################################################################################################################
def build_exafs_k_space_encoder(input_shape_exafs, latent_dim=128):
    """
    Encodeur optimisé pour le signal EXAFS dans l'espace k.
    input_shape_exafs : (N_points, N_especes) - ex: (200, 3) pour 3 spectres de 200 pts pour trois espèces chimiques.
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

##########################################################################################################################
def resampling(filename,xmin=2.0,xmax=8.0,N=100):
    """
    Méthode pour rééchantilloner des données contenues dans le fichier filename, en N points régulièrement
    répartis entre [xmin,xmax].
    
    * Les 4 étapes de l'algorithme
      ÉTAPE 1 : Chargement
    ────────────────────
    Charge un fichier texte avec 2 colonnes (x, y)
    Les lignes commençant par '#' sont ignorées (commentaires)
    
    ÉTAPE 2 : Validation
    ─────────────────────
    Vérifie que [xmin, xmax] est dans les limites du fichier
    Si non → lève une erreur (ValueError)
    
    ÉTAPE 3 : Rééchantillonnage
    ────────────────────────────
    Crée N points x uniformément espacés entre xmin et xmax
    Interpole les valeurs y correspondantes
    
    ÉTAPE 4 : Retour
    ────────────────
    Retourne un array (N, 2) avec les données rééchantillonnées
    
    Args:
        filename (str): Chemin du fichier de données
        xmin (float): Valeur x minimale souhaitée (défaut: 2.0)
        xmax (float): Valeur x maximale souhaitée (défaut: 8.0)
        N (int): Nombre de points dans le nouvel échantillon (défaut: 100)
    
    Returns:
        numpy.ndarray: Array de shape (N, 2) avec [x_nouveau, y_nouveau]
    
    Raises:
        ValueError: Si xmin < min(x_original) ou xmax > max(x_original)
    
    Example: Charger et rééchantillonner un spectre XAS
        >>> data = resampling(
                      filename="spectrum.txt",
                      xmin=4.5,           # Énérgie min (keV)
                      xmax=5.5,           # Énérgie max (keV)
                      N=500)              # 500 points réguliers
        >>> print(data.shape)  # Résultat : array de shape (500, 2) contenant exactement 500 points espacés uniformément

    
    """
    # Chargement du fichier
    x_original, y_original = numpy.loadtxt(filename, comments='#', usecols=(0,1),unpack=True)
    # Test de sécurité
    # On vérifie si xmin est trop petit OU si xmax est trop grand
    if xmin < x_original.min() or xmax > x_original.max():
        raise ValueError(
            f"🚨 Erreur de bornes ! Vous demandez un intervalle [{xmin}, {xmax}] "
            f"qui déborde du fichier original.\n"
            f"👉 Plage réelle du fichier : [{x_original.min():.3f}, {x_original.max():.3f}]"
    )

    # Si le test passe, le script continue en toute sécurité
    x_nouveau = numpy.linspace(xmin, xmax, N)
    y_nouveau = numpy.interp(x_nouveau, x_original, y_original)
    return numpy.array([x_nouveau,y_nouveau]).T

##########################################################################################################################
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

#################################################################################################################
def film_block(feature_map, conditioner):
    """Injection de l'information EXAFS dans l'image."""
    # Le conditionneur est un vecteur latent issu de l'EXAFS
    gamma = layers.Dense(feature_map.shape[-1], activation='linear')(conditioner)
    beta = layers.Dense(feature_map.shape[-1], activation='linear')(conditioner)
    
    # Reshaping pour permettre l'opération sur (H, W, C)
    gamma = layers.Reshape((1, 1, feature_map.shape[-1]))(gamma)
    beta = layers.Reshape((1, 1, feature_map.shape[-1]))(beta)

    return layers.Add()([layers.Multiply()([feature_map, gamma]), beta])



#################################################################################################################
def build_atomod_v2(config):
    # --- BRANCHE EXAFS ---
    exafs_input = layers.Input(shape=(config['exafs']['N_POINTS_EXAFS'],
                                      len(config['NP']['structure']['composition'])),
                               name="EXAFS_Input")
    x_ex = layers.Conv1D(64, 3, activation='relu')(exafs_input)
    x_ex = layers.GlobalAveragePooling1D()(x_ex)
    latent_exafs = layers.Dense(128, activation='relu')(x_ex)

    # --- BRANCHE TEM (UNet) ---
    tem_input = layers.Input(shape=(config['image']['H'],
                                    config['image']['W'],
                                    1),
                             name="TEM_Input")
    
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
    output = layers.Conv2D(config['train']['N_CHANNELS_OUT'], (1, 1), activation='sigmoid', name="Output")(c4)

    return models.Model(inputs=[tem_input, exafs_input], outputs=output)

##########################################################################################################################
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - tf.reduce_mean(numerator / (denominator + 1e-7))

##########################################################################################################################
def composition_constraint(y_true, y_pred):
    """Pénalise si la proportion d'atomes prédits s'écarte de la vérité terrain."""
    # Somme des probas par canal (espèce_plan)
    total_true = tf.reduce_sum(y_true, axis=(1, 2))
    total_pred = tf.reduce_sum(y_pred, axis=(1, 2))
    return tf.reduce_mean(tf.square(total_true - total_pred))

#################################################################################################################
def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + 0.1 * composition_constraint(y_true, y_pred)

#################################################################################################################
class CustomMultimodalGenerator(tf.keras.utils.Sequence):
    """
    Générateur de données à la volée pour le modèle multimodal ATOMOD.
    Fournit ([batch_tem, batch_exafs], batch_volume_target) à chaque itération.
    """
    def __init__(self,config,data_ids,shuffle=True,global_max_abs=None):
        """
        Args:
            data_ids (list): Liste des identifiants uniques des nanoparticules (ex: ['part_001', 'part_002', ...])
            data_root (str): Répertoire racine contenant toutes les simulations.
            batch_size (int): Taille des lots (BATCH_SIZE).
            global_max_abs (numpy.ndarray): Vecteur de taille (N_ESPECES,) pour normaliser l'EXAFS.
            shuffle (bool): Mélanger les données à la fin de chaque époque.
        """

        #logger.info(f"{config['root_dir']}")
        self.config=config
        self.batch_size = config['train']['BATCH_SIZE']
        self.data_ids=data_ids
        self.shuffle=shuffle
        self.indexes = numpy.arange(len(self.data_ids))
        if self.shuffle:
            numpy.random.shuffle(self.indexes)


        # on récupère la liste des seeds ayant servis à générer les NP in silico
        #sous_dossiers = [d for d in Path(config['root_dir']).iterdir() if d.is_dir()]
        #for dossier in sous_dossiers:
        #    self.data_ids.append(dossier.parts[-1])


        self.img_shape = (config['image']['W'],config['image']['H'])
        self.n_points_exafs = config['exafs']['N_POINTS_EXAFS']
        self.n_especes = len(config['NP']['structure']['composition'])
        print(self.data_ids)
        print(self.indexes)

        #tab=[]
        #for doss in sous_dossiers:
        #    print(doss)
        #    dossier_proba =  doss / config['train']['prob_maps_img_dir']
        #    for elt in config['NP']['structure']['composition']:
        #        fichiers = [f for f in dossier_proba.iterdir() if f.is_file() and elt in f.name]
        #        tab.append(len(fichiers))
        #tous_identiques = len(set(tab)) <= 1
        
        #if tous_identiques:  # Affiche : True
        #    self.n_z_plans=tab[0]
        self.n_z_plans=config['atomic presence probability map']['N_Z_PLANS']
        #config['NP']['structure']['N_ESPECES']=self.n_especes
        #config['atomic presence probability map']['N_Z_PLANS']=self.n_z_plans
        #config['train']['N_CHANNELS_OUT']=config['NP']['structure']['N_ESPECES']*config['atomic presence probability map']['N_Z_PLANS']

        # Si aucun max absolu n'est fourni, on initialise à 1.0 (pas de normalisation)
        if global_max_abs is None:
            self.global_max_abs = numpy.ones(self.n_especes, dtype=numpy.float32)
        else:
            self.global_max_abs = numpy.array(global_max_abs, dtype=numpy.float32)


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
        end_idx = min((index + 1) * self.batch_size, len(self.data_ids))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Ajustement au cas où le dernier batch est plus petit que BATCH_SIZE
        current_batch_size = len(batch_indexes)
        

        logger.info(f"start_idx={start_idx} end_idx={end_idx}")
        # 2. Initialiser les tableaux NumPy (qui seront envoyés au GPU)
        batch_tem = numpy.zeros((current_batch_size, self.img_shape[0], self.img_shape[1], 1), dtype=numpy.float32)
        batch_exafs = numpy.zeros((current_batch_size, self.n_points_exafs, self.n_especes), dtype=numpy.float32)
        batch_volume_target = numpy.zeros((current_batch_size, self.img_shape[0], self.img_shape[1], self.n_especes * self.n_z_plans), dtype=numpy.float32)
        # 3. Remplir le lot particule par particule
        for i, idx in enumerate(batch_indexes):
            part_id = self.data_ids[idx]
            
            # --- A. Chargement et normalisation de l'image TEM ---
            dossier_tem =  Path(self.config['root_dir']) / part_id / self.config['train']['TEM_img_dir']
            chemin_image = dossier_tem / "TEM.png"
            if chemin_image.exists():
                print(f"✅ L'image {chemin_image} existe bien ! Lecture en cours...")
                
                #img = plt.imread(chemin_image)
                #hauteur = img.shape[0]
                #largeur = img.shape[1]
                
                img=Image.open(chemin_image)
                largeur, hauteur = img.size
                print(img.mode)
                print(f"📷 Image analysée : {chemin_image.name}")
                print(f"📏 Largeur : {largeur} pixels")
                print(f"📐 Hauteur : {hauteur} pixels")
                img_redimensionnee = img.resize((self.config['image']['W'],self.config['image']['H']),
                                                Image.Resampling.LANCZOS)
                print(type(img_redimensionnee))
                tem_img = (img_redimensionnee - numpy.min(img_redimensionnee)) / (numpy.max(img_redimensionnee) - numpy.min(img_redimensionnee) + 1e-7)
                print(type(tem_img),tem_img.shape)
                img_gris = 0.299 * tem_img[:, :, 0] + 0.587 * tem_img[:, :, 1] + 0.114 * tem_img[:, :, 2]

                
                batch_tem[i, ..., 0] = img_gris
                #plt.imshow(img_gris,cmap='gray' )
                #plt.show()
            else:
                print(f"❌ Erreur : Impossible de trouver l'image à l'emplacement :")
                print(f"   {chemin_image.resolve()}")  # .resolve() affiche le chemin absolu complet pour aider à déboguer
            # --- B. Chargement et normalisation des spectres EXAFS ---
            exafs_matrix = numpy.zeros((self.n_points_exafs, self.n_especes), dtype=numpy.float32)
            dossier_exafs =  Path(self.config['root_dir']) / part_id / self.config['train']['EXAFS_dir']
            fichiers = [f for f in dossier_exafs.iterdir() if f.is_file() and "k2" in f.name.lower()]
            #print(f"fichiers={fichiers}")
            data_exafs={}
            #print(self.n_points_exafs, self.n_especes)
            for f in fichiers:
                data_exafs[f]=resampling(f,xmin=2.0,xmax=8.0,N=self.config['exafs']['N_POINTS_EXAFS'])
                print(f"data_exafs.shape={data_exafs[f].shape}")

            for ielt,elt in enumerate(self.config['NP']['structure']['composition']):
                chemin_trouve = next((p for p in fichiers if elt in p.name), None)
                #print(elt)
                #plt.plot(data_exafs[chemin_trouve][0],data_exafs[chemin_trouve][1],label=elt)
                #plt.show()
                #print(data_exafs[chemin_trouve][1])
                #data = numpy.loadtxt(chemin_trouve, skiprows=1) 
                #print(f"type(data)={type(data)},data.shape={data.shape}")
                #ff=resampling(chemin_trouve,xmin=2.0,xmax=8.0,N=self.config['exafs']['N_POINTS_EXAFS'])
                #print(f"type(ff)={type(ff)},ff.shape={ff.shape}")
                #print(type(data_exafs[chemin_trouve][1]),data_exafs[chemin_trouve][1].shape)
                #print(type(exafs_matrix),exafs_matrix.shape)
                exafs_matrix[:, ielt] = data_exafs[chemin_trouve][:,1]

                #exafs_matrix[:, ielt] = data[:,1]
            # Normalisation Max-Abs indépendante pour chaque colonne (Broadcasting)
            print(f"exafs_matrix.shape={exafs_matrix.shape}")
            batch_exafs[i] = numpy.clip(exafs_matrix / self.global_max_abs, -1.0, 1.0)
            print(f"batch_exafs.shape={batch_exafs.shape}")
            print(batch_exafs[i][:][0].shape)

            #for ielt in range(self.n_especes):
            #    plt.plot(batch_exafs[i,:,ielt])
            #plt.show()
            #for ielt in range(self.n_especes):
            #    plt.plot(batch_exafs[i][ielt])
            #plt.show()

            # --- C. Chargement et concaténation des volumes 3D cibles ---
            # On doit fusionner les plans Z de toutes les espèces dans la dernière dimension
            dossier_proba =  Path(self.config['root_dir']) / part_id / self.config['train']['prob_maps_img_dir']
            channels_list = []
            for elt in self.config['NP']['structure']['composition']:
                for z in range(self.n_z_plans):
                    name=f"img_0000_{elt}_{z:04d}"
                    fichier = [f for f in dossier_proba.iterdir() if f.is_file() and name in f.name]
                    image_path = fichier[0]
                    if image_path.exists():
                        channels_list.append(self.image_reading_and_processing(image_path))
            print(type(channels_list))
            print(numpy.shape(channels_list))
            tab = numpy.array(channels_list)
            batch_volume_target[i]=numpy.moveaxis(tab, 0, -1)
            print(numpy.shape(batch_volume_target[i]))

        return [batch_tem, batch_exafs],  batch_volume_target

    def image_reading_and_processing(self,img_path):
        print(f"✅ L'image {img_path} existe bien ! Lecture en cours...")
        
        #img = plt.imread(img_path)
        #hauteur = img.shape[0]
        #largeur = img.shape[1]
        
        img=Image.open(img_path)
        largeur, hauteur = img.size
        print(img.mode)
        print(f"📷 Image analysée : {img_path.name}")
        print(f"📏 Largeur : {largeur} pixels")
        print(f"📐 Hauteur : {hauteur} pixels")
        img_redimensionnee = img.resize((self.config['image']['W'],
                                         self.config['image']['H']),
                                        Image.Resampling.LANCZOS)
        print(type(img_redimensionnee))
        new_img = (img_redimensionnee - numpy.min(img_redimensionnee)) / (numpy.max(img_redimensionnee) - numpy.min(img_redimensionnee) + 1e-7)
        print(type(new_img),new_img.shape)
        img_grey = 0.299 * new_img[:, :, 0] + 0.587 * new_img[:, :, 1] + 0.114 * new_img[:, :, 2]
        print(type(img_grey),img_grey.shape)
        return img_grey
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def train(config):

    data_ids=[]
    list_seeds_dir = [d for d in Path(config['root_dir']).iterdir() if d.is_dir()]
    for d in list_seeds_dir:
        data_ids.append(d.parts[-1])
    print(data_ids)

    list_proba_map=[]
    for d in list_seeds_dir:
        proba_dir =  d / config['train']['prob_maps_img_dir']
        for elt in config['NP']['structure']['composition']:    
            proba_map = [f for f in proba_dir.iterdir() if f.is_file() and elt in f.name]
            list_proba_map.append(len(proba_map)) # comptage des maps de proba pour chaque elt
        same_number_of_maps = len(set(list_proba_map)) <= 1
        
    if same_number_of_maps:  # Affiche : True
        config['atomic presence probability map']['N_Z_PLANS']=list_proba_map[0]

    config['NP']['structure']['N_ESPECES']=len(config['NP']['structure']['composition'])
    config['train']['N_CHANNELS_OUT']=config['NP']['structure']['N_ESPECES']*config['atomic presence probability map']['N_Z_PLANS']
    
    train_generator = CustomMultimodalGenerator(config,data_ids)
    logger.info(10*"-")
    a=train_generator[1]
    print(type(a))

    model = build_atomod_v2(config)    # instanciation du modèle
    model.compile(optimizer=config['train']['optimizer'],
                  loss=combined_loss,
                  metrics=['accuracy'])


    # 2. Préparation des données (Simulation de générateurs)
    # Note: Tes générateurs doivent renvoyer ([batch_tem, batch_exafs], batch_volume_3d)
    # train_gen = CustomMultimodalGenerator(...) 

    # 3. Callbacks pour monitorer l'évolution
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_atomod.keras", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]

    # Instanciation du générateur




    #print(f"steps_per_epoch={len(train_generator),}")
    # 4. Lancement de l'instruction
    print("Début de l'entraînement multimodal...")
    history = model.fit(
         train_generator,                        # données d'entrées
         epochs=100,   # Le nombre total de fois où le modèle va parcourir l'intégralité du jeu de données d'entraînement.
         callbacks=callbacks
    )
    #history = model.fit(
    #     x=train_generator,                        # données d'entrées
    #     validation_data=validation_generator,   #
    #     epochs=100,   # Le nombre total de fois où le modèle va parcourir l'intégralité du jeu de données d'entraînement.
    #     callbacks=callbacks
    #)
    
    print("Modèle prêt pour la reconstruction 3D.")

    
##########################################################################################################################
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
