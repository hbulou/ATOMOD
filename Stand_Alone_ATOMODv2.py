"""
=============================================================================
ATOMOD v2 - Reconstruction Tomographique 3D Implicite d'une Nanoparticule HEA
Projet M2P2_HEA (PEPR DIADEM)

Architecture multimodale :
  - Encodeur CNN-1D pour les spectres EXAFS k³χ(k)
  - UNet 2D conditionné par des blocs FiLM pour l'image TEM
  - Sortie : volume de probabilité de présence atomique par espèce et plan Z

Usage :
  python Stand_Alone_ATOMODv2.py           (entraînement)
  ou dans un notebook Jupyter : exécuter les cellules section par section
=============================================================================
"""

# =============================================================================
# SECTION 0 : Imports standards et pré-vérification GPU (avant import TF)
# =============================================================================
import os
import subprocess
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif pour serveurs sans écran
import matplotlib.pyplot as plt


def _pre_verifier_gpu_nvidia_smi():
    """
    Vérifie la Compute Capability via nvidia-smi AVANT l'import de TensorFlow,
    afin que CUDA_VISIBLE_DEVICES soit correctement pris en compte dès le départ.
    Retourne True si un GPU compatible (CC >= 6.0) est détecté, False sinon.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False

        for ligne in result.stdout.strip().split('\n'):
            parties = [p.strip() for p in ligne.split(',')]
            if len(parties) >= 2:
                try:
                    cc = float(parties[1])
                    nom_gpu = parties[0]
                    if cc < 6.0:
                        print(f"[PRÉ-CHECK] GPU '{nom_gpu}' CC={cc:.1f} < 6.0 → forçage CPU.")
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        return False
                    else:
                        print(f"[PRÉ-CHECK] GPU '{nom_gpu}' CC={cc:.1f} >= 6.0 → GPU activé.")
                        return True
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False  # nvidia-smi absent → TF décidera


# Exécution de la pré-vérification avant toute initialisation CUDA
_gpu_pre_ok = _pre_verifier_gpu_nvidia_smi()

# Import de TensorFlow après la configuration de CUDA_VISIBLE_DEVICES
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# SECTION 1 : Détection matérielle via TensorFlow (vérification finale)
# =============================================================================

def configurer_materiel():
    """
    Vérifie la Compute Capability du GPU via tf.config.
    Si CC < 6.0 (ex : Nvidia Quadro M1000M), bascule sur CPU.
    Retourne 'GPU' ou 'CPU'.
    """
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("[MATÉRIEL] Aucun GPU visible → CPU sélectionné.")
        return 'CPU'

    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            cc = details.get('compute_capability', (0, 0))
            version = cc[0] + cc[1] / 10.0
            nom = details.get('device_name', 'Inconnu')
            print(f"[MATÉRIEL] GPU : {nom} | CC : {version:.1f}")

            if version < 6.0:
                print(f"[MATÉRIEL] CC {version:.1f} < 6.0 → basculement CPU.")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                tf.config.set_visible_devices([], 'GPU')
                return 'CPU'

            # Activation de la croissance mémoire dynamique
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"[MATÉRIEL] Impossible de lire les détails GPU : {e} → CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return 'CPU'

    print(f"[MATÉRIEL] GPU compatible → GPU sélectionné.")
    return 'GPU'


DISPOSITIF = configurer_materiel()


# =============================================================================
# SECTION 2 : Hyperparamètres et configuration globale
# =============================================================================

# --- Espèces chimiques (ordre alphabétique conforme aux données) ---
ESPECES = ['Au', 'Co', 'Pd', 'Pt', 'Rh']
N_ESPECES = len(ESPECES)          # 5

# --- Dimensions des données ---
HEIGHT_IMAGE_TEM = 128            # Hauteur de l'image TEM en pixels
WIDTH_IMAGE_TEM  = 128            # Largeur de l'image TEM en pixels
N_Z_PLANS        = 14             # Nombre de plans Z (déterminé depuis les données)

# --- EXAFS : rééchantillonnage dans [KMIN, KMAX] avec NPTS_SPECTRE points ---
NPTS_SPECTRE     = 200
K_MIN            = 2.0            # k minimum (Å⁻¹) pour le rééchantillonnage
K_MAX            = 8.0            # k maximum (Å⁻¹)

# --- Dimension dérivée ---
N_CANAUX_SORTIE  = N_ESPECES * N_Z_PLANS   # 70 canaux en sortie

# --- Entraînement ---
BATCH_SIZE            = 4
N_EPOQUES             = 100
TAUX_APPRENTISSAGE    = 1e-4
PATIENCE_ARRET        = 15       # EarlyStopping
PATIENCE_REDUCE_LR    = 5        # ReduceLROnPlateau
RATIO_VALIDATION      = 0.2      # 20 % des nanoparticules pour la validation

# --- Chemins ---
REPERTOIRE_SIMUL = './simul'
CHEMIN_MODELE    = './atomod_v2_best.keras'

# --- Encodeur EXAFS ---
TAILLE_VECTEUR_LATENT = 128

# --- Normalisation Max-Abs (un scalaire par espèce, ordre conforme à ESPECES) ---
# Vecteur prédéfini ; à recalculer avec calculer_global_max_abs() lors du premier lancement.
GLOBAL_MAX_ABS = np.ones(N_ESPECES, dtype=np.float32)

# Graine aléatoire pour la reproductibilité du découpage train/val
GRAINE_ALEATOIRE = 42


# =============================================================================
# SECTION 3 : Couche FiLM (Feature-wise Linear Modulation)
# =============================================================================

class BlocFiLM(layers.Layer):
    """
    Bloc FiLM : modulation affine des cartes de caractéristiques par le vecteur EXAFS.
    Pour chaque carte feature_map de forme (B, H, W, C), calcule :
        sortie = gamma(latent_exafs) * feature_map + beta(latent_exafs)
    où gamma et beta sont des projections linéaires du vecteur latent EXAFS.
    """

    def __init__(self, nb_filtres, **kwargs):
        super().__init__(**kwargs)
        self.nb_filtres = nb_filtres
        # Dense sans activation : gamma et beta peuvent être négatifs
        self.dense_gamma = layers.Dense(nb_filtres, activation='linear',
                                        name=f'{self.name}_gamma')
        self.dense_beta  = layers.Dense(nb_filtres, activation='linear',
                                        name=f'{self.name}_beta')

    def call(self, inputs):
        """
        inputs : [feature_map (B,H,W,C), vecteur_exafs (B,LATENT_DIM)]
        """
        feature_map, vecteur_exafs = inputs

        gamma = self.dense_gamma(vecteur_exafs)            # (B, C)
        beta  = self.dense_beta(vecteur_exafs)             # (B, C)

        # Expansion pour broadcast spatial (B, 1, 1, C)
        gamma = tf.reshape(gamma, (-1, 1, 1, self.nb_filtres))
        beta  = tf.reshape(beta,  (-1, 1, 1, self.nb_filtres))

        return gamma * feature_map + beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'nb_filtres': self.nb_filtres})
        return cfg


# =============================================================================
# SECTION 4 : Encodeur EXAFS (CNN 1D)
# =============================================================================

def construire_encodeur_exafs(npts=NPTS_SPECTRE, n_esp=N_ESPECES,
                               latent_dim=TAILLE_VECTEUR_LATENT):
    """
    CNN-1D qui extrait une empreinte chimique globale à partir des spectres k³χ(k).

    Entrée  : (BATCH, NPTS_SPECTRE, N_ESPECES)
    Sortie  : (BATCH, TAILLE_VECTEUR_LATENT)

    Filtres décroissants (11→7→3) pour capturer oscillations basses, moyennes et hautes
    fréquences du signal EXAFS, représentatives des couches de coordination successives.
    """
    entree = keras.Input(shape=(npts, n_esp), name='entree_exafs')
    x = entree

    # Bloc 1 : oscillations basse fréquence (premières couches de coordination)
    x = layers.Conv1D(32, kernel_size=11, padding='same', activation='relu',
                      name='exafs_conv1')(x)
    x = layers.BatchNormalization(name='exafs_bn1')(x)
    x = layers.MaxPooling1D(2, name='exafs_pool1')(x)

    # Bloc 2 : oscillations intermédiaires
    x = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu',
                      name='exafs_conv2')(x)
    x = layers.BatchNormalization(name='exafs_bn2')(x)
    x = layers.MaxPooling1D(2, name='exafs_pool2')(x)

    # Bloc 3 : détails haute fréquence (désordre structural du HEA)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                      name='exafs_conv3')(x)
    x = layers.BatchNormalization(name='exafs_bn3')(x)

    # Agrégation globale → vecteur unique indépendant de la longueur du signal
    x = layers.GlobalAveragePooling1D(name='exafs_gap')(x)

    # Projection dans l'espace latent
    latent = layers.Dense(latent_dim, activation='relu',
                          name='exafs_latent')(x)

    return keras.Model(inputs=entree, outputs=latent, name='Encodeur_EXAFS')


# =============================================================================
# SECTION 5 : Blocs UNet (encodage et décodage)
# =============================================================================

def bloc_encodeur(x, nb_filtres, nom):
    """
    Bloc d'encodage : Conv2D → BN → ReLU → Conv2D → BN → ReLU → MaxPool.
    Retourne (sortie_poolée, connexion_de_saut_avant_pool).
    """
    x = layers.Conv2D(nb_filtres, 3, padding='same', activation='relu',
                      name=f'{nom}_c1')(x)
    x = layers.BatchNormalization(name=f'{nom}_bn1')(x)
    x = layers.Conv2D(nb_filtres, 3, padding='same', activation='relu',
                      name=f'{nom}_c2')(x)
    x = layers.BatchNormalization(name=f'{nom}_bn2')(x)
    saut   = x                                                 # connexion de saut
    poolee = layers.MaxPooling2D(2, name=f'{nom}_pool')(x)
    return poolee, saut


def bloc_decodeur(x, saut, nb_filtres, nom):
    """
    Bloc de décodage : UpSampling → concaténation avec saut → 2×(Conv2D → BN → ReLU).
    """
    x = layers.UpSampling2D(2, name=f'{nom}_up')(x)
    x = layers.Concatenate(name=f'{nom}_concat')([x, saut])
    x = layers.Conv2D(nb_filtres, 3, padding='same', activation='relu',
                      name=f'{nom}_c1')(x)
    x = layers.BatchNormalization(name=f'{nom}_bn1')(x)
    x = layers.Conv2D(nb_filtres, 3, padding='same', activation='relu',
                      name=f'{nom}_c2')(x)
    x = layers.BatchNormalization(name=f'{nom}_bn2')(x)
    return x


# =============================================================================
# SECTION 6 : Architecture complète ATOMOD v2 (UNet + FiLM)
# =============================================================================

def construire_atomod_v2(H=HEIGHT_IMAGE_TEM, W=WIDTH_IMAGE_TEM,
                          npts=NPTS_SPECTRE, n_esp=N_ESPECES,
                          n_z=N_Z_PLANS, latent_dim=TAILLE_VECTEUR_LATENT):
    """
    Modèle ATOMOD v2 : UNet 2D conditionné par FiLM à partir d'un encodeur EXAFS.

    Entrées  : [image TEM (B,H,W,1), spectres EXAFS (B,NPTS,N_ESP)]
    Sortie   : volume de probabilité (B,H,W, N_ESP*N_Z) ∈ [0,1]
    """
    n_canaux = n_esp * n_z

    # --- Entrées ---
    entree_tem   = keras.Input(shape=(H, W, 1),      name='entree_tem')
    entree_exafs = keras.Input(shape=(npts, n_esp),  name='entree_exafs')

    # --- Encodeur EXAFS → vecteur latent chimique ---
    enc_exafs    = construire_encodeur_exafs(npts, n_esp, latent_dim)
    latent_exafs = enc_exafs(entree_exafs)            # (B, latent_dim)

    # --- UNet encodeur avec conditionnement FiLM après chaque bloc ---
    # Niveau 1 – 64 filtres
    p1, s1  = bloc_encodeur(entree_tem, 64, 'enc1')
    s1_film = BlocFiLM(64,  name='film_enc1')([s1, latent_exafs])

    # Niveau 2 – 128 filtres
    p2, s2  = bloc_encodeur(p1, 128, 'enc2')
    s2_film = BlocFiLM(128, name='film_enc2')([s2, latent_exafs])

    # Niveau 3 – 256 filtres
    p3, s3  = bloc_encodeur(p2, 256, 'enc3')
    s3_film = BlocFiLM(256, name='film_enc3')([s3, latent_exafs])

    # Niveau 4 – 512 filtres
    p4, s4  = bloc_encodeur(p3, 512, 'enc4')
    s4_film = BlocFiLM(512, name='film_enc4')([s4, latent_exafs])

    # --- Goulot d'étranglement (bottleneck) – 1024 filtres ---
    b = layers.Conv2D(1024, 3, padding='same', activation='relu',
                      name='bottleneck_c1')(p4)
    b = layers.BatchNormalization(name='bottleneck_bn1')(b)
    b = layers.Conv2D(1024, 3, padding='same', activation='relu',
                      name='bottleneck_c2')(b)
    b = layers.BatchNormalization(name='bottleneck_bn2')(b)
    b = BlocFiLM(1024, name='film_bottleneck')([b, latent_exafs])

    # --- UNet décodeur – utilise les connexions de saut modulées par FiLM ---
    x = bloc_decodeur(b, s4_film, 512, 'dec4')
    x = bloc_decodeur(x, s3_film, 256, 'dec3')
    x = bloc_decodeur(x, s2_film, 128, 'dec2')
    x = bloc_decodeur(x, s1_film,  64, 'dec1')

    # --- Couche de sortie : Conv 1×1 + Sigmoid → probabilités ∈ [0,1] ---
    sortie = layers.Conv2D(n_canaux, 1, activation='sigmoid',
                           name='sortie_volume')(x)

    return keras.Model(inputs=[entree_tem, entree_exafs],
                       outputs=sortie,
                       name='ATOMOD_v2')


# =============================================================================
# SECTION 7 : Fonctions de perte personnalisées
# =============================================================================

def dice_loss(y_reel, y_pred, eps=1e-7):
    """
    Dice Loss multi-canaux pour la segmentation des positions atomiques.
    Minimise 1 - Dice, ce qui maximise le chevauchement entre prédit et réel.
    """
    axes = [1, 2]   # dimensions spatiales H et W
    intersection = tf.reduce_sum(y_reel * y_pred, axis=axes)
    somme        = tf.reduce_sum(y_reel + y_pred,  axis=axes)
    dice         = (2.0 * intersection + eps) / (somme + eps)
    return 1.0 - tf.reduce_mean(dice)


def perte_stoechiometrie(y_reel, y_pred, n_esp=N_ESPECES, n_z=N_Z_PLANS):
    """
    Contrainte de stœchiométrie globale : pénalise les déséquilibres de composition.
    Pour chaque espèce, compare la fraction atomique prédite à la vérité terrain.
    Les canaux sont organisés par bloc espèce : [esp0_z0..esp0_zN, esp1_z0..esp1_zN, ...]
    """
    pertes = []
    for i_esp in range(n_esp):
        debut = i_esp * n_z
        fin   = debut + n_z
        # Sélection des canaux de cette espèce (tous les plans Z)
        ch_reel = y_reel[..., debut:fin]
        ch_pred = y_pred[..., debut:fin]
        # Fraction atomique moyenne sur H, W et plans Z
        frac_reel = tf.reduce_mean(ch_reel, axis=[1, 2, 3])
        frac_pred = tf.reduce_mean(ch_pred, axis=[1, 2, 3])
        pertes.append(tf.reduce_mean(tf.abs(frac_reel - frac_pred)))
    return tf.add_n(pertes) / float(n_esp)


def perte_combinee(y_reel, y_pred, w_dice=0.7, w_stoech=0.3):
    """Perte combinée = w_dice × Dice Loss + w_stoech × Contrainte stœchiométrique."""
    return w_dice * dice_loss(y_reel, y_pred) + w_stoech * perte_stoechiometrie(y_reel, y_pred)


# =============================================================================
# SECTION 8 : Rééchantillonnage EXAFS
# =============================================================================

def reechantillonner_exafs(chemin_fichier, kmin=K_MIN, kmax=K_MAX, n=NPTS_SPECTRE):
    """
    Charge un fichier k³χ(k) FEFF et le rééchantillonne uniformément sur [kmin, kmax].

    Format attendu : 2 colonnes (k, k³χ(k)), lignes de commentaire commençant par '#'.
    Retourne un vecteur numpy de longueur n.
    """
    k_orig, y_orig = np.loadtxt(chemin_fichier, comments='#', usecols=(0, 1), unpack=True)

    if kmin < k_orig.min() or kmax > k_orig.max():
        raise ValueError(
            f"Intervalle [{kmin}, {kmax}] hors des bornes du fichier "
            f"[{k_orig.min():.3f}, {k_orig.max():.3f}] : {chemin_fichier}"
        )

    k_nouveau = np.linspace(kmin, kmax, n)
    y_nouveau = np.interp(k_nouveau, k_orig, y_orig)
    return y_nouveau.astype(np.float32)


# =============================================================================
# SECTION 9 : Générateur de données personnalisé
# =============================================================================

class GenerateurATOMOD(tf.keras.utils.Sequence):
    """
    Générateur de données on-the-fly pour l'entraînement d'ATOMOD v2.

    Charge à la volée pour chaque nanoparticule :
      - L'image TEM : simul/<id>/train/TEM/TEM.png
      - Les spectres EXAFS : simul/<id>/train/EXAFS/k3chi(k)_<ESPECE>.dat
      - Les cartes de probabilité : simul/<id>/train/prob_maps/img_0000_<ESP>_<plan>_*.png

    Retourne des lots ([batch_tem, batch_exafs], batch_volume).
    """

    def __init__(self, repertoire_simul, liste_ids, especes, n_z_plans,
                 H, W, npts_spectre, batch_size, global_max_abs,
                 melange=True, kmin=K_MIN, kmax=K_MAX):
        """
        Args:
            repertoire_simul : Chemin racine vers ./simul/
            liste_ids        : Liste des identifiants de nanoparticules à utiliser.
            especes          : Liste ordonnée des noms d'espèces.
            n_z_plans        : Nombre de plans Z par espèce.
            H, W             : Dimensions cibles de l'image TEM.
            npts_spectre     : Nombre de points EXAFS après rééchantillonnage.
            batch_size       : Taille d'un lot.
            global_max_abs   : np.array (N_ESPECES,) pour la normalisation Max-Abs.
            melange          : Mélange des données à chaque fin d'époque.
            kmin, kmax       : Plage de rééchantillonnage EXAFS.
        """
        self.repertoire = Path(repertoire_simul)
        self.liste_ids  = list(liste_ids)
        self.especes    = especes
        self.n_esp      = len(especes)
        self.n_z        = n_z_plans
        self.H          = H
        self.W          = W
        self.npts       = npts_spectre
        self.batch_size = batch_size
        self.gmax       = np.array(global_max_abs, dtype=np.float32)
        self.melange    = melange
        self.kmin       = kmin
        self.kmax       = kmax
        self.indices    = np.arange(len(self.liste_ids))
        if self.melange:
            np.random.shuffle(self.indices)
        print(f"[GÉNÉRATEUR] {len(self.liste_ids)} nanoparticule(s) | "
              f"{'train' if melange else 'val'} | {len(self)} lots/époque")

    def __len__(self):
        return int(np.floor(len(self.liste_ids) / self.batch_size))

    def __getitem__(self, idx):
        """Retourne le lot idx sous forme ([batch_tem, batch_exafs], batch_volume)."""
        indices_lot = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_tem    = np.zeros((self.batch_size, self.H, self.W, 1),          dtype=np.float32)
        batch_exafs  = np.zeros((self.batch_size, self.npts, self.n_esp),      dtype=np.float32)
        batch_volume = np.zeros((self.batch_size, self.H, self.W,
                                  self.n_esp * self.n_z),                       dtype=np.float32)

        for i, idx_nano in enumerate(indices_lot):
            pid  = self.liste_ids[idx_nano]
            base = self.repertoire / pid / 'train'

            batch_tem[i]    = self._charger_tem(base / 'TEM')
            batch_exafs[i]  = self._charger_exafs(base / 'EXAFS')
            batch_volume[i] = self._charger_prob_maps(base / 'prob_maps')

        return [batch_tem, batch_exafs], batch_volume

    def on_epoch_end(self):
        if self.melange:
            np.random.shuffle(self.indices)

    # ------------------------------------------------------------------
    # Méthodes de chargement privées
    # ------------------------------------------------------------------

    def _charger_tem(self, dossier_tem):
        """Charge TEM.png, convertit en niveaux de gris, normalise dans [0,1]."""
        chemin = Path(dossier_tem) / 'TEM.png'
        if not chemin.exists():
            # Fallback : premier .png trouvé dans le dossier
            candidats = list(Path(dossier_tem).glob('*.png'))
            if not candidats:
                raise FileNotFoundError(f"Aucune image TEM dans : {dossier_tem}")
            chemin = candidats[0]

        img = Image.open(chemin).convert('L')             # conversion directe en niveaux de gris
        img = img.resize((self.W, self.H), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[..., np.newaxis]                        # (H, W, 1)

    def _charger_exafs(self, dossier_exafs):
        """
        Charge k3chi(k)_<ESPECE>.dat, rééchantillonne et normalise Max-Abs.
        Retourne (NPTS, N_ESPECES).
        """
        matrice = np.zeros((self.npts, self.n_esp), dtype=np.float32)

        for i_esp, esp in enumerate(self.especes):
            chemin = Path(dossier_exafs) / f'k3chi(k)_{esp}.dat'
            if not chemin.exists():
                # Recherche permissive en cas de variante de nommage
                candidats = list(Path(dossier_exafs).glob(f'*k3*{esp}*.dat'))
                if not candidats:
                    print(f"[AVERTISSEMENT] EXAFS manquant pour {esp} dans {dossier_exafs}")
                    continue
                chemin = candidats[0]

            try:
                signal = reechantillonner_exafs(chemin, self.kmin, self.kmax, self.npts)
                max_abs = self.gmax[i_esp]
                if max_abs > 0:
                    signal = np.clip(signal / max_abs, -1.0, 1.0)
                matrice[:, i_esp] = signal
            except Exception as e:
                print(f"[ERREUR] {chemin} : {e}")

        return matrice

    def _charger_prob_maps(self, dossier_maps):
        """
        Charge les cartes de probabilité img_0000_<ESP>_<plan:04d>_*.png.
        Retourne (H, W, N_ESP * N_Z).
        """
        volume = np.zeros((self.H, self.W, self.n_esp * self.n_z), dtype=np.float32)

        for i_esp, esp in enumerate(self.especes):
            for i_plan in range(self.n_z):
                pattern  = f'img_0000_{esp}_{i_plan:04d}_*.png'
                fichiers = list(Path(dossier_maps).glob(pattern))

                if not fichiers:
                    continue   # canal laissé à zéro si le fichier est absent

                try:
                    img = Image.open(fichiers[0]).convert('L')
                    img = img.resize((self.W, self.H), Image.LANCZOS)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    canal = i_esp * self.n_z + i_plan
                    volume[:, :, canal] = arr
                except Exception as e:
                    print(f"[ERREUR] {fichiers[0]} : {e}")

        return volume


# =============================================================================
# SECTION 10 : Calcul du vecteur global_max_abs
# =============================================================================

def calculer_global_max_abs(repertoire_simul=REPERTOIRE_SIMUL,
                             especes=ESPECES,
                             kmin=K_MIN, kmax=K_MAX, npts=NPTS_SPECTRE):
    """
    Parcourt toutes les nanoparticules pour calculer le max absolu du signal k³χ(k)
    après rééchantillonnage, par espèce.

    À exécuter une fois avant l'entraînement pour calibrer la normalisation Max-Abs.
    Retourne np.array de forme (N_ESPECES,).
    """
    print("[NORMALISATION] Calcul du global_max_abs sur l'ensemble du jeu de données...")
    rep     = Path(repertoire_simul)
    n_esp   = len(especes)
    max_abs = np.zeros(n_esp, dtype=np.float64)

    for dossier in sorted(rep.iterdir()):
        if not dossier.is_dir():
            continue
        rep_exafs = dossier / 'train' / 'EXAFS'
        if not rep_exafs.exists():
            continue

        for i_esp, esp in enumerate(especes):
            chemin = rep_exafs / f'k3chi(k)_{esp}.dat'
            if not chemin.exists():
                continue
            try:
                signal  = reechantillonner_exafs(chemin, kmin, kmax, npts)
                mx_loc  = np.max(np.abs(signal))
                if mx_loc > max_abs[i_esp]:
                    max_abs[i_esp] = mx_loc
            except Exception as e:
                print(f"[ERREUR] {chemin} : {e}")

    max_abs[max_abs == 0] = 1.0   # évite la division par zéro pour les espèces sans données
    print(f"[NORMALISATION] global_max_abs = {max_abs}")
    return max_abs.astype(np.float32)


# =============================================================================
# SECTION 11 : Découverte et découpage train / validation
# =============================================================================

def decouvrir_et_decouper(repertoire_simul=REPERTOIRE_SIMUL,
                           ratio_val=RATIO_VALIDATION,
                           graine=GRAINE_ALEATOIRE):
    """
    Liste les identifiants de nanoparticules valides et les divise en train/val.
    Retourne (ids_train, ids_val).
    """
    rep = Path(repertoire_simul)
    ids_valides = sorted([
        d.name for d in rep.iterdir()
        if d.is_dir()
        and (d / 'train' / 'TEM').exists()
        and (d / 'train' / 'EXAFS').exists()
        and (d / 'train' / 'prob_maps').exists()
    ])

    if not ids_valides:
        raise FileNotFoundError(
            f"Aucune nanoparticule valide trouvée dans {repertoire_simul}. "
            "Vérifiez la structure : simul/<id>/train/{{TEM,EXAFS,prob_maps}}/"
        )

    rng = np.random.default_rng(graine)
    perm = rng.permutation(len(ids_valides))
    n_val = max(1, int(len(ids_valides) * ratio_val))

    ids_val   = [ids_valides[i] for i in perm[:n_val]]
    ids_train = [ids_valides[i] for i in perm[n_val:]]

    print(f"[DONNÉES] {len(ids_valides)} nanoparticules | "
          f"train : {len(ids_train)} | val : {len(ids_val)}")
    return ids_train, ids_val


# =============================================================================
# SECTION 12 : Pipeline d'entraînement principal
# =============================================================================

def lancer_entrainement(
    repertoire_simul=REPERTOIRE_SIMUL,
    ratio_val=RATIO_VALIDATION,
    recalculer_max_abs=False,
    global_max_abs=None
):
    """
    Pipeline complet d'entraînement ATOMOD v2.

    Args:
        repertoire_simul   : Chemin vers ./simul/
        ratio_val          : Fraction de nanoparticules réservée à la validation.
        recalculer_max_abs : Si True, recalcule global_max_abs depuis les données.
        global_max_abs     : Vecteur prédéfini (ignoré si recalculer_max_abs=True).
    Returns:
        (modele, historique)
    """
    # --- Normalisation EXAFS ---
    if recalculer_max_abs:
        gmax = calculer_global_max_abs(repertoire_simul)
    elif global_max_abs is not None:
        gmax = np.array(global_max_abs, dtype=np.float32)
    else:
        gmax = GLOBAL_MAX_ABS

    # --- Découpage train / val ---
    ids_train, ids_val = decouvrir_et_decouper(repertoire_simul, ratio_val)

    if len(ids_train) < BATCH_SIZE:
        raise ValueError(
            f"Jeu d'entraînement trop petit ({len(ids_train)} nanoparticules) "
            f"pour un batch_size={BATCH_SIZE}."
        )

    # --- Générateurs ---
    kw_gen = dict(
        repertoire_simul=repertoire_simul,
        especes=ESPECES, n_z_plans=N_Z_PLANS,
        H=HEIGHT_IMAGE_TEM, W=WIDTH_IMAGE_TEM,
        npts_spectre=NPTS_SPECTRE, batch_size=BATCH_SIZE,
        global_max_abs=gmax, kmin=K_MIN, kmax=K_MAX
    )
    train_gen = GenerateurATOMOD(**kw_gen, liste_ids=ids_train, melange=True)
    val_gen   = GenerateurATOMOD(**kw_gen, liste_ids=ids_val,   melange=False)

    # --- Construction du modèle ---
    print("[MODÈLE] Construction d'ATOMOD v2...")
    modele = construire_atomod_v2()
    modele.summary()

    # --- Compilation ---
    modele.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TAUX_APPRENTISSAGE),
        loss=perte_combinee,
        metrics=['mae']
    )

    # --- Callbacks ---
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE_ARRET,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=CHEMIN_MODELE,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=PATIENCE_REDUCE_LR,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger('historique_atomod.csv', append=True),
        keras.callbacks.TensorBoard(log_dir='./logs_atomod', histogram_freq=0),
    ]

    # --- Entraînement ---
    print(f"[ENTRAÎNEMENT] Démarrage sur {DISPOSITIF} | {N_EPOQUES} époques max...")
    historique = modele.fit(
        train_gen,
        validation_data=val_gen,
        epochs=N_EPOQUES,
        callbacks=callbacks,
        verbose=1
    )

    print(f"[TERMINÉ] Meilleur modèle → {CHEMIN_MODELE}")
    return modele, historique


# =============================================================================
# SECTION 13 : Visualisation de l'historique
# =============================================================================

def tracer_historique(historique, chemin_sortie='historique_atomod.png'):
    """Trace et sauvegarde les courbes de perte et de MAE."""
    h = historique.history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(h['loss'],     label='Train')
    axes[0].plot(h['val_loss'], label='Validation')
    axes[0].set_title('Perte (Dice + Stœchiométrie)')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Perte')
    axes[0].legend()
    axes[0].grid(True)

    if 'mae' in h:
        axes[1].plot(h['mae'],     label='Train')
        axes[1].plot(h['val_mae'], label='Validation')
        axes[1].set_title('MAE')
        axes[1].set_xlabel('Époque')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    print(f"[VISU] Courbes sauvegardées → {chemin_sortie}")


# =============================================================================
# SECTION 14 : Point d'entrée principal
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  ATOMOD v2 — Reconstruction Tomographique 3D Implicite HEA")
    print("  Projet M2P2_HEA (PEPR DIADEM)")
    print(f"  Dispositif : {DISPOSITIF}")
    print("=" * 70)

    # Mettre recalculer_max_abs=True lors de la première exécution pour calibrer
    # la normalisation EXAFS depuis les données réelles.
    modele, historique = lancer_entrainement(
        repertoire_simul=REPERTOIRE_SIMUL,
        ratio_val=RATIO_VALIDATION,
        recalculer_max_abs=True   # ← passer à False après la première exécution
    )

    tracer_historique(historique)
