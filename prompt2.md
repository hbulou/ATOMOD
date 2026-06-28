---
type: Type
---
# Prompt
Tu es un expert en Deep Learning appliqué à la physique des matériaux et à la microscopie. Je travaille sur le projet M2P2_HEA (PEPR DIADEM) pour développer le modèle "ATOMOD". L'objectif est d'effectuer une reconstruction tomographique 3D implicite d'une nanoparticule d'alliage à haute entropie (HEA) à partir de deux entrées multimodales  : une unique image TEM 2D et des spectres d'absorption des rayons X (EXAFS).

Génère un script Python complet utilisant TensorFlow 2.15+ / Keras qui implémente l'ensemble de l'architecture et du pipeline d'entraînement selon les spécifications strictes suivantes :

1. FORMAT DES DONNÉES ET ENTRÉES :
- Entrée 1 (Image TEM  de dimensions HEIGHT_IMAGE_TEM x WIDTH_IMAGE_TEM) : Tensor de forme (BATCH_SIZE, HEIGHT_IMAGE_TEM, WIDTH_IMAGE_TEM, 1).
- Entrée 2 (Spectres EXAFS dans l'espace k) : Matrice de forme (BATCH_SIZE, NPTS_SPECTRE, N_ESPECES). Le signal correspond au chi(k) pondéré en k^3 pour N_ESPECES chimiques (ex: Au, Co, Pt, Rh, Pd).
- Sortie Cible (Volume 3D) : Grille de probabilité de présence atomique de forme (BATCH_SIZE, HEIGHT_IMAGE_TEM, WIDTH_IMAGE_TEM, N_ESPECES * N_Z_PLANS), où les plans Z de chaque espèce sont concaténés sur le canal final.

2. ARCHITECTURE DU MODÈLE (ATOMOD v2) :
- Encodeur EXAFS : Un réseau convolutif 1D (CNN-1D) prenant la forme (NPTS_SPECTRE, N_ESPECES). Utilise des filtres de taille décroissante (kernel_size=11, puis 7, puis 3) avec BatchNormalization pour capturer les oscillations de diffusion. Termine par un GlobalAveragePooling1D et une couche Dense pour projeter le signal dans un vecteur latent global de taille 128 (représentant l'empreinte chimique).
- Réseau Principal (UNet 2D) : Une architecture de type UNet prenant l'image TEM en entrée.
- Conditionnement Multimodal (FiLM) : Intègre des blocs FiLM (Feature-wise Linear Modulation) après chaque bloc d'encodage du UNet. Ces blocs FiLM doivent prédire des paramètres gamma (échelle) et beta (décalage) à partir du vecteur latent EXAFS pour moduler les cartes de caractéristiques du UNet.

3. PIPELINE DE DONNÉES (Custom Data Generator) :
- Implémente une classe héritant de `tf.keras.utils.Sequence`.
- Le générateur doit charger à la volée (on-the-fly) les fichiers `.png` pour le TEM et les volumes 3D, et les fichiers `.dat` de FEFF pour l'EXAFS.
- Les données pour l'entraînement sont stockées dans des répertoires dont la structure est ./simul/<numero nanoparticule>/train/TEM qui est le répertoire où est stockée l'image TEM.png,./simul/<numero nanoparticule>/train/EXAFS où sont stockés les N_ESPECES fichiers k3chi(k)_<ESPECES>.dat et ./simul/<numero nanoparticule>/train/prob_maps où sont stockés les N_ESPECES*N_Z_PLANS fichiers img_000_<ESPECE>_<numéro plan>_*.png
- Intègre une étape de normalisation Max-Abs indépendante pour chaque colonne (espèce) des spectres EXAFS en utilisant un vecteur `global_max_abs` prédéfini.
- Gère le mélange des données (shuffle=True) uniquement pour l'entraînement.

4. LOGIQUE DE CONFIGURATION MATÉRIELLE :
- Incluts une fonction de détection automatique en début de script qui vérifie la Compute Capability du GPU via `tf.config.experimental.get_device_details`. Si la Compute Capability est inférieure à 6.0 (ex: Nvidia Quadro M1000M), elle doit automatiquement définir `os.environ["CUDA_VISIBLE_DEVICES"] = ""` et basculer proprement sur le CPU pour éviter l'erreur 'CUDA_ERROR_NO_BINARY_FOR_GPU'.

5. COMPILATION ET ENTRAÎNEMENT :
- Compile le modèle avec l'optimiseur 'adam' et une fonction de perte personnalisée combinant une Dice Loss (pour la segmentation des positions atomiques) et une contrainte de stœchiométrie globale.
- Configure l'appel à `model.fit` en lui passant le train_generator, le validation_generator, N_EPOQUES époques, et les callbacks standards (EarlyStopping, ModelCheckpoint).

Fournis un code Python propre, modulaire, commenté en français, prêt à être exécuté dans un notebook Jupyter ou en Stand-Alone.
