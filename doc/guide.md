---
type: Note
---
# Guide de développement

**fusion de données multimodale** (Image TEM + Spectroscopie EXAFS) pour résoudre un problème de reconstruction 3D (plans atomiques) et de résolution chimique (Problème de la **reconstruction 3D inverse).**

L'idée est de reconstruire une structure 3D à partir d'informations projetées 2D (TEM) et intégrées statistiquement (EXAFS global) à une carte atomique 3D discrète d'un alliage à haute entropie (HEA), où le désordre chimique est maximal.

***

### 1. Stratégie d'Architecture : Le "Conditionnement Interne"

Deux approches possibles :

- la "Late Fusion" (fusion tardive) où l'on colle les résultats à la fin. On traite chaque modalité (image et spectre) de manière totalement indépendante jusqu'à la toute fin du réseau.
  - Le mécanisme : il faut un encodeur pour l'image (UNet) et un encodeur pour le spectre. Ils produisent chacun un vecteur ou une carte de caractéristiques. On les concatènes juste avant la dernière couche de décision.
  - Analogie : Deux experts travaillent dans des pièces séparées, écrivent chacun un rapport, et un troisième expert lit les deux rapports à la fin pour prendre une décision.
  - Inconvénient : L'information EXAFS arrive "trop tard" pour aider le UNet à décider où placer les atomes pendant qu'il analyse les formes dans l'image.
- Le conditionnement : L'information d'une modalité (l'EXAFS) vient influencer ou "moduler" la manière dont l'autre modalité (l'image) est traitée **pendant** le processus de calcul. Approche absée sur une architecture de type **UNet 2D** conditionnée par un **Encodeur EXAFS**. Elle permet que l'information spectroscopique **guide** la reconstruction d'image tout au long du processus. L'image TEM donne les positions $x,y$, l'EXAFS donne les distances locales pour la dimension $z$ et la chimie.
  - Le mécanisme : À chaque étage de UNet, on injecte l'information EXAFS. Le spectre dit au réseau : "Hé, je détecte beaucoup de coordination de type Platine, donc quand tu regardes l'image, sois plus attentif aux contrastes forts à cet étage."
  - Technique phare (FiLM) : Le spectre génère des coefficients qui multiplient et s'ajoutent aux cartes de caractéristiques de l'image ($y = \gamma \cdot x + \beta$).
  - Analogie : Un expert regarde l'image TEM pendant qu'un deuxième expert (l'EXAFS) lui chuchote des indices à l'oreille en temps réel.

### Comparaison pour le projet ATOMOD

|                          |                                                                           |                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Caractéristique**      | **Fusion Tardive**                                                        | **Conditionnement (Recommandé)**                                                                |
| **Complexité**           | Simple à coder.                                                           | Plus complexe (nécessite de modifier les blocs du UNet).                                        |
| **Synergie**             | Faible (simple juxtaposition).                                            | Forte (le spectre guide la vision).                                                             |
| **Pertinence TEM/EXAFS** | Le spectre ne sert qu'à "étiqueter" les formes déjà trouvées par le UNet. | Le spectre aide le UNet à "voir" des plans atomiques qui seraient invisibles sur l'image seule. |

### Pourquoi le Conditionnement est la meilleure approche pour ATOMOD ?

L'image TEM est une projection 2D (tous les plans sont écrasés). Le spectre EXAFS contient, de manière cachée, l'information sur la profondeur et l'environnement chimique.

Si on faits une **Fusion Tardive**, le UNet risque de segmenter ce qu'il "voit" en 2D, et le spectre essaiera de deviner la chimie globalement.

Si on fait un **Conditionnement**, le spectre peut forcer le UNet à activer certains canaux de sortie (certains plans atomiques $z$) qu'il aurait ignorés autrement.

#### Branche A : Encodeur EXAFS (MLP ou CNN 1D)

- **Entrée :** Une matrice de spectres globaux (ex: taille $(Points_K \times N_{especes})$).
- **Rôle :** Compresser ce signal hautement dimensionnel en un vecteur "latent" compact (ex: 128 ou 256 valeurs) qui représente l' "empreinte digitale" chimique et structurale locale de la particule.

#### Branche B : UNet de Reconstruction guide (Image TEM)

C'est un UNet standard, mais ses blocs de convolution doivent être modifiés pour accepter le vecteur latent EXAFS.

- **Technique recommandée : FiLM (Feature-wise Linear Modulation).** À chaque étage de l'encodeur et du décodeur du UNet, tu utilises le vecteur latent EXAFS pour générer des paramètres $\gamma$ et $\beta$ qui vont normaliser et décaler les cartes de caractéristiques (*feature maps*) de l'image.
- *Formule :* $FiLM(F) = \gamma(Latent_EXAFS) \cdot F + \beta(Latent_EXAFS)$
- **Pourquoi ?** Cela permet au spectre de dire au UNet : *"Pour cette particule, je vois beaucoup de Platine et peu de Cobalt, donc quand tu décode, active plus fortement les canaux de sortie du Platine."*

***

### 2. Structure de Sortie (Représentation de la vérité terrain)

C'est le point critique pour un HEA. Puisqu'il y a trop d'atomes pour que le réseau sorte une liste de coordonnées, l'idée est de générer un **volume binaire discret segmenté chimique par chimique.**

Si on a $N_{sp}$ espèces chimiques et que les particules tiennent dans un volume de $N_{pz}$ plans atomiques d'épaisseur :

- **Format de sortie du réseau :** Un tenseur de forme $(H \times W \times (N_{sp} \times N_{pz}))$.
- **Interprétation :** Le réseau sort $(N_{sp} \times N_{pz})$ canaux binaires (ou probabilités sigmoid).
- *Canal 0 :* Probabilité de présence de l'Espèce 1 au Plan $z=1$.
- *Canal 1 :* Probabilité de présence de l'Espèce 1 au Plan $z=2$.
- ...
- *Canal $N_{pz}$ :* Probabilité de présence de l'Espèce 2 au Plan $z=1$.

Il est donc nécessaire de pré-traiter les positions atomiques *in silico* pour les discrétiser sur une grille voxel correspondant à cette structure de sortie.

***

### 3. Gestion des Données et Augmentation

Le réseau va sur-apprendre très vite sur des données simulées. Il est donc nécessaire de rendre les simulations "sales" pour qu'elles ressemblent à des données expérimentales.

#### Pour l'Image TEM :

1. **Fonction de Transfert de Contraste (CTF) :** Appliquer une CTF réaliste (défocalisation, aberrations sphériques) pour flouter l'image comme un vrai microscope.
2. **Bruit de Poisson :** Ajouter du bruit de comptage d'électrons.
3. **Augmentation Géométrique :** Rotations, translations, légers cisaillements des couples (Image+Volume).

#### Pour les Spectres EXAFS :

1. **Bruit :** Ajouter du bruit blanc gaussien réaliste sur le signal $\chi(k)$ avant la Transformée de Fourier.

***

### 4. Fonctions de Perte (Loss Functions) avec Contraintes Physiques

Pour un HEA, une simple `binary_crossentropy` ne suffira pas car le volume est très vide (beaucoup de zéros). Il est donc nécessaire d'utiliser une perte composée :

$$Loss_{totale} = Dice + \lambda_{stoich} \cdot Stoich + \lambda_{inter} \cdot Interface$$

1. **Déséquilibre de classe (Dice Loss) :** Utilise la **Soft Dice Loss** ou la **Focal Loss** sur les $(N_{sp} \times N_{pz})$ canaux. Cela force le réseau à se concentrer sur les pixels où il y a des atomes, plutôt que sur le vide autour de la particule.
2. **Contrainte Physique 1 : Stoechiométrie globale.** Si tes spectres EXAFS globaux te donnent la fraction atomique de chaque espèce (ex: CoCrFeMnNi 20/20/20/20/20), ajoute un terme qui pénalise la somme des probabilités de chaque espèce si elle s'éloigne de cette fraction.
3. **Contrainte Physique 2 : Exclusion d'interface (Hard Constraint).** Au post-traitement (ou via une loss très forte), impose qu'un pixel $(x,y,z)$ donné ne peut contenir qu'**une seule** espèce chimique. $\sum_{species} P_{i,x,y,z} \leq 1$.

### En résumé

1. **Modèle :** UNet 2D guide par FiLM via un encodeur EXAFS.
2. **Sortie :** Volume de voxels discrétisés $(H, W, N_{sp} \times N_{pz})$.
3. **Simulations :** Bruite massivement tes images TEM (CTF + Poisson) et EXAFS.
4. **Loss :** Dice Loss + Contraintes de stoechiométrie.


### Exemple de prompt à soumettre àç une IA générative

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
