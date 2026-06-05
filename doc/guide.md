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
