# Projet ATOMOD
ATOMOD est un environnement numérique dans lequel vous trouverez une série de logiciels et de scripts python permettant de générer des nanoparticules, d'en calculer les images TEM, les spectres EXAFS, de déterminer la composition idéal d'une nanoparticule en fonction de propriétés catalytiques souhaitées (modèle REACT2COMPO à venir), de déterminer la structure 3D d'une nanoparticule à partir d'images TEM et de spectres EXAFS (modèle ATOMOD), de détecter des amas à partir d'images TEM, etc.


## 1. Installation d'ATOMOD

L'installation d'ATOMOD se fait au moyen du script python 
[install.py](https://github.com/hbulou/ATOMOD/releases/download/v0.1/install.py)



### 2. Récupérer les fichiers du projet ATOMOD

Vous avez le choix entre quatre méthodes pour installer l'environnement ATOMOD :

<details>
  <summary>1. Télécharger sans utiliser GitHub (Archive ZIP)</summary>

* (a). Allez sur la page du dépôt [ATOMOD](https://github.com/hbulou/ATOMOD).
* (b). Cliquez sur **Download ZIP**.
* (c).  Dans votre terminal, placez-vous là où vous voulez mettre le projet (Par exemple `~/src`) et décompressez le fichier `ATOMOD-main.zip`. Vous devriez avoir un nouveau répertoire `ATOMOD-main`.

<img width="900" alt="image" src="https://github.com/user-attachments/assets/61958c5a-2257-48ec-bd23-513842970727" />

</details>


<details><summary>2. Télécharger en utilisant GitHub (recommandé)</summary>

<details>
  <summary>&emsp;(a). La méthode classique (SSH)</summary>
    
Si vous avez configuré une clé SSH (ce qui est fortement recommandé), c'est la méthode la plus simple et la plus rapide (voir [Comment configurer une clef SSH ?](https://github.com/hbulou/ATOMOD/wiki/Configurer-une-cl%C3%A9-SSH)).

1. Sur GitHub, allez sur la page du dépôt [ATOMOD](https://github.com/hbulou/ATOMOD).
2. Cliquez sur le bouton vert **Code**.
3. Vérifiez que l'onglet **SSH** est sélectionné et copiez l'adresse `git@github.com:hbulou/ATOMOD.git`.
4. Dans votre terminal, placez-vous là où vous voulez mettre le projet (Par exemple `~/src`). et tapez :
```bash
git clone git@github.com:hbulou/ATOMOD.git
```
<img width="900" alt="image" src="https://github.com/user-attachments/assets/d0b9c4bf-047f-4876-9ba2-65345a733ebf" />


</details>
<details>
  <summary>&emsp;(b). Si vous n'avez pas de clé SSH (HTTPS)</summary>
  Si vous êtes sur un ordinateur tiers où votre clé n'est pas installée, utilisez l'adresse HTTPS.

  1. Sur GitHub, allez sur la page du dépôt [ATOMOD](https://github.com/hbulou/ATOMOD).
  2. Cliquez sur le bouton vert **Code**.
  3. Vérifiez que l'onglet **HTTPS** est sélectionné et copiez l'adresse `https://github.com/hbulou/ATOMOD.git`.
  4. Dans votre terminal, placez-vous là où vous voulez mettre le projet (Par exemple `~/src`). et tapez :
```bash
git clone https://github.com/hbulou/ATOMOD.git

```

*Note : GitHub vous demandera alors votre nom d'utilisateur et votre **Personal Access Token** (pas votre mot de passe).*

<img width="900" alt="image" src="https://github.com/user-attachments/assets/61958c5a-2257-48ec-bd23-513842970727" />

---
</details>

</details>

### 3. Installer l'environnement vituel "ATOMOD"
L'environnement virtuel ATOMOD est un espace de travail isolé sur votre ordinateur dédiée au projet ATOMOD. Il contient toutes les bibliothèques nécéssaires à l'exécution d'ATOMOD.
<details>
<summary> Linux </summary>

```bash
cd
python3 -m venv venv/ATOMOD   # création de l'environnement - répertoire ~/venv/ATOMOD
source venv/ATOMOD/bin/activate  # activation de l'environnement
pip install --upgrade pip                   # màj de pip


pip install jupyterlab
pip install numpy
pip install "scipy<1.17"
pip install matplotlib
pip install mace-torch
pip install dask            #  Dask est une bibliothèque de calcul parallèle et de gestion des grandes masses de données (Big Data) en Python.
pip install tabulate
pip install numba
pip install threadpoolctl
pip install zarr
pip install ipywidgets
pip install pyfftw

cd ; mkdir -p src ; cd src
wget https://github.com/hbulou/site-packages/archive/refs/heads/main.zip -O HBPy.zip
unzip HBPy.zip ; rm HBPy.zip ; mv site-packages-main HBPy
cd HBPy ; pip install -e .
cd ..
wget https://github.com/hbulou/ATOMOD/archive/refs/heads/main.zip -O ATOMOD-main.zip
unzip ATOMOD-main.zip ; rm ATOMOD-main.zip ; mv ATOMOD-main ATOMOD


cd ATOMOD/doc/tutorials
jupyter lab tuto-TEM_image_simulation.ipynb 



```

</details>

<details>
<summary>Windows</summary>
Il est fortement conseillé de ne pas installer les bibliothèques globalement pour éviter les conflits de versions.
<li>Ouvrez le Terminal Windows (clic droit sur le bouton Démarrer).</li>
<li>Naviguez vers le dossier de votre projet :</li>

```
cd "C:\chemin\vers\votre\dossier"
```

<li>Créez l'environnement :</li>

```
py -m venv venv
```

<li>Activez l'environnement :</li>

```
.\venv\Scripts\activate
```

<li>Lancer l'installation groupée</li>

```
pip install -r requirements.txt
```
</details>


### 1. Récupérer les fichiers de la bibliothèque HBPy
Pour récupérer les fichiers de la bibliothèque HBPy, on utilise l'opération de **clonage**. Cela crée une copie locale identique au projet distant, incluant tout l'historique des modifications.

Vous avez le choix entre quatre méthodes :

<details>
  <summary>1. La méthode classique (SSH)</summary>
    
Si vous avez configuré une clé SSH (ce qui est fortement recommandé), c'est la méthode la plus simple et la plus rapide (voir [Comment configurer une clef SSH ?](https://github.com/hbulou/ATOMOD/wiki/Configurer-une-cl%C3%A9-SSH)).

1. Sur GitHub, allez sur la page du dépôt [HBPy](https://github.com/hbulou/site-packages).
2. Cliquez sur le bouton vert **Code**.
3. Vérifiez que l'onglet **SSH** est sélectionné et copiez l'adresse `git@github.com:hbulou/ATOMOD.git`.
4. Dans votre terminal, placez-vous là où vous voulez mettre le projet (Par exemple `~/src`). et tapez :
```bash
git clone git@github.com:hbulou/site-packages.git
```
<img width="900" alt="image" src="https://github.com/user-attachments/assets/59a50e8b-d70e-42fd-8661-6ca796d64e4d" />


</details>
<details>
  <summary>2. Si vous n'avez pas de clé SSH (HTTPS)</summary>
  Si vous êtes sur un ordinateur tiers où votre clé n'est pas installée, utilisez l'adresse HTTPS.

  1. Sur GitHub, allez sur la page du dépôt [HBPy](https://github.com/hbulou/site-packages).
  2. Cliquez sur le bouton vert **Code**.
  3. Vérifiez que l'onglet **HTTPS** est sélectionné et copiez l'adresse `https://github.com/hbulou/ATOMOD.git`.
  4. Dans votre terminal, placez-vous là où vous voulez mettre le projet (Par exemple `~/src`). et tapez :
```bash
git clone https://github.com/hbulou/ATOMOD.git

```

*Note : GitHub vous demandera alors votre nom d'utilisateur et votre **Personal Access Token** (pas votre mot de passe).*

<img width="900" alt="image" src="https://github.com/user-attachments/assets/11066d0d-702b-444d-8fda-54262d42c0a9" />


</details>
<details>
  <summary>3. Télécharger sans utiliser Git (Archive ZIP)</summary>
Si vous voulez juste les fichiers sans l'historique Git (pour une consultation rapide par exemple) :

1. Sur GitHub, allez sur la page du dépôt [HBPy](https://github.com/hbulou/site-packages).
2. Cliquez sur **Download ZIP**.
3.  Dans votre terminal, placez-vous là où vous voulez mettre le projet (Par exemple `~/src`). et décompressez le fichier sur votre ordinateur.

<img width="900" alt="image" src="https://github.com/user-attachments/assets/11066d0d-702b-444d-8fda-54262d42c0a9" />



</details>
<details>
  <summary> 4. Mettre à jour un dépôt déjà cloné</summary>
Si vous avez déjà cloné le dépôt il y a quelque temps et que vous voulez récupérer les dernières modifications faites sur GitHub (par exemple après avoir édité un fichier directement via l'interface web), utilisez :

```bash
git pull origin main

```


</details>

### 2. Installer la biliothèque HBPy

1. Placez vous dans le répertoire dans lequel la bibliothèque HBPy a été installée (par exemple `~/src/site-packages-main` si vous avez utilisé "Downlad ZIP" ou  `~/src/site-packages` autrement)
```bash
cd ; cd ~/src/site-packages
```
2. charger l'environnement ATOMOD. 
```sh
source ~/venv/ATOMOD/bin/activate
```
2. Installer la biliothèque HBPy dans l'environnement ATOMOD
```bash
pip install -e .
```
## 2. Lancer ATOMOD
L'installation d'ATOMOD n'est à faire qu'une seule fois. Ensuite, vous pouvez utiliser l'environnement ATOMOD pour lancer des scripts python permettant de générer des nanoparticules, d'en calculer les images TEM, les spectres EXAFS, de déterminer la composition idéal d'une nanoparticule en fonction de propriétés catalytiques souhaitées (modèle REACT2COMPO à venir), de déterminer la structure 3D d'une nanoparticule à partir d'images TEM et de spectres EXAFS (modèle ATOMOD), de détecter des amas à partir d'images TEM, etc.
### 2.1 Tutoriels
C'est également dans cet environnement que vous pourrez exécuter les scripts `jupyter` fournis dans le répertoire `./doc/tutorials`
* Allez dans le répertoire ATOMOD. Par exemple, si vous avez choisit le répertoire `~/src` dans la section "Récupérer les fichiers du projet ATOMOD", vous devriez avoir un répertoire `~/src/ATOMOD`
```sh
cd ~/src/ATOMOD/doc/tutorials
```
* charger l'environnement ATOMOD. 
```sh
source ~/venv/ATOMOD/bin/activate
# ou
source ~/venv/ATOMOD_gpu/bin/activate
```
selon que vous disposez de `GPU` ou pas.
* tapez la commande `jupyter lab`.

`JupyterLab` s'ouvre dans votre navigateur, et vous voyez instantanément tous vos scripts Python et fichiers `.ipynb` issus de GitHub dans la colonne de gauche. Vous pouvez les exécuter, les modifier et les tester normalement.

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/d797f407-4052-4e34-a196-13ea65c1f56e" />

<!--
RQ : sur les serveurs de calcul, il peut être nécessaire de charger le module python au préalable
```sh
module load python
```

-----------------------------------------------------------------------------------------------------
### 2.1. sur hpc
```bash
cd
module load python
python3 -m venv venv/ATOMOD
source venv/ATOMOD/bin/activate
cd workdir/ATOMOD
pip install torch
pip install tensorflow[and-cuda]
pip install opencv-python
```

```bash
cd /home/bulou/ownCloud/Notebooks/M2P2_HEA/Modelisation/ATOMOD/GUI
source /home/bulou/venv/ATOMOD/bin/activate
python Py_ATOMODv0.5.py
```

```bash
pyuic6 HB_ATOMOD_GUI.ui -o HB_ATOMOD_GUI.py
```
-----------------------

### 2.2. Utilisation de MACE (Multi-Atomic Cluster Expansion)

MACE (Multi-Atomic Cluster Expansion) est une architecture de potentiel interatomique basé sur l'apprentissage automatique (Machine Learning Interatomic Potential, MLIP).

MACE est un réseau de neurones de type "message-passing" (MPNN) qui apprend à prédire l'énergie potentielle et les forces d'un système atomique à partir de sa géométrie, en s'appuyant sur la théorie de l'"Atomic Cluster Expansion" (ACE).
Il combine :
* La représentation ACE : une base de descripteurs atomiques systématique, complète et invariante par rotation/translation/permutation.
* Les réseaux de neurones équivariants : les messages échangés entre atomes portent des informations vectorielles et tensorielles (pas seulement scalaires), ce qui permet de capturer des interactions directionnelles complexes.

Des versions pré-entraînées (modèles universels) couvrent une grande partie du tableau périodique. Modèles universels notables
* MACE-MP-0 : potentiel universel entraîné sur la base de données Materials Project, capable de simuler la plupart des matériaux inorganiques sans réentraînement.
* MACE-OFF : variante pour les molécules organiques.


**Caractéristiques clés**

| Propriété       | Détail | 
|-----------------|--------------|
|Équivariance     | Respecte les symétries physiques (rotation, translation, permutation)|
| Ordre de l'interaction | Interactions à plusieurs corps (many-body) via ACE |
| Précision              |Comparable aux méthodes DFT pour un coût bien moindre | 
| Efficacité | Très compétitif en termes de vitesse d'entraînement et d'inférence | 
| Généralisation | Bonne capacité à extrapoler hors des données d'entraînement|

#### 2.2.1. MACE dans ATOMOD
#### 2.2.1.1. Installation

**Chargement du modèle**

```bash
source ~/venv/ATOMOD/bin/activate
pip install --upgrade pip
pip install mace-torch
CUDA_VISIBLE_DEVICES="" python -c "from mace.calculators import mace_mp; mace_mp(model='medium')"
```

La commande ```python from mace.calculators import mace_mp``` est une commande de vérification d'installation et de pré-téléchargement du modèle. Elle ne fait aucun calcul physique.
Ell charge  le module Python de MACE dans la mémoire et vérifie que l'installation est correcte.
La commande ```python mace_mp(model='medium')```
* Télécharge le fichier ```bash 2023-12-03-mace-128-L1_epoch-199.model``` depuis [GitHub](https://github.com/ACEsuit/mace) si absent
* Le sauvegarde dans ```bash ~/.cache/mace/``` pour ne pas le retélécharger à chaque fois
* Initialise le calculateur (charge les poids du réseau de neurones)
* Détecte le device disponible (GPU/CPU)

Deux approches peuvent être utilisées dans ATOMOD pour optimiser la structure des nanoparticules d'HEA : ASE ou LAMMPS.
LAMMPS est conçu pour tourner sur des centaines de cœurs en parallèle alors que ASE est mono-processus. Aussi, pour de petits systèmes (< 5000 atomes) ou des simulations courtes (quelques centaines de ps), l'approche ASE est suffisante.
Dans le cas contraire, il vaut mieux utiliser LAMMPS.


**Si vous voulez utiliser ASE**

On utilise le modèle via la bibliothèque ASE.
```python
    from mace.calculators import mace_mp
    import ase
    import numpy as np
    model_path = '/home/bulou/.cache/mace/20231203mace128L1_epoch199model'

    calc = mace_mp(model=model_path,   # chemin explicite - model='medium',
                   device='cpu',
                   default_dtype='float32')
    # Charger la NP depuis le fichier XYZ sauvegardé à l'étape 1.2
    atoms = ase.io.read('NP.xyz')
    logger.info(f"Structure chargée : {len(atoms)} atomes")
    logger.info(f"Composition : {atoms.get_chemical_formula()}")

    # Boîte de simulation avec vide autour de la NP (nécessaire pour MACE)
    atoms.center(vacuum=10.0)

    # Attacher le calculateur
    atoms.calc = calc

    # Energie avant minimisation
    e_avant = atoms.get_potential_energy()
    logger.info(f"Energie avant minimisation : {e_avant:.4f} eV")
    logger.info(f"Soit {e_avant/len(atoms):.4f} eV/atome")
    # Minimisation LBFGS
    logger.info("Démarrage de la minimisation...")
    traj_file = 'NP_minimisation.traj'
    opt = ase.optimize.LBFGS(atoms, trajectory=traj_file, logfile='minimisation.log')
    opt.run(fmax=0.05)   # convergence à 0.05 eV/Å sur les forces

    # Energie après minimisation
    e_apres = atoms.get_potential_energy()
    logger.info(f"Energie après minimisation  : {e_apres:.4f} eV")
    logger.info(f"Soit {e_apres/len(atoms):.4f} eV/atome")
    logger.info(f"Relaxation : {e_avant - e_apres:.4f} eV")

    # Sauvegarder la structure relaxée
    ase.io.write('NP_relaxed.xyz', atoms)
    logger.info("Structure relaxée sauvegardée dans NP_relaxed.xyz")
```


Boucle d'optimisation :
```bash
ASE (LBFGS)
    │
    ├── demande énergie + forces
    │         │
    │         └── MACE calcule via PyTorch (pur Python/C++)
    │
    ├── déplace les atomes
    │
    └── recommence jusqu'à fmax < 0.05 eV/Å
```

**Si vous voulez utiliser LAMMPS**

Il est nécessaire de convertir le modèle MACE dans un format compatible avec LAMMPS (*.pt)

```python
 find ~/venv/ATOMOD -name "create_lammps_model.py" 2>/dev/null # pour localiser le script create_lammps_model.py
 python /home/bulou/venv/ATOMOD/lib/python3.12/site-packages/mace/cli/create_lammps_model.py /home/bulou/.cache/mace/20231203mace128L1_epoch199model 
```
Cela génère un fichier ```bash 20231203mace128L1_epoch199model-lammps.pt``` utilisable directement dans un script LAMMPS. Par exemple
```lammps
units           metal
atom_style      atomic
atom_modify     map yes
newton          on

read_data       hea_np.lammps   # converti depuis extxyz avec ASE

pair_style      mace no_domain_decomposition
pair_coeff      * * mace_mp_medium.model-lammps.pt Co Cr Fe Mn Ni

# Équilibration NVT
velocity        all create 300.0 42 dist gaussian
fix             1 all nvt temp 300.0 300.0 $(100*dt)
timestep        0.002           # 2 fs
thermo          100
dump            1 all custom 500 traj.dump id type x y z

run             50000           # 100 ps d'équilibration
unfix           1

# Production NVT
fix             2 all nvt temp 300.0 300.0 $(100*dt)
run             250000          # 500 ps de production
```

-->
