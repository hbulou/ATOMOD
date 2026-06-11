# Projet ATOMOD
Le projet ATOMOD est hébergé sur GitHub. Bien qu'il ne soit pas nécessaire d'avoir un compte GitHub pour récupérer le projet, c'est toutefois fortement recommandé pour bénéficier facilement des mises à jour régulières (voir [Créer un compte GitHub](https://github.com/hbulou/ATOMOD/wiki/Cr%C3%A9er-un-compte-GitHub)).

## Récupérer les fichiers du projet ATOMOD
Pour récupérer les fichiers du projet ATOMOD du dépôt Github, on utilise l'opération de **clonage**. Cela crée une copie locale identique au projet distant, incluant tout l'historique des modifications.

Vous avez le choix entre quatre méthodes :



<details>
  <summary>1. La méthode classique (SSH)</summary>
    
Si vous avez configuré une clé SSH (ce qui est fortement recommandé), c'est la méthode la plus simple et la plus rapide (voir [Comment configurer une clef SSH ?](https://github.com/hbulou/ATOMOD/wiki/Configurer-une-cl%C3%A9-SSH)).

1. Sur GitHub, allez sur la page de votre dépôt.
2. Cliquez sur le bouton vert **Code**.
3. Vérifiez que l'onglet **SSH** est sélectionné et copiez l'adresse `git@github.com:hbulou/ATOMOD.git`.
4. Dans votre terminal, placez-vous là où vous voulez mettre le projet et tapez :
```bash
git clone git@github.com:hbulou/ATOMOD.git
```

</details>
<details>
  <summary>2. Si vous n'avez pas de clé SSH (HTTPS)</summary>
Si vous êtes sur un ordinateur tiers où votre clé n'est pas installée, utilisez l'adresse HTTPS :

```bash
git clone https://github.com/hbulou/ATOMOD.git

```

*Note : GitHub vous demandera alors votre nom d'utilisateur et votre **Personal Access Token** (pas votre mot de passe).*

</details>
<details>
  <summary>3. Télécharger sans utiliser Git (Archive ZIP)</summary>
Si vous voulez juste les fichiers sans l'historique Git (pour une consultation rapide par exemple) :

1. Sur la page du dépôt, cliquez sur **Code**.
2. Cliquez sur **Download ZIP**.
3. Décompressez le fichier sur votre ordinateur.



</details>
<details>
  <summary> 4. Mettre à jour un dépôt déjà cloné</summary>
Si vous avez déjà cloné le dépôt il y a quelque temps et que vous voulez récupérer les dernières modifications faites sur GitHub (par exemple après avoir édité un fichier directement via l'interface web), utilisez :

```bash
git pull origin main

```


</details>


---
## 1. Installer l'environnement vituel "ATOMOD"
L'environnement virtuel ATOMOD est un espace de travail isolé sur votre ordinateur dédiée au projet ATOMOD. Il contient toutes les bibliothèques nécéssaires à l'exécution d'ATOMOD.
<details>
<summary> Linux </summary>

### Installer l'environnement python "ATOMOD"
```bash
cd
python3 -m venv venv/ATOMOD   # création de l'environnement - répertoire ~/venv/Emergence
source venv/ATOMOD/bin/activate  # activation de l'environnement
pip install --upgrade pip                   # màj de pip
pip install -r requirements.txt
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




## 2. Installation
* choisir un répertoire où stocker ATOMOD et s'y placer
* cloner ATOMOD
```sh
git clone git@github.com:hbulou/ATOMOD.git
```
* en principe un répertoire ATOMOD a été créé. Aller dans ATOMOD
```sh
cd ATOMOD
```
* charger l'environnement ATOMOD. 
```sh
source ~/venv/ATOMOD/bin/activate
# ou
source ~/venv/ATOMOD_gpu/bin/activate
```
RQ : sur les serveurs de calcul, il peut être nécessaire de charger le module python au préalable
```sh
module load python
```


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
