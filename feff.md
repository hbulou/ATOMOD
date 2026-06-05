---
type: Note
---
# FEFF

<https://gemini.google.com/app/072a3a409273a186>

- comment exécuter FEFF sans utiliser JFEFF ?
- je veux bien que tu m'aide à rédiger un ficher feff.inp pour calculer l'EXAFS des atomes de Rh et d'Ir contenu dans une nanoparticule dont les coordonnées sont dans un fichier [RhIr.xyz](https://RhIr.xyz)
- FEFF utilise l'écrantage RPA pour calculer le potentiel du trou de coeur durant le processus d'absorpriton. Qu'est ce l'écrantage RPA ?
- qu'est ce que le spectre Compton dans FEFF ?
- je souhaite exécuter un calcule feff à partir d'un script python? Que me conseilles tu ?
- comment utiliser Python pour centrer les coordonnées de ton fichier .xyz avant de générer le feff.inp ?
- comment faire la moyenne pondérée des spectres chi.dat des atomes de la Nanoparticule ?
- J'ai reçu ce msg : Hervé,\
  Christine il faudrait qu’on discute avec vous pour l’aspect simulation\
  FEFF. Ici à SOLEIL nous avons des données expérimentales qui en principe\
  pourrait aider à valider des modèles que vos\
  simulations font ressortir. Pour l’instant je vous avoue que nous\
  (Emiliano et moi) sommes un peu dans le flou comment mettre tous nos\
  efforts en commun au-delà d’alimenter au mieux une base de données\
  (l’exposé de Cynthia était très intéressant). Dans le\
  domaine il y a deux écoles dans les papiers Machine Learning XAS (qui\
  ont néanmoins de commun les fameux descripteurs …). J’ai mis les papiers\
  dans le zip\
  Soit\
  on va vers une approche où les descripteurs sont obtenus via une base\
  de données de spectres c’est ce qui est décrit dans les papiers\
  J. Phys Chem C (2024) 128 17921-17927. On s’affranchit des simulations\
  dans ce cas. Nos mesures à SOLEIL suffisent-elles à être cette base de\
  données ?\
  Soit\
  on va vers l’approche développée par Timoshenko décrits dans le papier\
  ACS Catalysis ou Guda dans npj Computational Materials (2023) qui\
  s’appuie sur des spectres calculés par exemple par FDMNES et on essaye\
  d’inverser le problème ie de la mesure des spectres expérimentaux on en\
  déduit une structure probable. On a compris en écoutant récemment Janis\
  en conférence que le succès de cette approche\
  est conditionné par le système. Ce qui marche pour un ne s’extrapolera\
  pas pour d’autres systèmes et donc c’est à chaque fois time consuming de\
  partir sur un nouveau système. Mais bon on peut faire l’effort pour les\
  « HEA »\
  Peut-on imaginer aussi faire un mixte des deux approches ?\
   \
  En tout\
  cas, plus que jamais je pense qu’il faudrait qu’on prenne une journée\
  ensemble pour discuter de la façon d’orchestrer tout ceci.  On peut\
  venir à Strasbourg ou on peut vous accueillir à SOLEIL\
  sur une journée de travail ou faire un mixte en visio. Cette journée de\
  réflexion est importante car elle permettra de bien préparer je crois\
  l’alimentation de la base de données discuter aujourd’hui, de bien\
  cadrer aussi la thèse qui devrait démarrer en septembre\
  chez nous et peut être aussi d’intensifier nos efforts à SOLEIL via\
  pourquoi pas une demande de post-doc de notre côté.  Sur ce dernier\
  point, j’ai mis en copie Emmanuel Fahri et Olga Roudenko du groupe Data\
  Analysis de SOLEIL, et avec lesquels nous discutons\
  aussi de ceci.\
  Si je\
  regarde notre planning avec la reprise de faisceau la semaine prochaine,\
  je pense que nous ne trouverons pas une date avant début mars. Est-ce\
  que cela vous conviendrait ?\
   \
  Voici ma réponse : Voici une réponse rapide à ton message. Je te ferai une réponse plus\
  détaillée la semaine prochaine après avoir lu (et pour certains relu)\
  les articles auxquels tu fais référence.\
  Pour te répondre, j'ai pris le parti de présenter l'état d'avancement de nos tâches dans le projet.\
  Notre partie du travail consiste à instruire deux réseaux neuronaux\
  REACT2COMPO qui sera exploité pour déterminer la structure et la\
  composition des nanoparticules possédant des propriétés catalytiques\
  ciblées ; pour l'instruction de ce modèle nous utiliserons une base de\
  données (DB) composée des propriétés catalytiques et de la structure 3D\
  (nature chimique et position xyz de chacun des atomes de la\
  nanoparticule) des nanoparticules élaborées par Marco --> pour l'instant nous n'avons encore travaillé à REACO2COMPO.\
  ATOMOD qui sera exploité pour déterminer la structure 3D des\
  nanoparticules élaborées par Marco à parttir de leurs caractérisations\
  TEM & XAS, et utilisée pour l'instruction de REACTO2COMPO. Nous\
  travaillons en ce moment sur ATOMOD au développement du script\
  d'instruction d'ATOMOD. Si une fois instruit, ATOMOD sera utilisé pour\
  analyser des données expérimentales, pour le développement du script\
  d'instruction nous utilisons une base de données simulées, à savoir des\
  images TEM (méthode multislice via le code abtem) et des spectres\
  XAS/EXAFS (code FEFF) à partir de nanoparticules générées in silico\
  (alliage binaire RhIr). Nous nous sommes d'abord focalisé sur la\
  simulation d'images TEM que nous avons utilisées pour tenter d'instruire\
  ATOMOD. La figure 2 présente ce\
  qu'ATOMOD fournit (après instruction à partir d'une DB de 4096\
  échantillons) lorqu'on lui donne entrée l'image TEM (fig.1). Sur la\
  fig.2 , chaque rangée correspond à un plan de la\
  nanoparticule Rh50Ir50, la colonne 1 (3) donne les cartes de\
  probabilité de présence du Rh (Ir), la colonne 2 (4) donne ce\
  qu'ATOMOD calcule.  \
  Pour l'instant ATOMOD ne différentie par Rh et Ir  (on remarque que\
  chaque image\
  est la somme de celles du Rh et du Ir). Toutefois ce que montre ce\
  premier résultat est qu'il est\
  possible de remonter à une distribution atomique en 3D juste à\
  partir d'une image TEM 2D. Sur la base de ce résultat, je suis\
  assez confiant sur les fait que l'intégration des spectres EXAFS devrait\
  permettre de séparer les différentes espèces dans chaque plan atomique.\
  En décembre, nous avons commencé à travailler sur la partie EXAFS. La\
  figure 3 présente un exemple de courbe χ(k) du spectre EXAFS pour une\
  nanoparticule de RhIr simulée avec FEFF. Il s'agit de la moyenne prise\
  sur l'ensemble des courbes χ(k) du spectre EXAFS calculés en prenant\
  chaque atomes de la nanoparticule en tant qu'absorbeur. J'ai mis un\
  exemple de feff.inp en PJ et la fig.4 présente un échantillon des\
  spectres calculés. A partir de la semaine prochaine, nous allons\
  poursuivre le développement du script d'instruction d'ATOMOD pour y\
  inclure les spectres EXFAS moyen (Fig. 3).\
  Pour le développement des scripts d'instruction du modèle ATOMOD,\
  nous utilisons des échantillons d'alliages binaires (RhIr) plus simples\
  que des HEAs mais en parallèle à ces développements Christine travaille à\
  la génération de la DB avec des NP d'HEAS.\
  Concernant la manière de coordonner nos travaux, ce dont nous avons besoin est d'une part de\
  votre expertise dans la simulation des spectres EXAFS car même si nous\
  ne sommes pas novices dans ce domaine, nous ne sommes pas des experts\
  non plus et d'autre part nous aurions besoin de disposer d'un moyen de\
  valider nos approches par des mesures expérimentales.\
  Pour ma part, une réunion début mars me convient pour faire le point.\
  \
  Qu'en penses tu ?

- Peux tu me proposer un descriptif de l'utilisation de FEFF dans le cadre du porjet M2P2_HEA ?
- dans feff, quel est l'ensemble minimal de programme à lancer pour réaliser une simulation d'EXAFS ?
- structurer l'appel de ces modules spécifiques dans votre script de génération de base de données
- a quoi sert atomic ?
- dmdw ?
- opconsat ?
- screen ?
- fms ?
- mkgtr ?
- genfmt ?
- sfconv ?
- compton ?
- eels ?
- ldos ?
- j'ai le problème suivant avec FEFF :\
   Calculating potentials ...\
  muffin tin radii and interstitial parameters\
   iph, rnrm(iph)*bohr, rmt(iph)*bohr, folp(iph)\
      0  1.08143E+00  9.57765E-01  1.15000E+00\
      1  1.15820E+00  1.04087E+00  1.15000E+00\
      2  1.13728E+00  1.00281E+00  1.15000E+00\
                Core-valence separation\
   ecv=   -40.000\
   mu_old=     9.717\
                SCF ITERATION NUMBER  1  OUT OF 100\
   Calculating energy and space dependent l-DOS ....\
       point #   1  energy = -40.000\
       Doing FMS for a cluster of   12 atoms around iph =   0 in fmsie\
      0   FMS matrix (LUD) at point   1, number of state kets = 108\
       Doing FMS for a cluster of   12 atoms around iph =   1 in fmsie\
      0   FMS matrix (LUD) at point   1, number of state kets = 108\
       Doing FMS for a cluster of   12 atoms around iph =   2 in fmsie\
      0   FMS matrix (LUD) at point   1, number of state kets = 108\
       point #  20  energy = -28.845\
       point #  40  energy =  -8.595\
       point #  60  energy =  11.655\
       point #  80  energy =  31.905\
       point # 100  energy =  52.155\
       point # 120  energy =  72.405\
       point # 140  energy =  92.655\
       point # 160  energy = 112.905\
       point # 180  energy = 133.155\
       point # 200  energy = 153.405\
       point # 220  energy = 173.655\
       point # 240  energy = 193.905\
       point # 260  energy = 214.155\
       point # 280  energy = 234.405\
       point # 300  energy = 254.655\
       point # 320  energy = 274.905\
       point # 340  energy = 295.155\
       point # 360  energy = 315.405\
       point # 380  energy = 335.655\
       point # 400  energy = 355.905\
       point # 420  energy = 376.155\
       point # 440  energy = 396.405\
       point # 460  energy = 416.655\
       point # 480  energy = 436.905\
       point # 500  energy = 457.155\
       point # 520  energy = 477.405\
       point # 540  energy = 497.655\
       point # 560  energy = 517.905\
       point # 580  energy = 538.155\
       point # 600  energy = 558.405\
       point # 620  energy = 578.655\
       point # 640  energy = 598.905\
       point # 660  energy = 619.155\
       point # 680  energy = 639.405\
       point # 700  energy = 659.655\
       point # 720  energy = 679.905\
       point # 740  energy = 700.155\
       point # 760  energy = 720.405\
       point # 780  energy = 740.655\
       point # 800  energy = 760.905\
       point # 820  energy = 781.155\
       point # 840  energy = 801.405\
       point # 860  energy = 821.655\
       point # 880  energy = 841.905\
       point # 900  energy = 862.155\
       point # 920  energy = 882.405\
       point # 940  energy = 902.655\
       point # 960  energy = 922.905\
       point # 980  energy = 943.155\
       point # ***  energy = 963.405\
       point # ***  energy = 983.655\
       point # *  **energy =** *****\
       point # *  **energy =** *****\
       point # *  **energy =** *****\
       point # *  **energy =** *****\
       point # *  **energy =** *****

- peux tu m'expliquer ce script feff.inp ?\
  TITLE FEFF Calculation - Atome absorbeur : Rh\
  DEBYE 190.0 315.0 0\
  EDGE K\
  SCF 5.0\
  RPATH 5.0\
  CONTROL 1 1 1 1 1 1\
  \
  POTENTIALS\
      0    45      Rh\
      1    77      Ir\
      2    45      Rh\
  \
  ATOMS\
     0.000000   0.000000   0.000000    0    Rh   0.0000 (Absorbeur)\
     6.860000   4.900000   4.900000    1    Ir   9.7509\
     5.880000   5.880000   1.960000    2    Rh   8.5434\
     6.860000   2.940000   0.980000    2    Rh   7.5275\
     5.880000   0.000000   1.960000    2    Rh   6.1981\
     2.940000   0.980000   0.980000    2    Rh   3.2503\
     0.000000   0.000000   1.960000    1    Ir   1.9600\
    -0.980000  -0.980000   0.980000    1    Ir   1.6974\
     0.980000  -0.980000   0.980000    2    Rh   1.6974\
    -0.980000   0.980000   0.980000    1    Ir   1.6974\
     2.940000  -0.980000  -0.980000    1    Ir   3.2503\
     2.940000   0.980000   0.980000    1    Ir   3.2503\
    -0.980000   2.940000  -0.980000    1    Ir   3.2503\
     0.980000   2.940000   0.980000    1    Ir   3.2503\
    -0.980000  -0.980000   2.940000    2    Rh   3.2503\
     0.980000   0.980000   2.940000    2    Rh   3.2503\
  END

- que penses tu de cette routine pour créer un fichier d'input feff.inp ?\
  def FEFF_create_parameter_file(\
  filename:str,\
  molecule: Crystal,\
  absorber_idx: int = 0,\
  config= None,\
  title: str = "FEFF Calculation") -> None:\
  \
  if config is None:\
  config = FEFF_config()\
  \
  # Positionner l'origine sur l'atome absorbeur\
  tmp_molecule=molecule.duplicate()\
  tmp_molecule.origin_at(origin=molecule.atoms[absorber_idx].q)\
  absorber = molecule.atoms[absorber_idx]\
  \
  with open(filename, "w") as f:\
  # En-tête\
  f.write(f'TITLE {title} - Atome absorbeur : {absorber.elt}\n')\
  f.write(f'DEBYE {config.debye_temp_0} {config.debye_temp} 0\n')\
  f.write(f'EDGE {config.edge}\n')\
  f.write(f'SCF {config.scf_radius}\n')\
  f.write(f'RPATH {config.rpath}\n')\
  f.write(f'CONTROL\t1 1 1 1 1 1\n')\
  \
  # Section POTENTIALS\
  f.write(f'\nPOTENTIALS\n')\
  f.write(f' {0:>4d} {Z_from_elt[absorber.elt]:>5d} {absorber.elt:>7s}\n')\
  # Liste des éléments uniques\
  for i, elt in enumerate(molecule.list_elt, start=1):\
  f.write(f' {i:>4d} {Z_from_elt[elt]:>5d} {elt:>7s}\n')\
  # Section ATOMS\
  f.write(f'\nATOMS\n')\
  f.write(\
  f' {absorber.q[0]:>10.6f} {absorber.q[1]:>10.6f} {absorber.q[2]:>10.6f} '\
  f'{0:>4d} {absorber.elt:>5s} {0:>8.4f} (Absorbeur)\n'\
  )\
  \
  # Autres atomes\
  for atm in tmp_molecule.atoms:\
  if atm.idx != absorber.idx:\
  # Trouver l'indice du potentiel\
  ipot = molecule.list_elt.index(atm.elt) + 1\
  \
  # Calculer la distance\
  R = atm.q - absorber.q\
  d = np.linalg.norm(R)\
  \
  f.write(\
  f' {atm.q[0]:>10.6f} {atm.q[1]:>10.6f} {atm.q[2]:>10.6f} '\
  f'{ipot:>4d} {atm.elt:>5s} {d:>8.4f}\n'\
  )\
  \
  f.write(f'END\n')\
  del tmp_molecule

- voici ce que j'obtiens au avec le fichier feff.inp ci-joint\
  2026-02-20 10:48:10,588 - INFO - ####################################################################################################\
  2026-02-20 10:48:10,589 - INFO - ### FEFF ### absorber 7\
  2026-02-20 10:48:10,589 - INFO - ####################################################################################################\
  rdinp -> True\
   FEFF 9.6.4\
  Resetting lmaxsc to 2 for iph =    1.  Use  UNFREEZE to prevent this.\
   Core hole lifetime set to    7.09929588076987      eV.\
   FEFF Calculation - Atome absorbeur : Rh\
   :WARNING  TWO ATOMS VERY CLOSE TOGETHER. CHECK INPUT.\
   atoms        6      12 distance   0.00000E+00 Angstrom\
      6  2.94000E+00  9.80000E-01  9.80000E-01 Z=  45\
     12  2.94000E+00  9.80000E-01  9.80000E-01 Z=  45\
  rdinp -> DONE!\
  atomic -> True\
   Calculating potentials ...\
      free atom potential and density for atom type    1\
      free atom potential and density for atom type    2\
      initial state energy\
      overlapped potential and density for unique potential    0\
      overlapped potential and density for unique potential    1\
      overlapped potential and density for unique potential    2\
  atomic -> DONE!\
  dmdw -> True\
  dmdw -> DONE!\
  pot -> True\
   Calculating potentials ...\
  muffin tin radii and interstitial parameters\
   iph, rnrm(iph)*bohr, rmt(iph)*bohr, folp(iph)\
      0  1.08143E+00  9.57765E-01  1.15000E+00\
      1  1.15820E+00  1.04087E+00  1.15000E+00\
      2  1.13728E+00  1.00281E+00  1.15000E+00\
                Core-valence separation\
   ecv=   -40.000\
   mu_old=     9.717\
                SCF ITERATION NUMBER  1  OUT OF 100\
   Calculating energy and space dependent l-DOS ....\
       point #   1  energy = -40.000\
       Doing FMS for a cluster of   12 atoms around iph =   0 in fmsie\
      0   FMS matrix (LUD) at point   1, number of state kets = 108\
       Doing FMS for a cluster of   12 atoms around iph =   1 in fmsie\
      0   FMS matrix (LUD) at point   1, number of state kets = 108\
       Doing FMS for a cluster of   12 atoms around iph =   2 in fmsie\
      0   FMS matrix (LUD) at point   1, number of state kets = 108\
       point #  20  energy = -28.845\
       point #  40  energy =  -8.595\
       point #  60  energy =  11.655\
       point #  80  energy =  31.905\
       point # 100  energy =  52.155\
       point # 120  energy =  72.405\
       point # 140  energy =  92.655\
       point # 160  energy = 112.905\
       point # 180  energy = 133.155\
       point # 200  energy = 153.405\
       point # 220  energy = 173.655\
       point # 240  energy = 193.905\
       point # 260  energy = 214.155\
       point # 280  energy = 234.405\
       point # 300  energy = 254.655\
       point # 320  energy = 274.905\
       point # 340  energy = 295.155\
       point # 360  energy = 315.405\
       point # 380  energy = 335.655\
       point # 400  energy = 355.905\
       point # 420  energy = 376.155\
       point # 440  energy = 396.405\
       point # 460  energy = 416.655\
       point # 480  energy = 436.905\
       point # 500  energy = 457.155\
       point # 520  energy = 477.405\
       point # 540  energy = 497.655\
       point # 560  energy = 517.905\
       point # 580  energy = 538.155\
       point # 600  energy = 558.405\
       point # 620  energy = 578.655\
       point # 640  energy = 598.905\
       point # 660  energy = 619.155\
       point # 680  energy = 639.405\
       point # 700  energy = 659.655\
       point # 720  energy = 679.905\
       point # 740  energy = 700.155\
       point # 760  energy = 720.405\
       point # 780  energy = 740.655\
       point # 800  energy = 760.905\
       point # 820  energy = 781.155\
       point # 840  energy = 801.405\
       point # 860  energy = 821.655\
       point # 880  energy = 841.905\
       point # 900  energy = 862.155\
       point # 920  energy = 882.405\
       point # 940  energy = 902.655\
       point # 960  energy = 922.905\
       point # 980  energy = 943.155\
       point # ***  energy = 963.405\
       point # ***  energy = 983.655\
       point # *  **energy =** *****\
       point # *  **energy =** *****\
       point # *  **energy =** *****\
       point # *  **energy =** *****\
       point # *  **energy =** *****

- de quoi traite cet article ?
- quels références dois je lire pour comprendre ce que sont les facteurs de Dabye-Waller ?
- dans un calcul feff avec rdinp, que signifie ce msg :\
           FEFF 9.6.4\
  Resetting lmaxsc to 2 for iph =    0.  Use  UNFREEZE to prevent this.\
  Resetting lmaxsc to 2 for iph =    1.  Use  UNFREEZE to prevent this.\
   Core hole lifetime set to    51.2579764290019      eV.\
   Atome absorbeur : Ir(idx=0)

- dans un calcul feff avec rdinp, que signifie ce msg :\
           FEFF 9.6.4\
  Resetting lmaxsc to 2 for iph =    0.  Use  UNFREEZE to prevent this.\
  Resetting lmaxsc to 2 for iph =    1.  Use  UNFREEZE to prevent this.\
   Core hole lifetime set to    51.2579764290019      eV.\
   Atome absorbeur : Ir(idx=0)

- dans ce script, comment ajouter la possibilité de choisir les colonnes à afficher ?
- que dois modifier pour choisir le range en x ?
- meme chose avec y ?
- dans FEFF, que signifie le msg suivant :\
  FEFF 9.6.4\
  Resetting lmaxsc to 2 for iph =    0.  Use  UNFREEZE to prevent this.\
  Resetting lmaxsc to 2 for iph =    1.  Use  UNFREEZE to prevent this.\
  Resetting lmaxsc to 2 for iph =    2.  Use  UNFREEZE to prevent this.\
   No atoms or overlap cards for unique pot        5\
   Cannot calculate potentials, etc.\
  RDINP-

- est il possible, en s'inspirant du script ci-joit, de faire un script python qui lit une série de spectres EXAFS xmu_elt_idx.dat pour en faire un spectre moyen ?
- comment modifier ce script pour mettre le file_pattern sur la ligne de commande
- dans un spectre exafs, le premier pic a t il un nom particulier ?
- que penses tu du texte suivant pour décrire la figure ci-jointe ?\
  La figure XXX présente une série de simulations de spectres EXAFS pour des nanoparticules à base de Ni, Ru et Ir.\
  Les nanoparticules ont un rayon de 9 angstroems, correspondant à 216 atomes au total.\
  Excepté la composition, toutes les nanoparticules considérées dans la simulation sont identiques.\
  Nous avons considéré deux types de compositions : des particules pures pour avoir des spectres de référence (a) et des particules trimétalliques. Pour les particules trimétalliques, chaque élément représente un tier de la composition.\
  Deux types d’arrangements chimiques ont été considérés : des nanoparticules de type core-shell (CS) (b) et des nanoparticules complètement désordonnées (c).\
  Pour les configurations CS, les 6 permuations ont été considérées.\
  Les figures XXX (d,e,f) mettent en évidence des signatures spécifiques pour chacune des configurations considérées.\
  Dans le cas des configurations CS, c’est la localisation de l’élément dans la couche qui détermine la forme des spectres, et ce quelque soit l’élément chimique.\
  Il est intéressant de remarquer que, selon l’élément, les spectres purs sont proches de type de couches différentes.\
  Dans le cas du ruhténium, le spectre pur est proche des spectres de particules CS lorsque le ruthénium est au coeur ou dans la couche intermédiare de la nanoparticule.\
  A l’inverse, dans le cas de l’irridium, le spectre pur est proche du spectre des particules CS lorsque l’irridium forme la couche de surface.\
  Enfin, pour le spectre du nickel pur, aucune correspondance avec un des spectres des nanoparticules CS n’est observée.\
  Concernant le cas de la nanoparticule désordonnée, le spectre de la particule s’apparente au spectre des nanoparticules CS lorsque l’élément est en couche intermédiare, avec toutefois des variations d’intensité.

- dans FEFF, la sphère autour de l'atome absorbeur a t elle une désignation particulière ?
- quelle est la structure du fichier de data généré en sortie par FEFF pour un calcul EXAFS ?
