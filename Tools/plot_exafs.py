import argparse
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration du gestionnaire d'arguments
parser = argparse.ArgumentParser(description="Trace les spectres EXAFS avec contrôle total des axes.")

# Fichiers
parser.add_argument('fichiers', nargs='+', help="Liste des fichiers .dat")

# Colonnes
parser.add_argument('--xcol', type=int, default=0, help="Index colonne X (défaut: 0)")
parser.add_argument('--ycol', type=int, default=4, help="Index colonne Y (défaut: 4)")

# Range X
parser.add_argument('--xmin', type=float, default=None, help="Limite min axe X")
parser.add_argument('--xmax', type=float, default=None, help="Limite max axe X")

# Range Y (La modification que vous avez demandée)
parser.add_argument('--ymin', type=float, default=None, help="Limite min axe Y")
parser.add_argument('--ymax', type=float, default=None, help="Limite max axe Y")

args = parser.parse_args()

# 2. Préparation du graphique
plt.figure(figsize=(10, 6)) 

# 3. Boucle de tracé
for nom_fichier in args.fichiers:
    try:
        data = np.loadtxt(nom_fichier, comments='#', usecols=(args.xcol, args.ycol))
        plt.plot(data[:, 0], data[:, 1], label=nom_fichier, linewidth=2)
    except Exception as e:
        print(f"⚠️ Erreur sur {nom_fichier} : {e}")

# 4. Personnalisation et application des Ranges
plt.xlabel(f'Colonne {args.xcol}')
plt.ylabel(f'Colonne {args.ycol}')

# Application des limites X et Y
plt.xlim(left=args.xmin, right=args.xmax)
plt.ylim(bottom=args.ymin, top=args.ymax)

plt.title('Comparaison de spectres')
plt.legend()       
plt.grid(True)     

plt.show()

# import argparse
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Configuration du gestionnaire d'arguments
# parser = argparse.ArgumentParser(description="Trace les spectres EXAFS avec choix des colonnes et du range X.")

# parser.add_argument('fichiers', nargs='+', help="Liste des fichiers .dat à comparer")
# parser.add_argument('--xcol', type=int, default=0, help="Index colonne X (défaut: 0)")
# parser.add_argument('--ycol', type=int, default=4, help="Index colonne Y (défaut: 4)")

# # Nouveaux arguments pour le range X
# parser.add_argument('--xmin', type=float, default=None, help="Limite minimale de l'axe X")
# parser.add_argument('--xmax', type=float, default=None, help="Limite maximale de l'axe X")
# parser.add_argument('--ymin', type=float, default=None, help="Limite minimale de l'axe y")
# parser.add_argument('--ymax', type=float, default=None, help="Limite maximale de l'axe y")

# args = parser.parse_args()

# # 2. Préparation du graphique
# plt.figure(figsize=(10, 6)) 

# # 3. Boucle sur chaque fichier
# for nom_fichier in args.fichiers:
#     try:
#         data = np.loadtxt(nom_fichier, comments='#', usecols=(args.xcol, args.ycol))
#         plt.plot(data[:, 0], data[:, 1], label=nom_fichier, linewidth=2)
#     except Exception as e:
#         print(f"⚠️ Erreur sur {nom_fichier} : {e}")

# # 4. Personnalisation et application du Range X
# plt.xlabel(f'Colonne {args.xcol}')
# plt.ylabel(f'Colonne {args.ycol}')

# # C'est ici que le range est appliqué
# plt.xlim(left=args.xmin, right=args.xmax)
# plt.ylim(left=args.ymin, right=args.ymax)

# plt.title('Comparaison de spectres')
# plt.legend()       
# plt.grid(True)     

# plt.show()

# # import argparse
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # 1. Configuration du gestionnaire d'arguments
# # parser = argparse.ArgumentParser(description="Trace les spectres de plusieurs fichiers FEFF en choisissant les colonnes.")

# # # Liste des fichiers (arguments positionnels)
# # parser.add_argument('fichiers', nargs='+', help="Liste des fichiers .dat à comparer")

# # # Nouveaux arguments pour le choix des colonnes (arguments optionnels)
# # # On définit des valeurs par défaut (0 et 4) pour conserver le comportement actuel par défaut
# # parser.add_argument('--xcol', type=int, default=0, help="Index de la colonne pour l'axe X (commence à 0, défaut: 0)")
# # parser.add_argument('--ycol', type=int, default=4, help="Index de la colonne pour l'axe Y (commence à 0, défaut: 4)")

# # args = parser.parse_args()

# # # 2. Préparation du graphique
# # # Note : Selon les instructions de l'environnement, évitez .figure() si vous exécutez en sandbox
# # plt.figure(figsize=(10, 6)) 

# # # 3. Boucle sur chaque fichier
# # for nom_fichier in args.fichiers:
# #     try:
# #         # On utilise les colonnes spécifiées dans les arguments
# #         # usecols accepte un tuple avec les indices choisis
# #         data = np.loadtxt(nom_fichier, comments='#', usecols=(args.xcol, args.ycol))
        
# #         x_values = data[:, 0]
# #         y_values = data[:, 1]
        
# #         plt.plot(x_values, y_values, label=f"{nom_fichier} (col {args.ycol})", linewidth=2)
        
# #     except IndexError:
# #         print(f"⚠️ Erreur : Le fichier {nom_fichier} n'a pas assez de colonnes pour l'index demandé.")
# #     except Exception as e:
# #         print(f"⚠️ Erreur lors de la lecture de {nom_fichier} : {e}")

# # # 4. Personnalisation du graphe
# # # Les labels deviennent génériques puisque les colonnes peuvent changer
# # plt.xlabel(f'Colonne {args.xcol}')
# # plt.ylabel(f'Colonne {args.ycol}')
# # plt.title(f'Comparaison de {len(args.fichiers)} spectre(s)')
# # plt.legend()       
# # plt.grid(True)     

# # # 5. Sauvegarde ou Affichage
# # # Pour une utilisation locale :
# # plt.show()
# # # Pour l'environnement de simulation (obligatoire ici) :
# # # plt.savefig('comparaison_spectres.png')
