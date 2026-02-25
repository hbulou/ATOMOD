import argparse
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration du gestionnaire d'arguments
parser = argparse.ArgumentParser(description="Trace les spectres (colonnes 1 et 5) de plusieurs fichiers FEFF.")

# Le '+' signifie : au moins un argument, mais sans limite de nombre
parser.add_argument('fichiers', nargs='+', help="Liste des fichiers .dat à comparer (séparés par un espace)")

# Récupération de la liste des fichiers tapés dans le terminal
args = parser.parse_args()

# 2. Préparation du graphique
plt.figure(figsize=(10, 6)) # Agrandit un peu la fenêtre pour plus de clarté

# 3. Boucle magique : on parcourt chaque fichier fourni
for nom_fichier in args.fichiers:
    try:
        # Lecture des données pour le fichier en cours
        data = np.loadtxt(nom_fichier, comments='#', usecols=(0, 4))
        
        omega = data[:, 0]
        mu0   = data[:, 1]
        
        # Tracé de la courbe (matplotlib gère automatiquement le changement de couleur !)
        plt.plot(omega, mu0, label=nom_fichier, linewidth=2)
        
    except Exception as e:
        # Sécurité : si un fichier n'existe pas ou est corrompu, on prévient et on passe au suivant
        print(f"⚠️ Erreur lors de la lecture de {nom_fichier} : {e}")

# 4. Personnalisation du graphe
plt.xlabel('Omega (eV)')
plt.ylabel('mu0 (Unité arbitraire)')
# Titre dynamique basé sur le nombre de fichiers
plt.title(f'Comparaison de {len(args.fichiers)} spectre(s) mu0')
plt.legend()       
plt.grid(True)     

# 5. Affichage
plt.show()
