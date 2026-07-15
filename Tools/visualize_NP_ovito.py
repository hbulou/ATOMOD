import sys
import ovito
from ovito.io import import_file
from ovito.vis import Viewport
from ovito.gui import create_qwidget          # ← clé manquante
from PySide6.QtWidgets import QApplication    # ← clé manquante
from pathlib import Path


def collecter_fichiers(args_paths):
    fichiers = []
    for p in args_paths:
        path = Path(p)
        if path.is_dir():
            fichiers.extend(sorted(path.glob('*.xyz')))
        elif path.is_file() and path.suffix.lower() == '.xyz':
            fichiers.append(path)
        else:
            print(f"⚠️  Ignoré : {p}")
    return fichiers


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualiseur NP XYZ — OVITO interactif')
    parser.add_argument('chemins', nargs='+', help='Fichier(s) .xyz ou dossier')
    args = parser.parse_args()

    fichiers = collecter_fichiers(args.chemins)
    if not fichiers:
        print("❌  Aucun fichier .xyz trouvé.")
        sys.exit(1)

    # Charger tous les fichiers dans la scène
    for f in fichiers:
        print(f"📂  Chargement : {f.name}")
        pipeline = import_file(str(f))
        pipeline.add_to_scene()

    # ✅ Créer le viewport
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(2, 1, -1))

    # ✅ Créer le widget Qt interactif (ouvre une vraie fenêtre)
    widget = create_qwidget(vp)
    widget.resize(800, 600)
    widget.setWindowTitle(f"ATOMOD — {fichiers[0].name}")
    widget.show()
    widget.raise_()

    # ✅ Ajuster la caméra pour voir toute la structure
    vp.zoom_all((widget.width(), widget.height()))

    # ✅ Lancer la boucle Qt (bloquant jusqu'à fermeture de la fenêtre)
    sys.exit(QApplication.instance().exec())


if __name__ == '__main__':
    main()

# import argparse
# from pathlib import Path



# import ovito
# from ovito.io import import_file
# from ovito.vis import Viewport





# # ═══════════════════════════════════════════════════════════════════════════
# # PARSING XYZ
# # ═══════════════════════════════════════════════════════════════════════════

# def lire_xyz(chemin):
#     """
#     Lit un fichier XYZ et retourne (n_atomes, commentaire, liste_atomes).
#     Format standard :
#         N
#         commentaire
#         Elt  x  y  z
#         ...
#     """
#     lignes = Path(chemin).read_text(encoding='utf-8').strip().splitlines()
#     try:
#         n = int(lignes[0].strip())
#     except ValueError:
#         raise ValueError(f"Format XYZ invalide dans {chemin} : ligne 1 doit être le nombre d'atomes")

#     commentaire = lignes[1].strip() if len(lignes) > 1 else ''
#     atomes = []

#     for i, ligne in enumerate(lignes[2:2 + n], start=2):
#         parties = ligne.split()
#         if len(parties) < 4:
#             continue
#         elt = parties[0]
#         x, y, z = float(parties[1]), float(parties[2]), float(parties[3])
#         atomes.append({'elt': elt, 'x': x, 'y': y, 'z': z})

#     return n, commentaire, atomes


# def statistiques(atomes):
#     """Retourne un dict espèce → nombre d'atomes."""
#     stats = {}
#     for a in atomes:
#         stats[a['elt']] = stats.get(a['elt'], 0) + 1
#     return dict(sorted(stats.items()))
# # ═══════════════════════════════════════════════════════════════════════════
# # POINT D'ENTRÉE
# # ═══════════════════════════════════════════════════════════════════════════

# def collecter_fichiers(args_paths):
#     """Collecte tous les fichiers XYZ depuis les chemins fournis."""
#     fichiers = []
#     for p in args_paths:
#         path = Path(p)
#         if path.is_dir():
#             fichiers.extend(sorted(path.glob('*.xyz')))
#         elif path.is_file() and path.suffix.lower() == '.xyz':
#             fichiers.append(path)
#         else:
#             print(f"⚠️  Ignoré (pas un .xyz ou dossier) : {p}")
#     return fichiers



# def main():
#     parser = argparse.ArgumentParser(
#         description='Visualiseur de nanoparticules XYZ',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Exemples :
#   python visualize_NP.py NP.xyz
#   python visualize_NP.py NP_0001.xyz NP_0002.xyz
#   python visualize_NP.py data/xyz/
#   python visualize_NP.py data/xyz/ --output ma_visu.html --no-open
#         """
#     )
#     parser.add_argument('chemins', nargs='+',
#                         help='Fichier(s) .xyz ou dossier contenant des .xyz')


#     args = parser.parse_args()

#     # Collecter les fichiers
#     fichiers = collecter_fichiers(args.chemins)

#     if not fichiers:
#         print("❌  Aucun fichier .xyz trouvé.")
#         sys.exit(1)

#     print(f"📂  {len(fichiers)} fichier(s) XYZ trouvé(s) :")
#     for f in fichiers:
#         n, _, atomes = lire_xyz(f)
#         stats = statistiques(atomes)
#         comp = ', '.join(f'{e}:{c}' for e, c in stats.items())
#         print(f"    • {f.name}  ({n} atomes  —  {comp})")

#         pipeline = import_file(f)
#         pipeline.add_to_scene()
#         vp = Viewport(type=Viewport.Type.Perspective)
#         vp.render_image(filename="NP.png", size=(800, 600))


# if __name__ == '__main__':
#     main()
