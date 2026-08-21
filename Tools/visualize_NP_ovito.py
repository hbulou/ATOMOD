"""
Ce script ouvre une fenêtre interactive OVITO pour visualiser des nanoparticules XYZ. Voici son déroulement :

python visualize_NP_ovito.py NP.xyz
"""
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

