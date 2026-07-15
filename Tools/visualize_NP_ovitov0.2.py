"""
visualize_NP_ovito.py
─────────────────────
Visualiseur interactif de nanoparticules XYZ basé sur OVITO.

Fonctionnalités :
  - Liste des NP dans un panneau gauche
  - Clic sur un nom → affichage dans la visionneuse 3D
  - Panneau d'info (composition, nombre d'atomes)
  - Fond noir, style sphères, couleurs Jmol

Utilisation :
  python visualize_NP_ovito.py NP.xyz
  python visualize_NP_ovito.py NP_001.xyz NP_002.xyz
  python visualize_NP_ovito.py data/xyz/

Dépendances :
  pip install ovito          (inclut PySide6)
"""

import sys
import argparse
from pathlib import Path

import ovito
from ovito.io import import_file
from ovito.vis import Viewport
from ovito.gui import create_qwidget

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor


# ═══════════════════════════════════════════════════════════════════════════
# PARSING XYZ (conservé depuis votre script d'origine)
# ═══════════════════════════════════════════════════════════════════════════

def lire_xyz(chemin):
    lignes = Path(chemin).read_text(encoding='utf-8').strip().splitlines()
    try:
        n = int(lignes[0].strip())
    except ValueError:
        raise ValueError(f"Format XYZ invalide dans {chemin}")
    commentaire = lignes[1].strip() if len(lignes) > 1 else ''
    atomes = []
    for ligne in lignes[2:2 + n]:
        parties = ligne.split()
        if len(parties) < 4:
            continue
        elt = parties[0]
        x, y, z = float(parties[1]), float(parties[2]), float(parties[3])
        atomes.append({'elt': elt, 'x': x, 'y': y, 'z': z})
    return n, commentaire, atomes


def statistiques(atomes):
    stats = {}
    for a in atomes:
        stats[a['elt']] = stats.get(a['elt'], 0) + 1
    return dict(sorted(stats.items(), key=lambda x: -x[1]))


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


# ═══════════════════════════════════════════════════════════════════════════
# FENÊTRE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════

class ViewerNP(QMainWindow):
    """
    Fenêtre principale :
    ┌──────────────────┬──────────────────────────────────┐
    │  Liste des NP    │                                  │
    │  ─────────────── │       Visionneuse OVITO 3D       │
    │  NP_0001.xyz     │                                  │
    │  NP_0002.xyz     │                                  │
    │  NP_0003.xyz     │                                  │
    │  ...             │                                  │
    │  ─────────────── │                                  │
    │  Infos NP        │                                  │
    │  N atomes: 309   │                                  │
    │  Rh: 155 (50%)   │                                  │
    │  Ir: 154 (50%)   │                                  │
    └──────────────────┴──────────────────────────────────┘
    """

    def __init__(self, fichiers):
        super().__init__()
        self.setWindowTitle("ATOMOD — Visualiseur Nanoparticules")
        self.resize(1100, 700)

        # ── Données : charger tous les pipelines SANS les ajouter à la scène
        self.structures = []
        for f in fichiers:
            n, commentaire, atomes = lire_xyz(f)
            stats = statistiques(atomes)
            pipeline = import_file(str(f))   # chargé mais PAS dans la scène
            self.structures.append({
                'chemin':      f,
                'nom':         f.name,
                'n':           n,
                'commentaire': commentaire,
                'stats':       stats,
                'pipeline':    pipeline,
            })
            print(f"  ✓ Chargé : {f.name}  ({n} atomes  — "
                  f"{', '.join(f'{e}:{c}' for e, c in stats.items())})")

        # ── Pipeline actuellement affiché
        self.pipeline_actif = None

        # ── Construire l'interface
        self._construire_ui()

        # ── Sélectionner la première NP automatiquement
        if self.structures:
            self.liste.setCurrentRow(0)
            self._afficher(0)

    # ─────────────────────────────────────────────────────────────
    # Construction de l'interface Qt
    # ─────────────────────────────────────────────────────────────

    def _construire_ui(self):
        """Crée la disposition : panneau gauche | visionneuse 3D."""
        central = QWidget()
        self.setCentralWidget(central)
        layout_principal = QHBoxLayout(central)
        layout_principal.setContentsMargins(0, 0, 0, 0)
        layout_principal.setSpacing(0)

        # ── Panneau gauche ──────────────────────────────────────────
        panneau_gauche = QWidget()
        panneau_gauche.setFixedWidth(240)
        panneau_gauche.setStyleSheet("background-color: #1a1d2e;")
        layout_gauche = QVBoxLayout(panneau_gauche)
        layout_gauche.setContentsMargins(0, 0, 0, 0)
        layout_gauche.setSpacing(0)

        # Titre de la liste
        titre_liste = QLabel("  Nanoparticules")
        titre_liste.setStyleSheet("""
            background-color: #252840;
            color: #a78bfa;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1px;
            padding: 10px 8px;
            border-bottom: 1px solid #2d3148;
        """)
        layout_gauche.addWidget(titre_liste)

        # Liste des NP
        self.liste = QListWidget()
        self.liste.setStyleSheet("""
            QListWidget {
                background-color: #1a1d2e;
                border: none;
                color: #94a3b8;
                font-size: 12px;
                outline: none;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #252840;
            }
            QListWidget::item:hover {
                background-color: #252840;
                color: #e2e8f0;
            }
            QListWidget::item:selected {
                background-color: #2d3058;
                color: #a78bfa;
                font-weight: bold;
                border-left: 3px solid #a78bfa;
            }
        """)

        for s in self.structures:
            item = QListWidgetItem(f"  {s['nom']}")
            item.setToolTip(str(s['chemin']))
            self.liste.addItem(item)

        # Connexion clic → affichage
        self.liste.currentRowChanged.connect(self._afficher)
        layout_gauche.addWidget(self.liste, stretch=1)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2d3148;")
        layout_gauche.addWidget(sep)

        # Panneau d'information
        self.panneau_info = QWidget()
        self.panneau_info.setStyleSheet("background-color: #1a1d2e;")
        layout_info = QVBoxLayout(self.panneau_info)
        layout_info.setContentsMargins(12, 12, 12, 12)
        layout_info.setSpacing(4)

        # Titre info
        titre_info = QLabel("INFOS")
        titre_info.setStyleSheet("""
            color: #4c4f8a;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 1px;
        """)
        layout_info.addWidget(titre_info)

        self.label_nom = QLabel("—")
        self.label_nom.setStyleSheet("color: #e2e8f0; font-size: 11px; font-weight: bold;")
        self.label_nom.setWordWrap(True)
        layout_info.addWidget(self.label_nom)

        self.label_n = QLabel("")
        self.label_n.setStyleSheet("color: #a78bfa; font-size: 20px; font-weight: bold;")
        layout_info.addWidget(self.label_n)

        self.label_n_texte = QLabel("atomes")
        self.label_n_texte.setStyleSheet("color: #64748b; font-size: 11px;")
        layout_info.addWidget(self.label_n_texte)

        self.label_composition = QLabel("")
        self.label_composition.setStyleSheet("""
            color: #94a3b8;
            font-size: 11px;
            margin-top: 8px;
        """)
        self.label_composition.setWordWrap(True)
        layout_info.addWidget(self.label_composition)

        self.label_commentaire = QLabel("")
        self.label_commentaire.setStyleSheet("""
            color: #4c4f8a;
            font-size: 10px;
            font-style: italic;
            margin-top: 4px;
        """)
        self.label_commentaire.setWordWrap(True)
        layout_info.addWidget(self.label_commentaire)

        layout_info.addStretch()
        layout_gauche.addWidget(self.panneau_info)

        # Aide clavier en bas
        aide = QLabel("  🖱  Clic gauche : rotation\n  🖱  Clic droit : translation\n  ⚙  Molette : zoom")
        aide.setStyleSheet("""
            color: #4c4f8a;
            font-size: 10px;
            padding: 10px 8px;
            border-top: 1px solid #2d3148;
        """)
        layout_gauche.addWidget(aide)

        layout_principal.addWidget(panneau_gauche)

        # ── Visionneuse OVITO (droite) ───────────────────────────────
        self.viewport = Viewport(
            type=Viewport.Type.Perspective,
            camera_dir=(1, 1, -1)
        )

        self.vp_widget = create_qwidget(
            self.viewport,
            show_orientation_indicator=True,
            show_title=False
        )
        self.vp_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        layout_principal.addWidget(self.vp_widget, stretch=1)

    # ─────────────────────────────────────────────────────────────
    # Logique d'affichage : clic sur un item de la liste
    # ─────────────────────────────────────────────────────────────

    def _afficher(self, index):
        """
        Appelée quand l'utilisateur clique sur une NP dans la liste.
        1. Retire le pipeline précédent de la scène
        2. Ajoute le nouveau pipeline à la scène
        3. Recentre la caméra
        4. Met à jour le panneau d'info
        """
        if index < 0 or index >= len(self.structures):
            return

        s = self.structures[index]

        # ── 1. Retirer l'ancien pipeline de la scène ──
        if self.pipeline_actif is not None:
            self.pipeline_actif.remove_from_scene()

        # ── 2. Ajouter le nouveau pipeline ──
        s['pipeline'].add_to_scene()
        self.pipeline_actif = s['pipeline']

        # ── 3. Recentrer la caméra sur la nouvelle NP ──
        self.viewport.zoom_all((self.vp_widget.width(), self.vp_widget.height()))

        # ── 4. Mettre à jour le titre de la fenêtre ──
        self.setWindowTitle(f"ATOMOD — {s['nom']}")

        # ── 5. Mettre à jour le panneau d'info ──
        self._mettre_a_jour_info(s)

    def _mettre_a_jour_info(self, s):
        """Met à jour le panneau d'information avec les données de la NP sélectionnée."""
        self.label_nom.setText(s['nom'])
        self.label_n.setText(str(s['n']))

        # Composition détaillée
        lignes_comp = []
        for elt, n in s['stats'].items():
            pct = n / s['n'] * 100
            lignes_comp.append(f"{elt} : {n}  ({pct:.1f}%)")
        self.label_composition.setText('\n'.join(lignes_comp))

        # Commentaire du fichier XYZ
        if s['commentaire']:
            self.label_commentaire.setText(s['commentaire'])
        else:
            self.label_commentaire.setText('')


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Visualiseur OVITO de nanoparticules XYZ avec liste interactive',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python visualize_NP_ovito.py NP.xyz
  python visualize_NP_ovito.py NP_0001.xyz NP_0002.xyz NP_0003.xyz
  python visualize_NP_ovito.py data/xyz/
        """
    )
    parser.add_argument('chemins', nargs='+',
                        help='Fichier(s) .xyz ou dossier contenant des .xyz')
    args = parser.parse_args()

    fichiers = collecter_fichiers(args.chemins)
    if not fichiers:
        print("❌  Aucun fichier .xyz trouvé.")
        sys.exit(1)

    print(f"\n📂  {len(fichiers)} nanoparticule(s) trouvée(s) :\n")

    # Lancer l'application Qt
    app = QApplication.instance() or QApplication(sys.argv)

    fenetre = ViewerNP(fichiers)
    fenetre.show()
    fenetre.raise_()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
    
