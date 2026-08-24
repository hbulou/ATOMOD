"""
view_molv0.5.py
───────────────
Visualiseur OVITO de molécules avec édition interactive.

Utilisation :
    python view_molv0.5.py --file NP.xyz

Contrôles :
    Clic gauche + glisser   : rotation
    Clic droit + glisser    : translation
    Molette                 : zoom
    Shift + clic gauche     : supprimer l'atome le plus proche du curseur
    Ctrl  + clic droit      : changer la nature chimique de l'atome le plus proche
    Ctrl  + Z               : annuler la dernière action
    Ctrl  + E               : afficher la liste des atomes restants
"""


import tempfile, os

os.environ["HBPY_MACE"] = "False"

import sys

print("Répertoire du script:", os.path.dirname(os.path.abspath(__file__)))
print("Répertoire courant:", os.getcwd())
print("\nPremiers chemins de sys.path:")
for p in sys.path[:5]:
    print(f"  {p}")

# Vérifier si ATOMOD est accessible
atomod_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"\nRépertoire parent: {atomod_parent}")
print(f"Existe HBPy? {os.path.exists(os.path.join(atomod_parent, 'HBPy'))}")

# Ajouter le répertoire parent (ATOMOD) à sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import subprocess

import math
import argparse
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QMenu,
    QDialog, QTableWidget, QTableWidgetItem, QPushButton, QHeaderView,
    QAbstractItemView
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeyEvent, QColor, QIcon, QPixmap, QFont

from HBPy.Molecule.Atom import CPK_COLOR
import ovito
import ovito.data
import ovito.vis
import ovito.gui
from ovito.pipeline import StaticSource, Pipeline

import HBPy.Molecule.Crystal
# ═══════════════════════════════════════════════════════════════════════════
# Lecture du fichier
# ═══════════════════════════════════════════════════════════════════════════
def read_file(args):
    if ":/" in args.file:
        suf=".xyz"
        if args.format == "ORCA_input":
            suf=".inp"
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
            subprocess.run(["scp", args.file, tmp.name], check=True)
            args.file = tmp.name
        except subprocess.CalledProcessError as e:
            print(f"❌  Erreur scp : {e}")
            sys.exit(1)
    if args.format == "ORCA_input":
        atoms=read_ORCA_input_file(args.file)
    elif args.format == "xyz":
        atoms=read_xyz_file(args.file)
    else:
        print(f"❌  Format inconnu : {args.format}")
        sys.exit(1)
    return atoms
def read_xyz_file(fichier):
    mol=HBPy.Molecule.Crystal.Crystal()
    mol.load_file(fichier)
    #mol.Main_Axis()
    atoms=[]
    for atm in mol.atoms:
        atoms.append({
            'elt': atm.elt,
            'x': atm.q[0],
            'y': atm.q[1],
            'z': atm.q[2]})
    del mol
    return atoms
def read_ORCA_input_file(fichier):
    """Lit un bloc *xyz ... * et retourne la liste d'atomes."""
    lines = Path(fichier).read_text(encoding='utf-8').strip().splitlines()
    atoms = []
    atom_line = False
    for line in lines:
        #if atom_line and '*' in line:
        if atom_line and line.strip().startswith('*'):
            atom_line = False
        if atom_line:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append({
                    'elt': parts[0],
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3]),
                })
        if '*' in line and "xyz " in line:
            atom_line = True
    return atoms


# ═══════════════════════════════════════════════════════════════════════════
# Construction du pipeline OVITO
# ═══════════════════════════════════════════════════════════════════════════

def build_pipeline(atoms):
    """Construit un pipeline OVITO StaticSource depuis une liste d'atomes."""
    positions = [[a['x'], a['y'], a['z']] for a in atoms]
    elt_to_id = {elt: i for i, elt in enumerate(CPK_COLOR.keys(), start=1)}

    data  = ovito.data.DataCollection()
    parts = ovito.data.Particles()
    parts.create_property('Position', data=positions)

    tp = parts.create_property('Particle Type')
    for elt, tid in elt_to_id.items():
        tp.types.append(ovito.data.ParticleType(id=tid, name=elt,
                                                 color=CPK_COLOR[elt]))
    for i, atm in enumerate(atoms):
        tp[i] = elt_to_id.get(atm['elt'], 1)

    data.objects.append(parts)
    return Pipeline(source=StaticSource(data=data))


# ═══════════════════════════════════════════════════════════════════════════
# Widget transparent superposé au viewport — intercepte Shift+clic
# ═══════════════════════════════════════════════════════════════════════════

class PickOverlay(QWidget):
    """
    Widget 100% transparent superposé au viewport OVITO.

    Pourquoi un overlay ?
    Le widget OVITO (C++) consomme les événements souris avant Qt.
    Un widget transparent placé par-dessus reçoit les événements
    AVANT le widget OVITO, permettant d'intercepter Shift+clic.
    Les événements normaux (clic sans Shift) sont passés au parent
    via setAttribute(WA_TransparentForMouseEvents) dynamiquement.
    """
    SEUIL_NDC = 0.08   # seuil de picking en coordonnées NDC
    
    def __init__(self, parent, vp, atoms_ref, status_label, pipeline_ref):
        super().__init__(parent)
        self.vp           = vp
        self.atoms        = atoms_ref    # liste Python partagée
        self.status       = status_label
        self.pipeline_ref = pipeline_ref # [pipeline] liste à 1 élément (mutable)
        self.historique   = []           # pile Ctrl+Z

        # Fond transparent, intercepte la souris
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background: transparent;")

    def resizeEvent(self, event):
        """Toujours même taille que le parent."""
        self.resize(self.parent().size())
        super().resizeEvent(event)

    # ─────────────────────────────────────────────────────────────
    # Événements souris
    # ─────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):


        if (event.button() == Qt.MouseButton.LeftButton and
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            # Shift + clic gauche : supprimer l'atome le plus proche
            self._supprimer(event.pos())
            event.accept()
        elif (event.button() == Qt.MouseButton.RightButton and
              event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            # Ctrl + clic droit : changer la nature chimique
            self._changer_element(event.pos())
            event.accept()
            # ✅ NOUVEAU : Shift + clic droit → ajouter un atome
        elif (event.button() == Qt.MouseButton.RightButton and
              event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._ajouter_atome(event.pos())
            event.accept()
        else:
            # Clic normal : passer au viewport OVITO en dessous
            event.ignore()

    def wheelEvent(self, event):
        # Passer la molette au viewport OVITO
        event.ignore()

    def mouseMoveEvent(self, event):
        # Passer le mouvement au viewport OVITO
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()

    # ─────────────────────────────────────────────────────────────
    # Picking : trouver l'atome le plus proche du curseur
    # ─────────────────────────────────────────────────────────────
    def _ajouter_atome(self, pos_ecran):
        """
        Shift + clic droit : ajoute un atome à la position 3D
        correspondant au curseur, dans le plan moyen de la molécule.
        
        Stratégie de déprojection 2D → 3D :
        1. Calculer le centre de masse de la molécule → point du plan
        2. Construire le plan perpendiculaire à la caméra passant par ce centre
        3. Intersect er le rayon de visée avec ce plan
        4. Demander l'élément via un menu contextuel
        5. Insérer l'atome et rafraîchir
        """
        
        # ── Paramètres caméra (même logique que le picking) ──────────
        cam_pos = np.array(self.vp.camera_pos, dtype=float)
        cam_dir = np.array(self.vp.camera_dir, dtype=float)
        cam_dir /= np.linalg.norm(cam_dir)
        
        z_world = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(cam_dir, z_world)) > 0.99:
            z_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(cam_dir, z_world)
        right /= np.linalg.norm(right)
        up = np.cross(right, cam_dir)
        up /= np.linalg.norm(up)

        fov_rad = float(self.vp.fov)
        if fov_rad <= 0:
            fov_rad = math.radians(45.0)
        tan_fov = math.tan(fov_rad / 2.0)

        w = self.width()
        h = self.height()
        aspect = w / h if h > 0 else 1.0

        # ── NDC du clic ──────────────────────────────────────────────
        cx = (pos_ecran.x() / w) * 2.0 - 1.0
        cy = 1.0 - (pos_ecran.y() / h) * 2.0

        # ── Direction du rayon de visée depuis le clic ───────────────
        # rayon = cam_dir + cx*tan_fov*aspect*right + cy*tan_fov*up
        ray_dir = (cam_dir
                   + cx * tan_fov * aspect * right
                   + cy * tan_fov * up)
        ray_dir /= np.linalg.norm(ray_dir)

        # ── Plan de déprojection ─────────────────────────────────────
        # On utilise le centre de masse si des atomes existent,
        # sinon l'origine.
        if self.atoms:
            positions = np.array([[a['x'], a['y'], a['z']]
                                  for a in self.atoms])
            plan_centre = positions.mean(axis=0)
        else:
            plan_centre = np.array([0.0, 0.0, 0.0])

        # Plan perpendiculaire à cam_dir passant par plan_centre
        # Equation : cam_dir · (P - plan_centre) = 0
        # Intersection avec le rayon : P = cam_pos + t * ray_dir
        # → t = cam_dir · (plan_centre - cam_pos) / (cam_dir · ray_dir)
        denom = np.dot(cam_dir, ray_dir)
        if abs(denom) < 1e-8:
            self.status.setText("⚠️  Rayon parallèle au plan — impossible de placer l'atome")
            return

        t = np.dot(cam_dir, plan_centre - cam_pos) / denom
        if t <= 0:
            self.status.setText("⚠️  Position derrière la caméra")
            return

        pos3d = cam_pos + t * ray_dir
        x, y, z = float(pos3d[0]), float(pos3d[1]), float(pos3d[2])

        # ── Menu contextuel : choisir l'élément à insérer ────────────
        menu = QMenu(self)
        titre = menu.addAction(
            f"Ajouter un atome en ({x:.2f}, {y:.2f}, {z:.2f})"
        )
        titre.setEnabled(False)
        menu.addSeparator()
    
        for elt, couleur_rgb in CPK_COLOR.items():
            pix = QPixmap(14, 14)
            r, g, b = int(couleur_rgb[0]*255), int(couleur_rgb[1]*255), int(couleur_rgb[2]*255)
            pix.fill(QColor(r, g, b))
            menu.addAction(QIcon(pix), elt)

        action_choisie = menu.exec(self.mapToGlobal(pos_ecran))

        if action_choisie is None or action_choisie is titre:
            return

        nouvel_elt = action_choisie.text()

        # ── Insérer à la fin de la liste ─────────────────────────────
        idx = len(self.atoms)
        nouvel_atome = {'elt': nouvel_elt, 'x': x, 'y': y, 'z': z}
        self.atoms.append(nouvel_atome)

        # Sauvegarder pour Ctrl+Z
        self.historique.append(('add', idx, nouvel_atome))

        print(f"➕  Ajouté : [{idx}] {nouvel_elt}  "
              f"({x:.3f}, {y:.3f}, {z:.3f})")

        self._rafraichir()
    def _supprimer(self, pos_ecran):
        """Supprime l'atome le plus proche du point cliqué."""
        if not self.atoms:
            self.status.setText("⚠️  Aucun atome à supprimer")
            return

        idx, dist_min = self._atome_le_plus_proche(pos_ecran)

        if dist_min > self.SEUIL:
            self.status.setText(
                f"⚠️  Aucun atome assez proche (dist NDC = {dist_min:.4f})"
            )
            return

        atm_sup = self.atoms.pop(idx)
        # Format historique suppression : ('delete', idx, atm)
        self.historique.append(('delete', idx, atm_sup))

        print(f"🗑️  Supprimé : [{idx}] {atm_sup['elt']}  "
              f"({atm_sup['x']:.3f}, {atm_sup['y']:.3f}, {atm_sup['z']:.3f})")
        self._rafraichir()

    # ─────────────────────────────────────────────────────────────
    # Ctrl + clic droit : changer la nature chimique d'un atome
    # ─────────────────────────────────────────────────────────────

    def _changer_element(self, pos_ecran):
        """
        Trouve l'atome le plus proche, affiche un menu contextuel
        avec tous les éléments de CPK_COLOR, et change l'élément
        si l'utilisateur en choisit un.
        """
        if not self.atoms:
            self.status.setText("⚠️  Aucun atome disponible")
            return

        idx, dist_min = self._atome_le_plus_proche(pos_ecran)

        if dist_min > SEUIL:
            self.status.setText(
                f"⚠️  Aucun atome assez proche (dist NDC = {dist_min:.4f})"
            )
            return

        atm = self.atoms[idx]
        elt_actuel = atm['elt']

        # ── Menu contextuel avec la liste des éléments CPK ───────
        menu = QMenu(self)
        menu.setTitle(f"Atome [{idx}] : {elt_actuel}")

        # Titre non cliquable
        titre = menu.addAction(f"Changer {elt_actuel} →")
        titre.setEnabled(False)
        menu.addSeparator()

        for elt, couleur_rgb in CPK_COLOR.items():
            # Icône colorée pour chaque élément
            pixmap = QPixmap(16, 16)
            r, g, b = int(couleur_rgb[0]*255), int(couleur_rgb[1]*255), int(couleur_rgb[2]*255)
            pixmap.fill(QColor(r, g, b))
            action = menu.addAction(QIcon(pixmap), elt)
            # Marquer l'élément actuel
            if elt == elt_actuel:
                action.setCheckable(True)
                action.setChecked(True)

        # ── Afficher le menu à la position du curseur ────────────
        action_choisie = menu.exec(self.mapToGlobal(pos_ecran))

        if action_choisie is None or action_choisie is titre:
            return  # Annulé

        nouvel_elt = action_choisie.text()
        if nouvel_elt == elt_actuel:
            return  # Pas de changement

        # ── Appliquer le changement ───────────────────────────────
        # Sauvegarder l'état avant modification pour Ctrl+Z
        self.historique.append(('change', idx, elt_actuel))

        self.atoms[idx]['elt'] = nouvel_elt
        print(f"⚗️  Changement : [{idx}] {elt_actuel} → {nouvel_elt}  "
              f"({atm['x']:.3f}, {atm['y']:.3f}, {atm['z']:.3f})")

        self._rafraichir()

    # ─────────────────────────────────────────────────────────────
    # Méthode de picking partagée (suppression + changement)
    # ─────────────────────────────────────────────────────────────

    def _atome_le_plus_proche(self, pos_ecran):
        """
        Projette tous les atomes en 2D et retourne (idx, dist_min).
        Factorisé pour être utilisé par _supprimer et _changer_element.
        """
        cam_pos = np.array(self.vp.camera_pos, dtype=float)
        cam_dir = np.array(self.vp.camera_dir, dtype=float)
        cam_dir /= np.linalg.norm(cam_dir)

        z_world = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(cam_dir, z_world)) > 0.99:
            z_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(cam_dir, z_world)
        right /= np.linalg.norm(right)
        up = np.cross(right, cam_dir)
        up /= np.linalg.norm(up)

        fov_rad = float(self.vp.fov)
        if fov_rad <= 0:
            fov_rad = math.radians(45.0)
        tan_fov = math.tan(fov_rad / 2.0)

        w = self.width()
        h = self.height()
        aspect = w / h if h > 0 else 1.0

        cx = (pos_ecran.x() / w) * 2.0 - 1.0
        cy = 1.0 - (pos_ecran.y() / h) * 2.0

        dist2d = []
        for atm in self.atoms:
            p = np.array([atm['x'], atm['y'], atm['z']], dtype=float)
            v = p - cam_pos
            depth = np.dot(v, cam_dir)
            if depth <= 1e-6:
                dist2d.append(float('inf'))
                continue
            px = np.dot(v, right) / (depth * tan_fov * aspect)
            py = np.dot(v, up)    / (depth * tan_fov)
            dist2d.append(math.hypot(px - cx, py - cy))

        idx = int(np.argmin(dist2d))
        return idx, dist2d[idx]

    # ─────────────────────────────────────────────────────────────
    # Ctrl+Z : annulation (suppression ET changement d'élément)
    # ─────────────────────────────────────────────────────────────
    def annuler(self):
        if not self.historique:
            self.status.setText("⚠️  Rien à annuler")
            return

        action = self.historique.pop()

        if action[0] == 'change':
            _, idx, elt_ancien = action
            elt_actuel = self.atoms[idx]['elt']
            self.atoms[idx]['elt'] = elt_ancien
            print(f"↩️  Annulé : [{idx}] {elt_actuel} → {elt_ancien}")

        elif action[0] == 'delete':
            _, idx, atm = action
            self.atoms.insert(idx, atm)
            print(f"↩️  Annulé : réinsertion de {atm['elt']} à l'index {idx}")

            # ✅ NOUVEAU : annuler un ajout
        elif action[0] == 'add':
            _, idx, atm = action
            self.atoms.pop(idx)
            print(f"↩️  Annulé : suppression de {atm['elt']} ajouté en [{idx}]")

        self._rafraichir()
    

    # ─────────────────────────────────────────────────────────────
    # Reconstruction du pipeline après modification
    # ─────────────────────────────────────────────────────────────

    def _rafraichir(self):
        old_pipeline = self.pipeline_ref[0]
        old_pipeline.remove_from_scene()

        new_pipeline = build_pipeline(self.atoms)
        new_pipeline.add_to_scene()
        self.pipeline_ref[0] = new_pipeline

        n = len(self.atoms)
        h = len(self.historique)
        self.status.setText(
            f"{n} atomes   |   Shift+clic gauche : supprimer   |   "
            f"Shift+clic droit : ajouter   |   "
            f"Ctrl+clic droit : changer élément   |   "
            f"Ctrl+Z : annuler ({h} action{'s' if h > 1 else ''})   |   Ctrl+E : liste"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fenêtre principale
# ═══════════════════════════════════════════════════════════════════════════

class FenetreViewer(QWidget):
    """
    Layout :
        ┌──────────────────────────────┐
        │  Widget OVITO  (viewport 3D) │
        │  + PickOverlay par-dessus    │
        ├──────────────────────────────┤
        │  Barre de statut             │
        └──────────────────────────────┘
    """

    def __init__(self, fichier, atoms, pipeline, vp):
        super().__init__()
        self.setWindowTitle(f"view_mol — {fichier}")
        self.resize(900, 700)
        self.atoms = atoms            # référence partagée avec l'overlay
        self.vp = vp  


        # Référence mutable sur le pipeline (pour _rafraichir)
        self.pipeline_ref = [pipeline]

        # ── Barre de statut ─────────────────────────────────────
        self.status = QLabel(
            f"{len(atoms)} atomes   |   Shift+clic gauche : supprimer   |   "
            f"Shift+clic droit : ajouter   |   "
            f"Ctrl+clic droit : changer élément   |   Ctrl+Z : annuler   |   Ctrl+E : liste"
        )
        self.status.setStyleSheet(
            "background:#1a1d2e; color:#94a3b8; padding:4px 8px; font-size:11px;"
        )

        # ── Widget OVITO ─────────────────────────────────────────
        self.vp_widget = ovito.gui.create_qwidget(
            vp,
            show_orientation_indicator=True,
            show_title=False,
            parent=self,
        )

        # ── Overlay transparent ──────────────────────────────────
        self.overlay = PickOverlay(
            parent      = self.vp_widget,   # enfant du widget OVITO
            vp          = vp,
            atoms_ref   = atoms,
            status_label= self.status,
            pipeline_ref= self.pipeline_ref,
        )
        self.overlay.resize(self.vp_widget.size())

        # ── Layout vertical ──────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.vp_widget, stretch=1)
        layout.addWidget(self.status)

    def _to_HBPy(self):
        mol=HBPy.Molecule.Crystal.Crystal()
        for atm in self.atoms:
            q=np.array([atm['x'],atm['y'],atm['z']])
            mol.add_atom(elt=atm['elt'],q=q)

        #for a,b in zip(mol.atoms,self.atoms):
        #    print(a.mass,a.q,b)
        return mol

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Garder l'overlay à la même taille que le widget OVITO
        self.overlay.resize(self.vp_widget.size())

    def keyPressEvent(self, event: QKeyEvent):
        if (event.key() == Qt.Key.Key_Z and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.overlay.annuler()
        elif (event.key() == Qt.Key.Key_E and
              event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._afficher_atomes()
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_1:
                self._aligner_axe(0)   # axe 1 : plus faible inertie (axe le plus LONG)
            elif event.key() == Qt.Key.Key_2:
                self._aligner_axe(1)   # axe 2 : inertie intermédiaire
            elif event.key() == Qt.Key.Key_3:
                self._aligner_axe(2)   # axe 3 : plus forte inertie (axe le plus COURT)
        else:
            super().keyPressEvent(event)

    def _afficher_atomes(self):
        """
        Ctrl+E : affiche une fenêtre avec la liste complète des atomes restants.
        Colonnes : Index | Élément | x | y | z
        Avec comptage par espèce en bas.
        """
        # ── Affichage terminal ────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"Atomes restants : {len(self.atoms)}")
        print(f"{'─'*50}")
        for i, atm in enumerate(self.atoms):
            print(f"{atm['elt']:3s}  {atm['x']:10.4f}  {atm['y']:10.4f}  {atm['z']:10.4f}")
        print(f"{'─'*50}")
        comptage_term = {}
        for atm in self.atoms:
            comptage_term[atm['elt']] = comptage_term.get(atm['elt'], 0) + 1
        for elt, n in sorted(comptage_term.items()):
            pct = n / len(self.atoms) * 100 if self.atoms else 0
            print(f"  {elt:3s} : {n:4d}  ({pct:.1f}%)")
        print(f"{'─'*50}\n")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Atomes restants — {len(self.atoms)} au total")
        dlg.resize(520, 500)
        dlg.setStyleSheet("background:#1a1d2e; color:#e2e8f0;")

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Tableau ───────────────────────────────────────────────
        table = QTableWidget(len(self.atoms), 5)
        table.setHorizontalHeaderLabels(["Index", "Élément", "x (Å)", "y (Å)", "z (Å)"])
        table.setStyleSheet("""
            QTableWidget {
                background: #252840;
                color: #e2e8f0;
                gridline-color: #2d3148;
                border: 1px solid #2d3148;
                font-size: 12px;
            }
            QHeaderView::section {
                background: #1a1d2e;
                color: #a78bfa;
                padding: 4px;
                border: 1px solid #2d3148;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background: #2d3058;
            }
        """)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Remplir les lignes
        for row, atm in enumerate(self.atoms):
            elt = atm['elt']

            # Couleur CPK de l'élément
            cpk = CPK_COLOR.get(elt, (1.0, 1.0, 1.0))
            r, g, b = int(cpk[0]*255), int(cpk[1]*255), int(cpk[2]*255)
            couleur_qt = QColor(r, g, b)

            # Icône colorée pour l'élément
            pix = QPixmap(14, 14)
            pix.fill(couleur_qt)
            icone = QIcon(pix)

            items = [
                QTableWidgetItem(str(row)),
                QTableWidgetItem(icone, f"  {elt}"),
                QTableWidgetItem(f"{atm['x']:.4f}"),
                QTableWidgetItem(f"{atm['y']:.4f}"),
                QTableWidgetItem(f"{atm['z']:.4f}"),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)

        layout.addWidget(table)

        # ── Comptage par espèce ───────────────────────────────────
        comptage = {}
        for atm in self.atoms:
            comptage[atm['elt']] = comptage.get(atm['elt'], 0) + 1

        # Ligne de résumé
        parties = []
        for elt in sorted(comptage):
            n = comptage[elt]
            pct = n / len(self.atoms) * 100 if self.atoms else 0
            parties.append(f"{elt}: {n} ({pct:.1f}%)")

        resume = QLabel("  Composition :  " + "   ".join(parties))
        resume.setStyleSheet(
            "background:#252840; color:#a78bfa; padding:6px 10px; "
            "border-radius:4px; font-size:11px; font-weight:bold;"
        )
        layout.addWidget(resume)

        # ── Bouton Fermer ─────────────────────────────────────────
        btn = QPushButton("Fermer")
        btn.setStyleSheet(
            "background:#2d3058; color:#e2e8f0; border:1px solid #4c4f8a; "
            "border-radius:4px; padding:6px 20px; font-size:12px;"
        )
        btn.clicked.connect(dlg.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        dlg.exec()

    def _aligner_axe(self, idx_axe):
        """
        Ctrl+1/2/3 : orienter la caméra perpendiculairement à l'axe principal idx_axe.
        
        Convention OVITO :
        camera_dir = direction de visée (vers la molécule)
        camera_pos = position de la caméra dans l'espace
        
        Axe 0 → moment le plus faible  → axe le plus LONG de la molécule
        Axe 1 → moment intermédiaire
        Axe 2 → moment le plus élevé   → axe le plus COURT
        """
        if not self.atoms:
            self.status.setText("⚠️  Aucun atome — impossible d'aligner")
            return
        print(idx_axe)
        mol=self._to_HBPy()
        print(mol.Main_Axis())

        #moments, axes, centre = axes_principaux(self.atoms)
        moments, axes, centre = mol.moments,mol.axis,mol.MC

        # L'axe voulu : colonne idx_axe de la matrice axes
        axe = axes[:, idx_axe]

        # Distance caméra → centre de masse (conserver le zoom actuel)
        cam_pos_actuel = np.array(self.vp.camera_pos, dtype=float)
        distance = np.linalg.norm(cam_pos_actuel - centre)
        if distance < 1.0:
            distance = 30.0   # valeur par défaut si caméra au centre

        # Placer la caméra le long de l'axe, regarder vers le centre
        self.vp.camera_pos = tuple(centre + axe * distance)
        self.vp.camera_dir = tuple(-axe)

        # Recentrer le zoom
        self.vp.zoom_all((self.vp_widget.width(), self.vp_widget.height()))

        noms = {0: "long (axe 1)", 1: "intermédiaire (axe 2)", 2: "court (axe 3)"}
        self.status.setText(
            f"👁  Vue alignée sur l'axe {noms[idx_axe]}   "
            f"|  moment = {moments[idx_axe]:.1f} u·Å²"
        )

        print(f"\n{'─'*50}")
        print(f"Alignement axe {idx_axe+1}")
        print(f"  Direction : {axe}")
        print(f"  Moment    : {moments[idx_axe]:.2f} u·Å²")
        print(f"  Centre    : {centre}")
        print(f"{'─'*50}")
# ═══════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Visualiseur OVITO avec suppression interactive (Shift+clic)',
        epilog="Exemple : python view_molv0.5.py --file NP.xyz"
    )
    parser.add_argument('--file',
                        required=True,
                        help='Fichier contenant les coordonnées')
    parser.add_argument('--format',
                        default="xyz",
                        choices=["xyz", "ORCA_input"],             # ← valeurs acceptées
                        help='Fichier contenant les coordonnées')
    args = parser.parse_args()

    # ── Lecture ──────────────────────────────────────────────────
    atoms = read_file(args)
    print(f"📂  {len(atoms)} atomes chargés depuis {args.file}")

    # ── Pipeline OVITO ───────────────────────────────────────────
    pipeline = build_pipeline(atoms)
    pipeline.add_to_scene()

    # ── Qt ───────────────────────────────────────────────────────
    app = QApplication.instance() or QApplication(sys.argv)

    vp = ovito.vis.Viewport(
        type=ovito.vis.Viewport.Type.Perspective,
        camera_dir=(2, 1, -1),
    )

    fenetre = FenetreViewer(args.file, atoms, pipeline, vp)
    fenetre.show()
    fenetre.raise_()

    vp.zoom_all((fenetre.vp_widget.width(), fenetre.vp_widget.height()))

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
