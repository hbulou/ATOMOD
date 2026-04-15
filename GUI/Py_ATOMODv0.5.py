import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
import time
import sys

from HB_ATOMOD_GUI import Ui_MainWindow
from HEAS import HEAS


from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (QMainWindow,QApplication,QTableWidgetItem,QFileDialog,
                             QInputDialog,QStatusBar,
                             QMessageBox,QProgressBar,QAbstractItemView,QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (QKeySequence,QIntValidator, QGuiApplication,QCloseEvent)

import shutil
from pathlib import Path

import paramiko
import pandas as pd
from pandasmodel import PandasModel
import pyqtgraph as pg
#from AtomTableModel import AtomTableModel  # ou collez la classe ci-dessus dans ce fichier
from collections import defaultdict

#sys.path.append('/home/bulou/src/lib/site-packages/')
from HBPy.Molecule.Crystal import Crystal
from HBPy.Molecule.ForceField import ForceField
import HBPy.Molecule.Atom
from HBPy.DataViewer.XYplot import DatViewerWidget

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

#from PyQt6.QtCore import QTimer
import random
# Create an orthogonal unit cell of hBN using a graphene constructor.


from ase import Atoms,Atom  # juste pour construire l'objet

from ase.lattice.hexagonal import Graphene

sys.path.append('../lib/')
from abtem.atoms import orthogonalize_cell
from abtem.visualize import show_atoms
from  abtem  import  PlaneWave, CTF
import abtem
# pour créer un objet doté d’attributs (et sous-attributs) sans définir une classe explicite.
from types import SimpleNamespace

from PIL import Image as PILImage
import os
from glob import glob
import torch
from torchvision import transforms

import cv2

import tensorflow as tf
#from tensorflow.keras.layers import Input,Conv2D,Dropout,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.models import Model,load_model


from ATOMOD.ATOMOD import CustomDataGenerator,UNet,ImageSamplingCallback
from PyFEFF.FEFF import (FEFF)

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ==========================================================================================
# CONSTANTES DE CONFIGURATION
# ==========================================================================================
# Écran par défaut

class Config:
    DEFAULT_SCREEN_INDEX = 0  # Écran secondaire si disponible
    #
    TOLERANCE=1.0e-8
    RADIUS=.5
    FEFF_PGM = [
        "rdinp",
        "atomic",
        "dmdw",
        "pot",
        "opconsat", 
        "screen",
        "xsph",
        "fms",
        "mkgtr",
        "path", 
        "genfmt",
        "ff2x",
        "sfconv",
        "compton",
        "eels",
        "ldos"
    ]


    
# ==========================================================================================
# TOOLS
# ==========================================================================================
def get_z_plane(z_coords):
    # 1. Charger vos données (remplacez par la lecture de votre fichier XYZ)
    # Supposons que 'z_coords' est un array numpy contenant toutes vos cotes z
    #z_coords = np.loadtxt("data/xyz/NP_2050.xyz", skiprows=2, usecols=3) # Exemple pour format XYZ standard
    #print(z_coords)
    # 2. Calculer le KDE (densité de probabilité)
    density = gaussian_kde(z_coords, bw_method=0.05) # Ajuster bw_method selon le bruit
    z_range = np.linspace(min(z_coords), max(z_coords), 1000)
    z_density = density(z_range)
    
    # 3. Trouver les pics
    peaks, _ = find_peaks(z_density, height=np.max(z_density)*0.1)
    z_planes = z_range[peaks]
    
    # 4. Visualisation
    #plt.plot(z_range, z_density)
    #plt.plot(z_planes, z_density[peaks], "x")
    #plt.title(f"Cotes des plans détectées : {z_planes}")
    #plt.xlabel("z")
    #plt.ylabel("Densité")
    #plt.show()
    
    print("Cotes des plans :", z_planes)
    d_mean=0.0
    for i in range(len(z_planes)-1):
        d=z_planes[i+1]-z_planes[i]
        #print(d)
        d_mean+=d
    d_mean=d_mean/(len(z_planes)-1)
    print("<d>=",d_mean)
    return z_planes,d_mean

def mk_mean(series_list):
    """
    Méthode  : Interpolation sur une grille commune.
    
    Stratégie :
    1. Trouver la plage d'énergie commune à toutes les séries
    2. Créer une grille uniforme sur cette plage
    3. Interpoler chaque série sur cette grille
    4. Moyenner
    
    Args:
        series_list: Liste de tuples (energy, intensity)
    
    Returns:
        energy_common, intensity_mean, intensity_std
    """
    # Trouver la plage commune (intersection de toutes les séries)
    energy_min = max(serie[0].min() for serie in series_list)
    energy_max = min(serie[0].max() for serie in series_list)
    
    print(f"Plage commune: [{energy_min:.2f}, {energy_max:.2f}]")
    
    # Créer une grille uniforme
    n_points = len(series_list[0][0])  # Utilise le nombre de points de la première série
    energy_common = np.linspace(energy_min, energy_max, n_points)
    
    # Interpoler chaque série sur la grille commune
    interpolated_intensities = []
    
    for energy, intensity in series_list:
        # Interpolation linéaire (ou 'cubic' pour plus de lissage)
        f = interp1d(energy, intensity, kind='linear', fill_value='extrapolate')
        intensity_interp = f(energy_common)
        interpolated_intensities.append(intensity_interp)
    
    # Convertir en array pour calculs vectorisés
    interpolated_intensities = np.array(interpolated_intensities)
    
    # Calculer moyenne et écart-type
    intensity_mean = np.mean(interpolated_intensities, axis=0)
    intensity_std = np.std(interpolated_intensities, axis=0)
    
    return energy_common, intensity_mean, intensity_std

    


class XAS:
    def __init__(self):
        self.energy=[]
        self.chi=[]
        self.checkboxes=None
        self.idx_curve=-1
# ============================================================================================
class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)

    def save_figure(self, *args):
        # On définit le nom par défaut ici
        original_get_filename = self.canvas.get_default_filename
        
        # On surcharge la fonction pour renvoyer notre nom personnalisé
        self.canvas.get_default_filename = lambda: os.path.join(Path.cwd(), f"img.png")
        
        try:
            # On appelle la méthode originale qui ouvrira la fenêtre de dialogue
            super().save_figure(*args)
        finally:
            # On remet la fonction d'origine pour éviter les effets de bord
            self.canvas.get_default_filename = original_get_filename


def _display_img(X,title="None"):
    cv2.imshow(title, X)
    # --- 3. Attendre l'entrée utilisateur ---
    # Argument '0' signifie d'attendre indéfiniment qu'une touche soit pressée.
    # Si vous mettez '1000', cela attendra 1000 millisecondes (1 seconde) puis continuera.
    cv2.waitKey(0) 
    # --- 4. Fermer les fenêtres ---
    # Nettoie la mémoire et ferme toutes les fenêtres d'affichage OpenCV
    cv2.destroyAllWindows()

def _info_img(X):
    # 1. Dimensions (Hauteur, Largeur, Canaux)
    H, L = X.shape[:2]  # On prend les 2 premières valeurs
    Canaux = X.shape[2] if len(X.shape) == 3 else 1
    
    print(f"--- Informations sur l'Image ---")
    print(f"Format NumPy (Shape): {X.shape}")
    print(f"Hauteur (H) : {H} pixels")
    print(f"Largeur (L) : {L} pixels")
    print(f"Nombre de Canaux : {Canaux}")
    
    if Canaux == 3:
        print("Type d'image : Couleurs (BGR)")
    elif Canaux == 1:
        print("Type d'image : Niveaux de Gris")
def clear_layout(widget: QtWidgets.QWidget):
    """Supprime tous les widgets et layouts à l’intérieur d’un conteneur Qt."""
    layout = widget.layout()
    if layout is not None:
        # Supprimer tous les widgets enfants du layout
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)  # détache le widget du layout
            else:
                sublayout = item.layout()
                if sublayout is not None:
                    clear_layout(sublayout)
        # Supprimer le layout lui-même
        QtWidgets.QWidget().setLayout(layout)


# ─────────────────────────────────────────────────────────────────────────────
# Fenêtre secondaire qui contient le DatViewerWidget
# ─────────────────────────────────────────────────────────────────────────────

class DatViewerWindow(QMainWindow):
    """
    Fenêtre externe dédiée au DatViewerWidget.
    Elle informe la fenêtre principale quand elle est fermée via
    le callback on_close_callback, afin de resynchroniser le bouton.
    """

    def __init__(self, on_close_callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DAT File Viewer — Fenêtre externe")
        self.resize(1200, 750)
        self.on_close_callback = on_close_callback

        # ── Widget central ────────────────────────────────────────────────
        self.viewer = DatViewerWidget(apply_stylesheet=True)

        # Barre de statut de cette fenêtre
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.viewer.status_message.connect(status_bar.showMessage)
        self.viewer.hide_internal_statusbar()

        self.setCentralWidget(self.viewer)
        self.setStyleSheet("QMainWindow { background-color: #1e2127; }")

    def closeEvent(self, event: QCloseEvent):
        """Notifie la fenêtre principale que cette fenêtre a été fermée."""
        self.on_close_callback()
        super().closeEvent(event)

############################################################################################
# Hérite de QMainWindow (fenêtre principale) et de Ui_MainWindow (interface graphique).
class MainApp(QMainWindow,Ui_MainWindow):  #  Crée une fenêtre principale vide
   
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)
        self.Config=Config()


        # ── Fenêtre externe (créée une fois, réutilisée) ──────────────────
        # On passe _on_viewer_window_closed comme callback pour être notifié
        # quand l'utilisateur ferme la fenêtre via la croix OS.
        self._viewer_window = DatViewerWindow(
            on_close_callback=self._on_viewer_window_closed,
            parent=None,   # None = fenêtre de premier niveau indépendante
        )
        self.data_viewer.clicked.connect(self._toggle_viewer_window)
        #self.molecule = None
        self.H=256
        self.W=256
        self.LE_H.setText(f"{self.H}")
        self.LE_W.setText(f"{self.W}")
        self.LE_H.textChanged.connect(self.update_lineedit)
        self.LE_W.textChanged.connect(self.update_lineedit)
        self.nz=10
        # actions du menu
        self.actionQuit.triggered.connect(self.close)
        self.actionQuit.setShortcut(QKeySequence("Ctrl+Q"))
        
        # partie affichage Etot=f(step) dans l'onglet "2-OPTIMIZE"
        # plotWidget vient du widget promu
        self.plot = self.plotWidget
        self.plot.addLegend()
        self.curve = self.plot.plot(pen=pg.mkPen('b', width=2), name="Total energy")

        #self.plot_exafs.addLegend()
        #self.curve_exafs = self.plot_exafs.plot(pen=pg.mkPen('b', width=2), name="exfas")
        
        # 3) Paramètres de connexion au serveur de calcul
        self.hostname = "hpc-login"
        self.username = "bulou"
        # Option A: mot de passe
        # password = "password"
        # Option B (recommandée): clé privée
        self.key_filename = str(Path.home() / ".ssh" / "id_rsa")



        # Dans __init__, elle appelle self.setupUi(self) pour charger
        # et afficher tous les widgets définis dans l’interface Qt Designer.

        self.setWindowTitle("Py_ATOMOD_v0.5")
        self.WD_lineedit_radius.setText(str(Config.RADIUS))
        #self.WD_lineedit_radius.textChanged.connect(self.new_NP)
        self.WD_lineedit_radius.returnPressed.connect(self.new_NP)


        self.WD_lineedit_seed.setText("0")
        #self.WD_lineedit_seed.textChanged.connect(self.new_NP)
        self.WD_lineedit_seed.returnPressed.connect(self.new_NP)
        self.WD_button_build.clicked.connect(self.new_NP)
        self.WD_button_save.clicked.connect(self.save_NP)
        
        self.WD_table_list_elt.setHorizontalHeaderLabels(["elt", "number","%"])

        #------------------------------------------------------------------------
        #
        # Buid part
        #

        self.WD_lineedit_configidx.setValidator(QIntValidator())
        
        self.elt=[self.elt_Fe,self.elt_Co,self.elt_Ni,self.elt_Cu,
                  self.elt_Ru,self.elt_Rh,self.elt_Pd,self.elt_Ag,
                  self.elt_Os,self.elt_Ir,self.elt_Pt,self.elt_Au]
        
        for elt in self.elt:
            radiobutton = getattr(self,f"elt_{elt.text()}")
            radiobutton.clicked.connect(self.get_composition)

                                
        self.get_composition()


        self.WD_Btn_save_session.clicked.connect(self.save_session)
        self.WD_btn_exchange.clicked.connect(self.exchange)
        self.WD_btn_mixing.clicked.connect(self.mixing)
        
        # bouton optimisation structurale
        self.WD_button_optimize.clicked.connect(lambda:  self.optimize_NP(tol=float(self.WD_opt_lineedit_tol.text())))
        self.WD_button_save_optimize.clicked.connect(self.save_NP)
        self.WD_opt_lineedit_tol.returnPressed.connect(lambda:  self.optimize_NP(tol=float(self.WD_opt_lineedit_tol.text())))

        # TEM image simulation part
        self.TEM_img=SimpleNamespace()
        self.TEM_img.name="image_tem.png"
        self.TEM_img.sampling=SimpleNamespace()
        self.TEM_img.sampling.dx=0.0
        self.TEM_img.sampling.dy=0.0
        self.TEM_img.sampling.dz=0.0
        self.WD_button_TEM_img.clicked.connect(self.abtem)
        self.WD_button_TEM_save.clicked.connect(self.saveTEM)
        #self.WD_lineedit_cellsize.textChanged.connect(self.abtem)
        # xyz -> slice
        self.PB_xyz2slice.clicked.connect(self.xyz2slice)

        # ATOMOD_training()
        self.PB_ATOMOD_Training.clicked.connect(self.ATOMOD_training)

        # ATOMOD_using()
        self.PB_load_TEM_img.clicked.connect(self.load_TEM_img)
        self.PB_extract_atomic_structure.clicked.connect(self.extract_atomic_structure)

        self.PB_ATOMOD_Training_starting_model.clicked.connect(self.load_model)
        
        # bouton pour sauver un NP
        # self.WD_save_button.clicked.connect(self.save_NP)
        # bouton pour charger un NP
        self.WD_load_button.clicked.connect(self.load_NP)
        self.WD_Btn_load.clicked.connect(self.load_NP)
        # bouton pour charger un NP
        self.WD_lineedit_rmt_directory.setText("/home2020/home/ipcms/bulou/workdir/HEAS")
        self.WD_lineedit_server.setText("hpc-login")
        self.WD_lineedit_rmt_xyzfile.setText("crystal_opt-pos-1.xyz")
        self.WD_lineedit_rmt_extension.setText("xyz")
        self.WD_lineedit_rmt_directory.returnPressed.connect(self.Update_rmt_list)
        self.WD_lineedit_rmt_extension.returnPressed.connect(self.Update_rmt_list)
        self.WD_load_button_rmt.clicked.connect(self.load_NP_rmt)
        self.pushButton.clicked.connect(self.Update_rmt_list)

        # ------------------------------------------------------------------------------
        # Table in onglet ???
        # ------------------------------------------------------------------------------
        self.filelist_model = PandasModel(pd.DataFrame())
        self.tableViewListFile.setSortingEnabled(False)  # au début
        self.tableViewListFile.setModel(self.filelist_model)
        self.tableViewListFile.setSortingEnabled(True)
        self.tableViewListFile.setAlternatingRowColors(True)
        self.tableViewListFile.horizontalHeader().setStretchLastSection(True)
        self.tableViewListFile.doubleClicked.connect(self.on_table_double_clicked)

        # ------------------------------------------------------------------------------------------------------
        # Table on the right giving the list of atoms in molecule
        # création & configuration de la table qui affiche les coordonnées des atomes
        # self.tableView_AtomList est le widget (table) qui affiche la liste des atomes composant "molecule"
        # il est important de bien comprendre la structure du code
        #  * self.molecule contient le modèle atomique à partir duquel on réalise l'ensemble  des opérations :
        #           - optimisation structurale & chimique
        #           - simulations d'images TEM
        #           - simulation des spectres XAS
        #           - calculs DFT
        #           - etc.
        #  * les représentations graphiques qui permettent de
        #           - visualiser à l'écran la molécule : NP_viewver. La commande pour associer self.molecule à NP_viwver est self.NP_viewer.set_molecule(self.molecule). NP_viewer est un objet de la classe MoleculeGLWidget définie dans moleculewidget.py. La fonction set_molecule() est définie dans moleculewidget.py. La màj de l'affichage du widget NP_viewer est faite à la fin de la fonction set_molecule().
        #           - visualiser la liste des atomes composants la molécule : self.tableView_AtomList. C'est un widget de la classe QTableVIew. Les données à afficher sont contenu dans self.atom_model, un objet de la classe PandasModel() définie dans pandasmodel.py. La méthode setData de la classe PandasModel() permet de mettre à jour self.atom_model.
        #     
        #       
        # ------------------------------------------------------------------------------------------------------
        self.atom_model= PandasModel(pd.DataFrame())          # objet contenant les coordonnées des atomes (données)
        self.tableView_AtomList.setSortingEnabled(False)      # déactivation de la possibilité de trier la table (sécurité pdt la configuration)
        self.tableView_AtomList.setModel(self.atom_model)     # connection des données à la table
        self.tableView_AtomList.setSortingEnabled(True)       # activation de la possibilité de trier la table (Clic sur en-tête → tri de la colonne)
        self.tableView_AtomList.setAlternatingRowColors(True) # apparence de la table : Lignes blanc/gris alternées
        self.tableView_AtomList.horizontalHeader().setStretchLastSection(True) # Apparence de la table : Dernière colonne s'étire
        self.tableView_AtomList.resizeColumnsToContents()                      # Apparence de la table : ajustement de la largeur
        self.tableView_AtomList.setEditTriggers(                               # Déclencheurs édition       DoubleClick | SelectedClick
            QAbstractItemView.EditTrigger.DoubleClicked |  # Double-clic pour éditer
            QAbstractItemView.EditTrigger.SelectedClicked   # Ou clic sur sélection
        )
        # lorsque l'on clique sur une des cellume de self.tableView_AtomList et que l'on modofie la valeur de la cellule, cela appelle la fonction setData() de la classe PandasModel().
        self.atom_model.dataModified.connect(self.on_atom_data_modified) # Connecter signal : si dataModified est acivé (emit)  → fonction self.on_atom_data_modfoed est appelée
                                                     
        """
        SCÉNARIO : L'utilisateur modifie une cellule
        
        1. [UTILISATEUR] Double-clique sur une cellule dans tableView_AtomList
        └─> La cellule devient éditable (curseur clignotant)

        2. [UTILISATEUR] Tape "2.5" et appuie sur Entrée

        3. [Qt FRAMEWORK] Détecte la fin de l'édition
        └─> Appelle AUTOMATIQUEMENT : model.setData(index, "2.5", EditRole) (class PandasModel)
   
        4. [VOTRE CODE] La méthode setData() de PandasModel s'exécute
        ├─> Récupère row et column_name depuis index
        ├─> Convertit "2.5" en float (2.5)
        ├─> Met à jour self._data.iloc[row, col] = 2.5
        ├─> Émet le signal : dataModified.emit(row, column_name, 2.5)
        └─> Retourne True (succès) ou False (échec)
        
        5. [Qt FRAMEWORK] Rafraîchit l'affichage de la cellule
        
        6. [VOTRE CODE] Le signal dataModified est capté par on_atom_data_modified()
        └─> Met à jour self.molecule.atoms[row].pos[0] = 2.5
        └─> Rafraîchit l'affichage 3D
        """





        
        self.WD_dial_x.setValue(int(self.NP_viewer.rot_x))
        self.WD_lineedit_dial_x.setText(f'{self.NP_viewer.rot_x}')
        self.WD_dial_x.valueChanged.connect(self.dial_changed)
        self.WD_lineedit_dial_y.setText(f'{self.NP_viewer.rot_y}')
        self.WD_dial_y.valueChanged.connect(self.dial_changed)

        self.plot_exafs = self.WD_PL_EXAFS
        self.WD_compute_XAS.clicked.connect(self.FEFF)
        self.WD_reset_XAS.clicked.connect(self.reset_all_FEFF_calculations)
        self.feff_pgm__checkboxes = {}  # Dictionnaire feff pgm -> QCheckBox
        for pgm in Config.FEFF_PGM:
            checkbox = QCheckBox(f"{pgm}")
            if pgm in ["rdinp","atomic","dmdw","pot","screen","xsph","path","genfmt","ff2x","sfconv","compton"]:
                checkbox.setChecked(True)  # Coché par défaut
            else:
                checkbox.setChecked(False)  # Coché par défaut
            #         checkbox.stateChanged.connect(self._on_feff_pgm_checkbox_changed)
            self.feff_pgm__checkboxes[pgm] = checkbox
            self.WD_VLY_FEFF_pgm_CB.addWidget(checkbox)
        self.exafs_curve_checkboxes={}
        self.curve_exafs=[]
        self.XAS={}
        self.feff=FEFF()
    ### ----------------------------------------------------------------------------------##
    def _toggle_viewer_window(self):
        """Ouvre la fenêtre si elle est cachée, la ferme sinon."""
        if self._viewer_window.isVisible():
            self._viewer_window.hide()
            self._set_state_closed()
        else:
            self._viewer_window.show()
            self._viewer_window.raise_()      # passer au premier plan
            self._viewer_window.activateWindow()
            self._set_state_open()
    def _on_viewer_window_closed(self):
        """Appelé par DatViewerWindow.closeEvent (croix OS)."""
        self._set_state_closed()

    def _set_state_open(self):
        self.data_viewer.setText("✕ Hide data viewer")
        #self._state_label.setText("● Visualiseur : ouvert")
        #self._state_label.setStyleSheet("color:#a6e3a1; font-size:9pt;")
        #self._status.showMessage("Fenêtre visualiseur ouverte.")

    def _set_state_closed(self):
        self.data_viewer.setText("📊 Show data viewer")
        #self._state_label.setText("● Visualiseur : fermé")
        #self._state_label.setStyleSheet("color:#f38ba8; font-size:9pt;")
        #self._status.showMessage("Fenêtre visualiseur fermée.")

    ### ----------------------- END of Data Viewer methods ------------------------------
    def reset_all_FEFF_calculations(self):
        """Réinitialise tous les modules de calcul (FEFF, XAS, Optimisation, etc.)"""
        
        # --- 1. Nettoyage de l'interface graphique (Widgets dynamiques) ---
        # Pour les checkboxes EXAFS/FEFF
        if hasattr(self, 'WD_GD_exafs_curve'):
            clear_layout(self.WD_GD_exafs_curve)
        
        # --- 2. Purge des données en mémoire ---
        self.XAS = {}
        self.exafs_curve_checkboxes = {}
        self.composition = []
    
        # --- 3. Nettoyage des Graphiques (PyQtGraph) ---
        # On efface toutes les courbes des différents plots
        if hasattr(self, 'plot_exafs'):
            self.plot_exafs.clear()
        if hasattr(self, 'plot'): # Le plot d'optimisation
            self.plot.clear()
            # Si tu avais une légende, il faut parfois la recréer après un clear()
            # self.plot.addLegend() 
        
        # --- 4. Réinitialisation des listes de courbes ---
        self.curve_exafs = []
        if hasattr(self, 'curve'):
            self.curve = None

        # --- 5. Rafraîchissement des tableaux de l'interface ---
        if hasattr(self, 'WD_table_list_elt'):
            self.WD_table_list_elt.setRowCount(0)
        
        # --- 6. Nettoyage des fichiers temporaires (Optionnel mais conseillé) ---
        for tmp_file in ['xmu.dat', 'feff.inp', 'paths.dat']:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass

        logger.info("Tous les calculs ont été réinitialisés.")
        self.statusBar().showMessage("Calculs réinitialisés.", 3000)
    ### ----------------------------------------------------------------------------------###
    def FEFF(self):
        print(f"####### FEFF PART #######################")
        if not hasattr(self, 'molecule') or self.molecule is None:
            self._show_error("Erreur",
                             f" Aucune structure atomique disponible.")
            return
        # ----------------------------------------
        # liste des atomes absorbeurs à considérer
        # ----------------------------------------
        list_ads=[]
        if "-" not in self.WD_le_selected_atom.text() and "all" not in self.WD_le_selected_atom.text():
            list_ads.append(int(self.WD_le_selected_atom.text()))
        else:
            if "-" in self.WD_le_selected_atom.text():
                idx=self.WD_le_selected_atom.text().split("-")
                list_ads = list(range(int(idx[0]), int(idx[-1])+1))

            if "all" in self.WD_le_selected_atom.text():
                idx=self.WD_le_selected_atom.text().split("-")
                list_ads = list(range(len(self.molecule.atoms)))
        print(f"list of absorber atoms: {list_ads}")
        # -----------------------------------------------------            
        # calcul des spectres pour chaque atome dans list_ads
        # -----------------------------------------------------            
        for  absorber_idx in list_ads:
            print(100*"#")
            print(f"### FEFF ### absorber {absorber_idx}")
            print(100*"#")
            self.XAS[str(absorber_idx)]=XAS()
            rpath=float(self.WD_le_rpath.text())
            
            self.feff.create_input_file(self.molecule,
                                        absorber_idx=absorber_idx,
                                        rpath=rpath)
            shutil.copy2("feff.inp",f"feff_{self.molecule.atoms[absorber_idx].elt}_{absorber_idx}.inp")

            self.feff.run(self.feff_pgm__checkboxes)

            try:
                energy, chi = np.loadtxt('xmu.dat', comments='#', usecols=(0, 4), unpack=True)
                xmu = Path("xmu.dat")
                # Renomme (ou déplace) le fichier instantanément
                xmu.rename(Path(f"xmu_{self.molecule.atoms[absorber_idx].elt}_{absorber_idx}.dat"))
                
                self.XAS[str(absorber_idx)].energy=energy
                self.XAS[str(absorber_idx)].chi=chi
                # Vérification rapide
                print(f"Premières valeurs colonne 2 (Energy) : {energy[:5]}")
                print(f"Premières valeurs colonne 3 (chi) : {chi[:5]}")

            except Exception as e:
                print(f"Une erreur est survenue : {e}")
                
            self.XAS[str(absorber_idx)].idx_curve=len(self.curve_exafs)            
            self.curve_exafs.append(self.plot_exafs.plot(pen=pg.mkPen('b', width=2), name=""))
            self.curve_exafs[-1].setData(self.XAS[str(absorber_idx)].energy,self.XAS[str(absorber_idx)].chi)
            
            self.XAS[str(absorber_idx)].checkbox = QCheckBox(f"{absorber_idx} ({self.molecule.atoms[absorber_idx].elt})")
            self.XAS[str(absorber_idx)].checkbox.setChecked(True)  # Coché par défaut
            self.XAS[str(absorber_idx)].checkbox.stateChanged.connect(self._on_feff_curve_checkbox_changed)
            self.exafs_curve_checkboxes[str(absorber_idx)] = self.XAS[str(absorber_idx)].checkbox


        # Initialise automatiquement avec une liste vide
        series = defaultdict(list)

        for  absorber_idx in list_ads:
            elt=self.molecule.atoms[absorber_idx].elt
            print(self.XAS[str(absorber_idx)].energy[0],self.XAS[str(absorber_idx)].energy[-1],len(self.XAS[str(absorber_idx)].energy))
            series[elt].append((self.XAS[str(absorber_idx)].energy, self.XAS[str(absorber_idx)].chi))

        total_items = len(list_ads) + len(series)
        ncol = int(np.ceil(np.sqrt(total_items))) if total_items > 0 else 1
        nrow=int(np.sqrt(len(list_ads)+len(series))/ncol)
        for  i,absorber_idx in enumerate(list_ads):
            idx_row=i//ncol
            idx_col=i%ncol
            self.WD_GD_exafs_curve.addWidget(self.XAS[str(absorber_idx)].checkbox,idx_row,idx_col)
        last_idx_row=idx_row
        last_idx_col=idx_col
        for i,elt in enumerate(series.keys()):
            print(elt)
            self.XAS[f"{elt}_mean"]=XAS()
            self.XAS[f"{elt}_mean"].energy,self.XAS[f"{elt}_mean"].chi, _ = mk_mean(series[elt])

            donnees_combinees = np.column_stack((self.XAS[f"{elt}_mean"].energy,self.XAS[f"{elt}_mean"].chi))
            np.savetxt(
                f'mean{elt}.dat',               # Le nom du fichier
                donnees_combinees,        # Le tableau 2D créé juste au-dessus
                fmt='%.6f',               # Le format (ici : 6 chiffres après la virgule)
                delimiter='    ',         # Le séparateur entre les colonnes (ici : 4 espaces)
                header='#energy       Mean_mu0', # (Optionnel) Ajoute un en-tête
                comments='# '             # (Optionnel) Le caractère pour commenter l'en-tête
            )

            
            self.XAS[f"{elt}_mean"].idx_curve=len(self.curve_exafs)            
            self.curve_exafs.append(self.plot_exafs.plot(pen=pg.mkPen('r', width=2), name=""))
            self.curve_exafs[-1].setData(self.XAS[f"{elt}_mean"].energy,self.XAS[f"{elt}_mean"].chi)
            self.XAS[f"{elt}_mean"].checkbox = QCheckBox(f"Mean ({elt})")
            self.XAS[f"{elt}_mean"].checkbox.setChecked(True)  # Coché par défaut
            self.XAS[f"{elt}_mean"].checkbox.stateChanged.connect(self._on_feff_curve_checkbox_changed)
            self.exafs_curve_checkboxes[f"{elt}_mean"] = self.XAS[f"{elt}_mean"].checkbox

            self.WD_GD_exafs_curve.addWidget(self.XAS[f"{elt}_mean"].checkbox,last_idx_row+1,i)
        for i,elt in enumerate(series.keys()):
            checkbox = QCheckBox(f"Show {elt}")
            checkbox.setChecked(True)  # Coché par défaut
            checkbox.stateChanged.connect(self._on_feff_curves_show)
            self.exafs_curve_checkboxes[f"{elt}_show"] = checkbox

            self.WD_GD_exafs_curve.addWidget(self.exafs_curve_checkboxes[f"{elt}_show"],last_idx_row+2,i)
        
        #print(len(self.curve_exafs))
        #FEFF_info(idx=self.WD_le_selected_atom.text())
        
        #FEFF_create_parameter_file("feff.inp",self.molecule)
        #calculator=FEFF_calculator(config=FEFF_config())

    def _on_feff_curves_show(self):
        for i,elt in enumerate(self.exafs_curve_checkboxes.keys()):
            print(20*"-")
            if "show" in elt:
                print(i,elt,self.exafs_curve_checkboxes[elt].isChecked())
                print(20*"-")
                for curve in self.XAS.keys():
                    if elt.split("_")[0] in self.XAS[curve].checkbox.text():
                        print(curve,self.XAS[curve].idx_curve,self.XAS[curve].checkbox.text())
                        if self.exafs_curve_checkboxes[elt].isChecked() :
                            self.curve_exafs[self.XAS[curve].idx_curve].setVisible(True)
                            self.XAS[curve].checkbox.setChecked(True)
                        else:
                            self.curve_exafs[self.XAS[curve].idx_curve].setVisible(False)
                            self.XAS[curve].checkbox.setChecked(False)
                #for j,curve in enumerate(self.exafs_curve_checkboxes.keys()):
                #    if elt.split("_")[0] in self.exafs_curve_checkboxes[curve].text() and "Show" not in self.exafs_curve_checkboxes[curve].text():
                #        print(self.exafs_curve_checkboxes[curve].text())
                #        if self.exafs_curve_checkboxes[elt].isChecked() :
                #            print(curve)
                #            #self.curve_exafs[curve].setVisible(True)
                #        else:
                #            print(curve)
                #            #self.curve_exafs[curve].setVisible(False)
        #self._on_feff_curve_checkbox_changed()
        self.plot_exafs.autoRange()
    def _on_feff_curve_checkbox_changed(self):
        #self.plot_exafs.clear()
        #print(len(self.curve_exafs))
        #print(len(self.exafs_curve_checkboxes))
        for curve in self.XAS.keys():
            idx_curve=self.XAS[curve].idx_curve
            print(f"idx_curve {idx_curve}")
            if self.curve_exafs[idx_curve].isVisible():
                print("La courbe est actuellement affichée.")
                if not self.XAS[curve].checkbox.isChecked():
                    self.curve_exafs[idx_curve].setVisible(False)
            else:
                print("La courbe est masquée.")
                if self.XAS[curve].checkbox.isChecked():
                    self.curve_exafs[idx_curve].setVisible(True)
        self.plot_exafs.autoRange()
                
    def exchange(self):
        self.molecule.exchange()
        self.atom_model.setDataFrame(self.molecule.to_df())
        self.NP_viewer.update()
        self.update_NP_info()
    def mixing(self):
        self.molecule.mixing(nexchange=int(self.WD_LE_mixing_nexchange.text()),seed=int(self.WD_LE_mixing_nexchange.text()))
        self.atom_model.setDataFrame(self.molecule.to_df())
        self.NP_viewer.update()
        self.update_NP_info()
    def save_session(self):
        print(f"Session saved!")
        
        # Capturer le contenu OpenGL
        image = self.NP_viewer.grabFramebuffer()
        
        # Sauvegarder
        success = image.save("top_view.png")

        # Sauvegarder avec PIL (simple et rapide)
        self.tem_current_simulation['pil_image'].save("TEM.png", quality=100)


    def on_atom_data_modified(self, field):
        """Appelée quand une cellule est modifiée"""
        print(f"fonction on_atom_data_modified()")
        print(field['idx'],field['row'],field['col'],field['val'])
        print("on_atom_data_modified(self, row, column_name, new_value):")
        if self.molecule is None:
            print("No molecule defined")
            return

        if field['col']==1:
            self.molecule.atoms[field['idx']].elt=field['val']
        else:
            self.molecule.atoms[field['idx']].q[field['col']-1]=field['val']


        for atm in self.molecule.atoms:
            print(atm.idx,atm.elt,atm.q)
        print(self.atom_model._df)
        self.molecule.get_element_distribution()
        self.NP_viewer.update()
        self.update_NP_info()
        
    def update_lineedit(self):
        self.H=int(self.LE_H.text())
        self.W=int(self.LE_W.text())
    def extract_atomic_structure(self):
        print(self.LE_TEM_img.text())
        img = PILImage.open(self.LE_TEM_img.text()).convert("L")               # grayscale
        H=self.H
        W=self.W
        # 1. Charger l'image (en niveaux de gris)
        # Assurez-vous que l'image est chargée comme un tableau numpy
        img = cv2.imread(self.LE_TEM_img.text(), cv2.IMREAD_GRAYSCALE) 

        if img is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

        # 2. Normaliser les pixels (si vous avez normalisé pendant l'entraînement, e.g., à [0, 1])
        # Vérifiez comment vous avez normalisé dans CustomDataGenerator. Souvent, c'est diviser par le max.
        img = img.astype(np.float32) / 255.0 

        # 3. Redimensionner ou Patcher (pour cet exemple, nous allons redimensionner *ou* supposer une image 128x128)
        if img.shape[0] != self.H or img.shape[1] != self.W:
            # Pour une démonstration simple, redimensionnons :
            # NOTE : Si vous utilisez le patching, cette partie serait beaucoup plus complexe.
            img_resized = cv2.resize(img, (self.W, self.H))
        else:
            img_resized = img
        # 4. Ajouter les dimensions Batch et Channel (Keras/TensorFlow attend le format (B, H, W, C))
        # B=1 (une image), C=1 (niveaux de gris)
        img_input = np.expand_dims(img_resized, axis=-1) # Ajoute la dimension C
        img_input = np.expand_dims(img_input, axis=0)  # Ajoute la dimension B

        # Assurez-vous d'avoir accès à la définition de votre UNet si vous utilisez une couche personnalisée ou une fonction de perte/métrique custom.
        # Si UNet est bien une classe/fonction standard, cela suffit souvent :
        try:
            self.model = load_model('unet_atomod_trained.h5')
        except ValueError as e:
            # Si load_model ne trouve pas la définition de la classe UNet, vous devrez la fournir :
            # loaded_model = load_model('unet_atomod_trained.h5', custom_objects={'UNet': UNet})
            print(f"Erreur de chargement du modèle : {e}")
            # ... gérez l'erreur ou chargez avec custom_objects si nécessaire

        
        # 1. Effectuer la prédiction
        # La sortie sera de forme (1, H, W, len(self.composition)*nz)
        prediction = self.model.predict(img_input)
        # prediction_map = prediction[0, ...] # On enlève la dimension Batch

        print(prediction.shape)
        print(prediction[0].shape)
        # Sauvegarder le canal 0
        nz=self.nz
        # prediction_map a la forme (H, W, len(self.composition)*nz)
        prediction_map = prediction[0]
        for iz in range(nz):
            segmentation_Atome_A = (prediction_map[..., iz] > 0.5).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmentation_Atome_A, 4, cv2.CV_32S)
            print(f"Nombre d'entités 'Atome A' détectées : {num_labels - 1}") # Souvent -1 car l'étiquette 0 est l'arrière-plan
            self.save_prediction_channel_as_png(prediction[0],
                                                iz,
                                                f"atom_A_probability_map_{iz}.png")
        print("DONE!")
    def save_prediction_channel_as_png(self,
                                       prediction_array,
                                       channel_index,
                                       output_filename="segmentation_channel_0.png"):
        """
        Transforme un canal spécifique de la carte de prédiction en une image PNG 
        en niveaux de gris (0-255).
        
        Args:
        prediction_array (np.ndarray): prediction[0] de forme (H, W, C).
        channel_index (int): L'indice du canal à visualiser.
        output_filename (str): Le nom du fichier de sortie.
        """

        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs("data/prediction")

        
        if prediction_array.ndim != 3:
            raise ValueError("L'entrée doit être de forme (H, W, C).")
        
        # --- 1. Extraction du canal ---
        # La carte de probabilité pour le canal sélectionné
        probability_map = prediction_array[..., channel_index] 
    
        # --- 2. Normalisation et Conversion en 8-bit (0-255) ---
        # Les probabilités sont de [0, 1]. Nous les multiplions par 255.
        # et les convertissons en entier non signé 8-bit (uint8)
        image_8bit = (probability_map * 255).astype(np.uint8)
    
        # --- 3. Sauvegarde en PNG (Utilisation de cv2) ---
        # cv2.imwrite est simple et efficace pour sauvegarder des matrices NumPy en images.
        cv2.imwrite(output_filename, image_8bit)
        print(f"Carte de probabilité (Canal {channel_index}) sauvegardée : {output_filename}")
    
        # --- Optionnel : Sauvegarde avec Matplotlib (souvent meilleure qualité pour les flottants) ---
        # plt.imsave(output_filename.replace(".png", "_mpl.png"), probability_map, cmap='gray')
    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier",
            "",
            "ImagesModel (*.h5 *.keras );;Tous (*.*)"
        )
        if filename:
            self.LE_ATOMOD_Training_starting_model.setText(filename)
    def load_TEM_img(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier",
            "",
            "Images (*.png *.jpg *.tif);;Tous (*.*)"
        )
        if filename:
            self.LE_TEM_img.setText(filename)
    def ATOMOD_training(self):
        print("ATOMOD_training")
        #self.image_paths = sorted(glob(os.path.join(image_dir, "*")))
        #if len(self.image_paths) == 0:
        #    raise ValueError(f"Aucune image trouvée dans {image_dir}")
        device=None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Utilisation de l'appareil :", device)
        
                    

        # # Liste fictive des IDs d'images
        # # En réalité, ceci serait généré par os.listdir(os.path.join(DATA_ROOT, 'images'))

        BATCH_SIZE=int(self.LE_batch_size.text())
        EPOCHS=int(self.LE_epochs.text())
        

        # sélection des images/masks pour l'entrainement
        train_IDs=[]
        for i in range(BATCH_SIZE):
            train_IDs.append(f'img_{i+1:04d}')

        # sélection des images/masks pours la validation
        val_IDs=[]
        for i in range(BATCH_SIZE,2*BATCH_SIZE):
            val_IDs.append(f'img_{i+1:04d}')
        print(train_IDs)
        print(val_IDs)
        #for sp in self.composition:
        #    f"{sp}_{i:04d}_{k:04d}.png"

        DATA_ROOT="data/train"

        # # --- 1. Initialisation des Générateurs ---
        train_generator = CustomDataGenerator(
             train_IDs, 
             DATA_ROOT, 
             (self.H, self.W), 
             BATCH_SIZE, 
             shuffle=True,
            composition=self.composition,
            nz=self.nz
         )
        
        validation_generator = CustomDataGenerator(
            val_IDs, 
            DATA_ROOT, 
            (self.H, self.W), 
            BATCH_SIZE, 
            shuffle=False,
            composition=self.composition,
            nz=self.nz
        )

        print("🆕 Création d'un nouveau modèle")
        self.model=UNet(self.H,self.W,len(self.composition)*self.nz)
                # --- 3. Compilation du Modèle ---
        if self.LE_ATOMOD_Training_starting_model.text().strip() and self.CB_ATOMOD_Training_starting_model.isChecked():
            print("OK : texte présent ET checkbox cochée")
            model_path = self.LE_ATOMOD_Training_starting_model.text()
            if os.path.exists(model_path):
                print("🔄 Reprise de l'entraînement depuis", model_path)
                self.model = load_model(model_path, compile=False)

                
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
            # Perte pour N classifications binaires indépendantes (H, W, N)
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)],
            run_eagerly=True
        )
        if self.LE_initial_epochs.text().strip():
            initial_epochs=int(self.LE_initial_epochs.text())
        else:
            initial_epochs=1
        self.model.summary() # Décommenter pour voir l'architecture et la forme de sortie (None, H, W, 10)

        # --- 4. Entraînement ---
        print("Démarrage de l'entraînement...")

        # --- Préparation du Callback pour la Visualisation ---
        # 1. Extraire un échantillon du générateur de validation
        # On récupère le premier lot (batch) du générateur de validation
        # Assurez-vous que votre CustomDataGenerator supporte l'indexation ([0]) ou utilisez next(iter(validation_generator))
        try:
            sample_batch_input, _ = validation_generator[0] 
        except Exception:
            # Si l'indexation n'est pas supportée, utilisez un itérateur :
            sample_batch_input, _ = next(iter(validation_generator))

        # 2. Sélectionner la première image du lot (Batch=1 pour la prédiction)
        # La forme doit être (1, H, W, C) pour la prédiction UNet

        intermediate_dir="data/train/intermediate"

        
        sample_input_for_callback = sample_batch_input[0:1]
        callbacks_list=[]
        callbacks_list.append(
            ImageSamplingCallback(
                sample_input_image=sample_input_for_callback, 
                class_channel_index=0,
                val_IDs=val_IDs,
                nz=self.nz,
                composition=self.composition,
                output_dir=intermediate_dir,
                H=self.H, 
                W=self.W)
        )
            
        #        Ce qui se passe à l’intérieur de fit#
        #
        #Pour chaque époque :
        #
        #for epoch in epochs:
        #    for batch in data:
        #        y_pred = model(x_batch, training=True)
        #        loss = loss_fn(y_batch, y_pred)
        #        gradients = tape.gradient(loss, weights)
        #        optimizer.apply_gradients(...)
        #        update_metrics()
        #    run_validation()
        #    callbacks.on_epoch_end()
        #
        # fit() encapsule tout cela automatiquement.

        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=EPOCHS,
            initial_epoch=initial_epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks_list
        )

        print("Entraînement terminé.")
        self.model.save('unet_atomod_trained.h5')
        print("Modèle sauvegardé sous 'unet_atomod_trained.h5'")
        
        # Ce fichier contient :
        #    l’architecture
        #    les poids
        #    l’optimizer avec son état interne (Adam, etc.)
        
        print("DONE!")
# ################################################################################
    def xyz2slice(self):
        """
        fonction permettant de passer du format xyz -> png
        """
        print("xyz2slice")
        print(self.composition)
        tmp_molecule=self.molecule.duplicate()
        tmp_molecule.origin_at(origin=np.array([-10.0,-10.0,-10.0]))
        tmp_molecule.get_structure()
        
        print(self.molecule.MC)
        nx=self.potential.shape[1]
        ny=self.potential.shape[2]
        nz=self.potential.shape[0]
        dx=self.TEM_img.sampling.dx
        dy=self.TEM_img.sampling.dy
        dz=self.TEM_img.sampling.dz



        z_coords=[]
        for atm in tmp_molecule.atoms:
            z_coords.append(atm.q[2])
        zp,dzmean=get_z_plane(z_coords)    
        print(zp,dzmean)
        dz=dzmean

        x = np.linspace(0.0, self.potential.extent[0], nx)
        y = np.linspace(0.0, self.potential.extent[1], ny)
        nz=len(zp)
        z = np.linspace(zp[0],zp[-1],nz)

        
        
        print(f"qmin={tmp_molecule.qmin}")
        print(f"qmax={tmp_molecule.qmax}")
        volumes = {}  # dict: espèce -> volume 3D
        for sp in self.composition:
            volumes[sp] = np.zeros((nx, ny, nz), dtype=float)
        
        sigma = 0.6  # en Å, largeur de la gaussienne ~ rayon atomique ou un peu moins
        for atom in tmp_molecule.atoms:
             sp = atom.elt
             vol = volumes[sp]
             # Position de l’atome
             ax, ay, az = atom.q[0], atom.q[1], atom.q[2]
             #     # Indices du voisinage à affecter (±3 sigma)
             ix_center = int((ax) / dx)
             iy_center = int((ay) / dy)
             iz_center = int((az) / dz)

             r = int(3 * sigma / dx)  # rayon en nombre de voxels

             ix_min = max(ix_center - r, 0)
             ix_max = min(ix_center + r + 1, nx)
             iy_min = max(iy_center - r, 0)
             iy_max = min(iy_center + r + 1, ny)
             iz_min = max(iz_center - r, 0)
             iz_max = min(iz_center + r + 1, nz)

        #     # Sous-grille locale
             Xsub = x[ix_min:ix_max]
             Ysub = y[iy_min:iy_max]
             Zsub = z[iz_min:iz_max]

             Xg, Yg, Zg = np.meshgrid(Xsub, Ysub, Zsub, indexing="ij")

             dx2 = (Xg - ax)**2
             dy2 = (Yg - ay)**2
             dz2 = (Zg - az)**2
             gauss = np.exp(-(dx2 + dy2 + dz2) / (2 * sigma**2))

             vol[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] += gauss



        output_dir = "data/train/prob_maps"
        os.makedirs(output_dir, exist_ok=True)
        xmin=0.0
        ymin=0.0
        zmin=0.0
        xmax=self.potential.extent[1]
        ymax=self.potential.extent[1]
        zmax=tmp_molecule.qmin[2]

        for sp in self.composition:
            vol=volumes[sp]
            nx, ny, nz = vol.shape

            # Optionnel : échelle globale fixe
            vmin = vol.min()
            vmax = vol.max()
            for k in range(nz):
                slice_z = vol[:, :, k]        # coupe dans le plan x-y
                z_value = zmin + k * dz       # valeur physique du z
                fig, ax = plt.subplots(figsize=(6, 6))  # carré pour être sûr
                im = ax.imshow(
                    slice_z.T,
                    origin='lower',
                    extent=[xmin, xmax, ymin, ymax],
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='nearest',
                    alpha=0.9
                )

                # impose ratio 1:1
                ax.set_aspect('equal')  # x et y même échelle

                # labels et titre
                ax.set_title(f"Coupe à z = {z_value:.2f} Å  (k={k})")
                ax.set_xlabel("x (Å)")
                ax.set_ylabel("y (Å)")
            
                # *** SUPPRESSION DES ÉLÉMENTS GRAPHIQUES ***
                ax.set_xticks([])   # pas de ticks x
                ax.set_yticks([])   # pas de ticks y
                ax.set_xlabel("")   # pas de labels
                ax.set_ylabel("")
                ax.set_title("")    # pas de titre
                ax.axis('off')      # supprime l’axe et le cadre
            
                #fig.colorbar(im, ax=ax, label="densité")

                # sauvegarde {int(self.WD_lineedit_configidx.text()):04d}
                filename = os.path.join(output_dir, f"img_{int(self.WD_lineedit_configidx.text()):04d}_{sp}_{k:04d}_{z_value:5.2f}.png")
                plt.savefig(filename,
                            dpi=150,
                            bbox_inches='tight',
                            transparent=True,
                            pad_inches=0.1,
                            facecolor='white')

                #plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
             
        
        
    def dial_changed(self):
        self.WD_lineedit_dial_x.setText(f'{self.WD_dial_x.value()}')
        self.NP_viewer.rot_x=self.WD_dial_x.value()
        self.WD_lineedit_dial_y.setText(f'{self.WD_dial_y.value()}')
        self.NP_viewer.rot_y=self.WD_dial_y.value()
        self.NP_viewer.update()
        self.update_NP_info()
    def saveTEM(self):
        x_axis, y_axis, z_axis = self.NP_viewer.get_axis_vectors()
        print(x_axis)
        print(y_axis)
        print(z_axis)
        R = self.NP_viewer.get_rotation_matrix()
        print(R)
        for atm in self.molecule.atoms:
            #print(atm.elt,bulou.Atom.Z_from_elt[atm.elt],atm.q)
            v=np.array([atm.q[0],atm.q[1],atm.q[2]])
            result=v@R
            atm.q[0]=result[0]
            atm.q[1]=result[1]
            atm.q[2]=result[2]
            print(atm.elt,result[0],result[1],result[2])
        self.NP_viewer.rot_x = 0.0
        self.NP_viewer.rot_y = 0.0
        self.NP_viewer.update()
        self.update_NP_info()

    ################################################################################
    #
    #       abtem
    #
    def abtem(self):

        output_dir = "data/train/images"
        os.makedirs(output_dir, exist_ok=True)

        # Crée une boîte vide de 10x10x10 Å
        # pour l'instant on passe par ASE pour fournir la structure à abtem
        cellsize=10.0*float(self.WD_lineedit_cellsize.text()) # taille de la zone à simuler
        atoms = Atoms(cell=[cellsize,cellsize,cellsize], pbc=True)
        
        #print(atoms)
        for atm in self.molecule.atoms:
            #print(atm.elt,bulou.Atom.Z_from_elt[atm.elt],atm.q)
            atoms += Atom(HBPy.Molecule.Atom.Z_from_elt[atm.elt], (atm.q[0],atm.q[1],atm.q[2]))
        #atoms.rotate(90, 'z', rotate_cell=True)
        #atoms.rotate(self.WD_dial_x.value(), 'x', rotate_cell=False)
        #atoms.rotate(self.WD_dial_x.value(), 'x', rotate_cell=True)

        # on récupère l'orientation de la nanoparticule. Le faisceau arrive perpendiculairement
        # au plan de l'image
        R = self.NP_viewer.get_rotation_matrix()
        # on positionne la nanoparticule relativement au faisceau incident
        atoms.positions = atoms.positions @ R # rotation active
        atoms.center()

        # en réalité nous n'avons pas besoin de mettre self.molecule à jour (translation)
        #for atm in atoms:
        #    self.molecule.atoms[atm.index].q[0]=atm.x
        #    self.molecule.atoms[atm.index].q[1]=atm.y
        #    self.molecule.atoms[atm.index].q[2]=atm.z
        #self.molecule.get_structure()
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        a=show_atoms(atoms,ax=ax1, title="Beam view", numbering=True, merge=False)
        a=show_atoms(atoms, ax=ax2, plane="xz", title="Side view", numbering=True,merge=False,legend=True)
        # 1) Le host est le QWidget posé dans QtDesigner
        host = self.WD_TEMview  # NE PAS ÉCRASER cette variable avec le canvas !
        clear_layout(host)
        # 3) Créer canvas (+ toolbar si tu veux)
        canvas = FigureCanvas(a[0])
        toolbar = CustomNavigationToolbar(canvas, self)
        #toolbar = NavigationToolbar(canvas, self)
        # 4) Récupérer ou créer le layout du host
        layout = host.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(host)
            layout.setContentsMargins(0, 0, 0, 0)
            host.setLayout(layout)
        # 5) Insérer dans la hiérarchie Qt
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        canvas.draw()

        # Nous avons fixé l'épaisseur de la tranche à la moitié de la hauteur de la cellule unitaire 4.08/2
        # et l'échantillonage dans le plan xy à 0.04 angstroem
        
        self.TEM_img.sampling.dx=0.04
        self.TEM_img.sampling.dy=0.04
        self.TEM_img.sampling.dz=4.08/2
        self.potential = abtem.Potential(atoms,
                                         slice_thickness= self.TEM_img.sampling.dz,
                                         sampling= self.TEM_img.sampling.dx)
        
        # fonction d'onde électronique qui est diffusée
        plane_wave = PlaneWave(sampling =0.01 , energy =300e3  )
        #exit_wave = plane_wave.multislice(atoms)
        exit_wave = plane_wave.multislice(self.potential)

        # exécution du calcul
        exit_wave.compute();

        # Dans les expériences HRTEM réalistes, les fonctions d'onde doivent être amplifiées par une
        # lentille d'objectif, ce qui introduit des aberrations et élimine de fait les grands angles
        # de diffusion.
        # ici on applique un flou de 50 angstreom et une ouverture d'objectif de 20 mrad
        #exit_wave.apply_ctf(defocus=-30,
        #                    focal_spread=40,
        #                    semiangle_cutoff=20)#.intensity()#.show(cbar=True);
        #image=exit_wave.intensity()#.show(common_color_scale=True, cbar=True);

        
        ctf = CTF(defocus =200 , focal_spread =40, semiangle_cutoff=20 )
        image_wave = ctf.apply(exit_wave) 
        image = image_wave.intensity()

        #image.save(self.TEM_img.name)
        #print(f"### abtem : type={type(exit_wave)}") # taille réelle en angstroems
        #print(f"### abtem : attr={dir(exit_wave)}") # taille réelle en angstroems
        logger.info(f"### abtem : extent={exit_wave.extent}") # taille réelle en angstroems
        logger.info(f"### abtem : gpts={exit_wave.gpts}")     # resolution en pixel
        logger.info(f"### abtem : sampling={exit_wave.sampling}") # taille d'un pixel en angstroem
        logger.info(f"### abtem : exit wave shape={exit_wave.shape}")  # dimension de la matrice
        logger.info(f"### abtem : potential shape={self.potential.shape}")  # dimension de la matrice
        logger.info(f"### abtem : potential extent={self.potential.extent}")  # dimension de la matrice
        
        #print(f"### abtem : type={type(image)}")  # 
        #print(f"### abtem : type={type(image.array)}")  #
        #print(f"### abtem : metadata={exit_wave.metadata}")  #
        #print(f"### abtem : metadata={image_wave.metadata}")  #
        #plt.imshow(image.array, cmap='gray')
        #plt.axis('off')
        #plt.savefig(self.TEM_img.name, dpi=300, bbox_inches="tight")

        fig, axes = plt.subplots(1,1, figsize=(10,10), gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
        a2=image.show(ax=axes)
        # Affiche dans une figure matplotlib
        #vis = measurement.show()
        plt.axis("off")
        # Sauvegarde en PNG (ou autre format suivant l’extension)
        #plt.savefig(self.TEM_img.name, dpi=300, bbox_inches="tight")

        filename = os.path.join(output_dir, f"img_{int(self.WD_lineedit_configidx.text()):04d}.png")
        plt.savefig(filename,
                    dpi=150,
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0.1,
                    facecolor='white')
        logger.info(f"{filename}")

        array = image.array
        array1 = (array * 255).astype(np.uint8)
        array_normalized = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

        self.tem_current_simulation = {
            'pil_image': PILImage.fromarray(array1, mode='L'),
            'pil_image_normalized': PILImage.fromarray(array_normalized, mode='L'),
            'array': array,
            'timestamp': time.time()
        }

        
        #plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)



        

        # 1) Le host est le QWidget posé dans QtDesigner
        host_2 = self.WD_TEMview_2  # NE PAS ÉCRASER cette variable avec le canvas !
        clear_layout(host_2)
        # 3) Créer canvas + toolbar
        canvas = FigureCanvas(a2.get_figure())
        toolbar = CustomNavigationToolbar(canvas, self)
        # 4) Récupérer ou créer le layout du host
        layout_2 = host_2.layout()
        if layout_2 is None:
            layout_2 = QtWidgets.QVBoxLayout(host_2)
            layout_2.setContentsMargins(0, 0, 0, 0)
            host_2.setLayout(layout_2)
        # 5) Insérer dans la hiérarchie Qt
        layout_2.addWidget(toolbar)
        layout_2.addWidget(canvas)
        # 6) Garder des références pour éviter le GC et pour redessiner plus tard
        #self._tem_canvas = canvas
        ##self._tem_figure = fig
        #self._tem_axes = ax
        canvas.draw()
        
        #plt.show()

        
    def _jobid_from_view_index(self, index):
        # 1) On part du modèle "vu" par la table
        model_in_view = self.tableViewListFile.model()

        # 2) Si c'est un proxy (tri/filtre), on remappe vers le modèle source
        if hasattr(model_in_view, "mapToSource"):
            source_index = model_in_view.mapToSource(index)
            source_model = model_in_view.sourceModel()
        else:
            source_index = index
            source_model = model_in_view

        row = source_index.row()

        # 3) Trouver la colonne "JobID" via l'en-tête du modèle source
        jobid_col = None
        for c in range(source_model.columnCount()):
            header = source_model.headerData(c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            if str(header) == "name":
                jobid_col = c
                break

        if jobid_col is None:
            return None  # pas trouvé

        # 4) Lire la valeur "JobID" directement dans le modèle source
        jobid_idx = source_model.index(row, jobid_col)
        return source_model.data(jobid_idx, Qt.ItemDataRole.DisplayRole)
    def on_table_double_clicked(self, index):
        # Remplace TON contenu actuel qui fait self.df.iloc[row]["JobID"] par ceci :
        jobid = self._jobid_from_view_index(index)
        if jobid is None:
            QMessageBox.warning(self, "Info", "Colonne 'JobID' introuvable.")
            return
        #QMessageBox.information(self, "Job", f"JobID : {jobid}")
        print(f"JobID : {jobid}")
        self.WD_lineedit_rmt_xyzfile.setText(jobid)
    def update_plot(self,step,Etot):
        #self.y = np.roll(self.y, -1)
        #self.y[-1] = np.sin(0.1*self.t)
        self.curve.setData(step, Etot)
        #self.t += 1
    def _on_opt_step(self, mol, istep:int,step,Etot):
        #print(Etot)
        self.update_plot(step,Etot)
        # mol est self.molecule ; on pousse l’état courant dans le viewer
        self.NP_viewer.set_molecule(self.molecule)
        self.atom_model.setDataFrame(self.molecule.to_df())
        self.tableView_AtomList.resizeColumnsToContents()
        self.tableView_AtomList.setSortingEnabled(True)   # maintenant OK

        # Laisse Qt traiter la file d'événements pour rafraîchir l'affichage
        QApplication.processEvents()
    def optimize_NP(self,tol=Config.TOLERANCE):
        print("---------- optimize_NP ----------")
        self.molecule.FF=ForceField()
        start = time.perf_counter()

        self.molecule.optimize(new_step=self._on_opt_step,tol=tol)
        end = time.perf_counter()
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Optimization finished")
        msg.setText("Your computation has completed.")
        msg.setInformativeText("Click OK to close this message.")
        msg.setDetailedText(f"Runtime: {end - start:.3f} s\nResult: Converged ✅")
        msg.exec()
    def Update_rmt_list(self):
        print(self.WD_lineedit_rmt_directory.text())
        try:
            # 4) Connexion SSH
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Avec mot de passe :
            #ssh.connect(hostname, username=username, password=password, timeout=10)
            # Ou avec clé :
            ssh.connect(self.hostname, username=self.username, key_filename=self.key_filename, timeout=10)
            # Exécution d'une commande
            #stdin, stdout, stderr = ssh.exec_command("/home2020/home/ipcms/bulou/bin/etat_jobs.sh --nday 4")
            cmd = f"cd {self.WD_lineedit_rmt_directory.text()} ; ls -rlt *.{self.WD_lineedit_rmt_extension.text()} --time-style='+%Y-%m-%d %H:%M:%S '"
            stdin, stdout, stderr = ssh.exec_command(cmd)


            
            #stdin, stdout, stderr = ssh.exec_command("cd "+self.WD_lineedit_rmt_directory.text()+" ; ls -rlt *.xyz")
            rows=[]
            for line in stdout.readlines():
                lsplit=line.split()
                dt = pd.to_datetime(f"{lsplit[5]} {lsplit[6]}", errors='coerce', format="%Y-%m-%d %H:%M:%S")

                rows.append({
                    "date": lsplit[5],
                    "time": lsplit[6],
                    "name": lsplit[7],
                    "datetime":dt,
                })

            self.df_filelist = pd.DataFrame(rows)
            print(self.df_filelist)
            #self.filelist_model.layoutChanged.emit()  # notifie la vue

            #self.df_filelist = pd.DataFrame(columns=["date","time", "name","datetime"])
            self.filelist_model.setDataFrame(self.df_filelist)
            self.tableViewListFile.resizeColumnsToContents()
            self.tableViewListFile.setSortingEnabled(True)   # maintenant OK


            ssh.close()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
        
    def load_NP_rmt(self):
        print(self.WD_lineedit_rmt_directory.text())
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Avec mot de passe :
            #ssh.connect(hostname, username=username, password=password, timeout=10)
            # Ou avec clé :
            ssh.connect(self.hostname, username=self.username, key_filename=self.key_filename, timeout=10)

            remote_path=self.WD_lineedit_rmt_directory.text()+"/"+self.WD_lineedit_rmt_xyzfile.text()
            sftp = ssh.open_sftp()

            # 5) Obtenir la taille pour la progression
            try:
                total_size = sftp.stat(remote_path).st_size
            except IOError:
                total_size = 0  # si inconnu, on mettra une progression indéterminée

            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)     
            bytes_so_far = {"n": 0}

            def cb(sent, total):
            #    # Paramiko envoie "sent" cumulé; "total" peut être 0 si inconnu
                if total == 0:
            #        # fallback: estimer via stat si disponible
                    t = total_size or 1
                else:
                    t = total
                pct = int((sent / t) * 100)
                self.progress.setValue(min(max(pct, 0), 100))

            # 6) Téléchargement
            local_path='./tmp.xyz'
            sftp.get(remote_path, local_path, callback=cb if total_size else None)

            # Fermeture de la connexion
            sftp.close()

            self.progress.setValue(100)
            QMessageBox.information(self, "Succès",
                                    f"{remote_path} loaded as :\n{local_path}")
            try:
                self.molecule
            except NameError:
                print("La variable n'existe pas")
            else:
                print("La variable existe")
            
            self.molecule.load_file(local_path)
            self.atom_model.setDataFrame(self.molecule.to_df())
            self.tableView_AtomList.resizeColumnsToContents()
            self.tableView_AtomList.setSortingEnabled(True)   # maintenant OK

            self.update_NP_info()

            



        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))


    def load_NP(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier",
            "",  # répertoire de départ ("" = courant)
            "Fichiers xyz (*.xyz);;Tous les fichiers (*)"
        )

        print(f"loading {filename}")
        if not filename:
            return  # Sortir si vide  
        self.molecule=Crystal()
        self.molecule.load_file(filename)
        self.molecule.MassCenter()
        self.molecule.get_element_distribution()
        self.molecule.get_structure()
        self.atom_model.setDataFrame(self.molecule.to_df())
        self.tableView_AtomList.resizeColumnsToContents()
        self.tableView_AtomList.setSortingEnabled(True)   # maintenant OK
        
        self.update_NP_info()
        
    def load_NP_old(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier",
            "",  # répertoire de départ ("" = courant)
            "Fichiers xyz (*.xyz);;Tous les fichiers (*)"
        )
        if filename:
            print("Fichier choisi:", filename)
            # ici tu charges ton fichier


            try:
                self.molecule
            except NameError:
                print("laod_NP - La variable n'existe pas")
            else:
                print("La variable existe")
            
            self.molecule.load_file(filename)
            self.atom_model.setDataFrame(self.molecule.to_df())
            self.tableView_AtomList.resizeColumnsToContents()
            self.tableView_AtomList.setSortingEnabled(True)   # maintenant OK

            self.update_NP_info()
            #del self.molecule

    def save_NP(self):
        self.molecule.save(prefix="NP")
        QMessageBox.information(self, "Succès",f"Saved as NP.xyz")


    def get_composition(self):
        """
        détermine la composition souhaitée de la nanoparticule à partir de la table
        des éléments.
        """
        print("get_composition")
        self.composition=[]
        for elt in self.elt:
            txt = elt.text()           # récupérer la chaîne associée
            is_in_list = txt in self.composition
            print(txt,elt.isChecked(), is_in_list)
            if elt.isChecked():
                self.composition.append(elt.text())
                
        print(self.composition)
        if len(self.composition)>0:
            stoechiometry=1.0/len(self.composition)
            for elt in self.elt:
                lineedit = getattr(self,f"lineEdit_{elt.text()}")
                if elt.isChecked():
                    lineedit.setText(f"{stoechiometry:4.2f}")   # modifie le bon widget
                else:
                    lineedit.setText(f"{0.0:4.2f}")   # modifie le bon widget

    def new_NP(self):
        print(100*"#","\nNew_NP")
        if self.WD_lineedit_radius.text().strip() and self.WD_lineedit_seed.text().strip():
            N=int(float(self.WD_lineedit_boxsize.text())/0.392)  # 0.392 est le parametre de maille du Pt en nm
            # test si un élément a déjà été défini
            if len(self.composition)<1:
                self.composition=['Pt']
                self.elt_Pt.setChecked(True)

            # récupère la composition souhaitée
            self.get_composition()

            # génération du bulk
            Bulk=Crystal()
            Bulk.build(Nx=N,Ny=N,Nz=N,elt=self.composition[0])
            # découpe de la nanoparticule
            Bulk.origin_at_mass_center()
            self.molecule=Bulk.transform(radius=10*float(self.WD_lineedit_radius.text()))
            
            self.molecule.get_element_distribution()
            for elt in self.composition:
                if elt not in self.molecule.pos_elt:
                    self.molecule.pos_elt[elt]=[]
                print(f"### {elt} {self.molecule.pos_elt[elt]} -> stoechiometry {len(self.molecule.pos_elt[elt])/len(self.molecule.atoms)}")
            
            #self.molecule.get_element_distribution()
            self.molecule.get_structure()

            #exit()
            #self.molecule = HEAS(radius=10*float(self.WD_lineedit_radius.text()),
            #                     seed=int(self.WD_lineedit_seed.text()),
            #                     N=N)

            print(100*"-","\n### Building alloy")
            print("### element(s) :",self.composition)
            print("### ",self.molecule.pos_elt)
            stoechiometry=1.0/len(self.composition)
            nmin=len(self.molecule.pos_elt[self.composition[0]])*stoechiometry
            seed=int(self.WD_lineedit_seed.text())
            random.seed(seed)
            idxfill=1
            while len(self.molecule.pos_elt[self.composition[0]])>nmin:
                # on choisit au hasard un des atomes de l'espèce en excés
                n = random.randrange(0, len(self.molecule.pos_elt[self.composition[0]]))   # 0 à 10 (11 exclu)
                if len(self.molecule.pos_elt[self.composition[idxfill]])>=nmin:
                    idxfill=idxfill+1
                idx=self.molecule.pos_elt[self.composition[0]].pop(n)
                self.molecule.pos_elt[self.composition[idxfill]].append(idx)
                self.molecule.atoms[idx].elt=self.composition[idxfill]
                print(n,idx,"->",
                      self.molecule.pos_elt[self.composition[0]],
                      "# ",
                      self.composition[idxfill],
                      self.molecule.pos_elt[self.composition[idxfill]])

            self.molecule.get_element_distribution()
            self.molecule.get_structure()

            self.atom_model.setDataFrame(self.molecule.to_df())
            self.tableView_AtomList.resizeColumnsToContents()
            self.tableView_AtomList.setSortingEnabled(True)   # maintenant OK

            self.update_NP_info()
    def update_NP_info(self):
        if hasattr(self.molecule, "filenfo"):
            print("L'attribut existe !")
            self.WD_label_filenfo.setText(f"""
            <b>Name :</b> {self.molecule.filenfo.name}<br>
            <b>Size :</b> {self.molecule.filenfo.size} Ko<br>
            <b>Type :</b> {self.molecule.filenfo.type_mime}<br>
            <b>number of structures:</b> {self.molecule.filenfo.nstruct}<br>
            <b>Modified :</b> {self.molecule.filenfo.modified.isoformat()}
            """)
            self.WD_label_filenfo.setTextFormat(Qt.TextFormat.RichText)

        else:
            print("Pas d'attribut de ce nom.")
        
        #self.moleculeChanged.emit(self.molecule)
        self.NP_viewer.set_molecule(self.molecule)
        self.atom_model.setDataFrame(self.molecule.to_df())
        self.tableView_AtomList.resizeColumnsToContents()
        self.tableView_AtomList.setSortingEnabled(True)   # maintenant OK

        natom=len(self.molecule.atoms)
        self.WD_label_natom.setText("Number of atoms: "+str(natom))
        #self.WD_label_natom.setText("Number of atoms: "+str(len(self.molecule.list_elt)))
        self.WD_table_list_elt.setRowCount(0)
        for elt,count in self.molecule.element_counts.items():
            r = self.WD_table_list_elt.rowCount()
            self.WD_table_list_elt.insertRow(r)
            self.WD_table_list_elt.setItem(r, 0, QTableWidgetItem(elt))
            self.WD_table_list_elt.setItem(r, 1, QTableWidgetItem(str(count)))
            self.WD_table_list_elt.setItem(r, 2, QTableWidgetItem("%6.2f"%(100.0*count/natom)))
        #print(elt,count)
    # ========================================================================================
    # UTILITAIRES
    # ========================================================================================
    
    def _show_error(self, title: str, message: str):
        """
        Affiche une boîte de dialogue d'erreur.
        
        Args:
            title: Titre de la boîte de dialogue
            message: Message d'erreur
        """
        QMessageBox.critical(self, title, message)
        logger.error(f"{title}: {message}")
        
# ##########################################################################################
# Point d’entrée du programme
if __name__ == "__main__":
    
    print(abtem.__file__)
    
    # initialise l’application Qt.
    app = QApplication(sys.argv)
    # crée une instance (l'objet) de la fenêtre principale.
    window = MainApp() # Instancie la fenêtre principale (`MainApp`)

    #window.showMaximized()
    #window.resize(1831,586) # Définit sa **taille : 1200 x 1000 pixels**
    #window.resize(1800,900) # Définit sa **taille : 1200 x 1000 pixels**



    # Positionnement sur l'écran
    screens = QGuiApplication.screens()
    screen_idx = Config.DEFAULT_SCREEN_INDEX
    
    # Sélection de l'écran
    if 0 <= screen_idx < len(screens):
        screen = screens[screen_idx]
    else:
        screen = app.primaryScreen()
        logger.warning(f"Écran {screen_idx} non disponible, utilisation de l'écran primaire")
    
    # Récupération de la géométrie
    screen_geometry = screen.availableGeometry()
    
    # Positionnement en haut à droite
    x = screen_geometry.right() - window.width()
    y = screen_geometry.top()
    window.move(x, y)

    
    # === Choix de l'écran ===
    # screens = app.screens()
    # print(len(screens))

    # if len(screens) > 1:
    #     target_screen = screens[1]  # écran secondaire
    # else:
    #     target_screen = screens[0]  # seul écran disponible

    # # === Positionnement de la fenêtre sur l'écran ciblé ===
    # target_screen = screens[0]
    # geometry = target_screen.geometry()

    # window.move(geometry.topLeft()) #  L’affiche (`show()`)

    # affiche la fenêtre window à l’écran.
    window.show()
    frame = window.frameGeometry()
    print("Frame size :", frame.width(), "x", frame.height())


    # app.exec() lance la boucle événementielle (qui attend les interactions utilisateur : clics, saisies, etc.).
    # Démarre la boucle événementielle Qt (`exec()`) ;
    # Lance la boucle événementielle (attente d’interactions)   
    sys.exit(app.exec()) 
