from PyQt6.QtWidgets import (QVBoxLayout,QLabel,QLineEdit,QGridLayout,QToolButton,
                             QPushButton,QHBoxLayout,QDialog)

class PeriodicTableDialog(QDialog):
    """
    Petit tableau périodique cliquable.
    - Clic sur un élément => (dé)sélection (bouton à bascule).
    - Barre de recherche pour filtrer par symbole (H, He, Fe, ...)
    - Boutons 'Tout effacer' et 'Valider'
    Utilisation:
        dlg = PeriodicTableDialog(self, preselected=['Pt','Pd'])
        if dlg.exec():
            selection = dlg.selected_elements()
    """
    PERIODS = [
        # Les cases vides "" créent des décalages pour aligner le tableau
        ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
        ["Li","Be","","","","","","","","","","","B","C","N","O","F","Ne"],
        ["Na","Mg","","","","","","","","","","","Al","Si","P","S","Cl","Ar"],
        ["K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr"],
        ["Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe"],
        ["Cs","Ba","La","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"],
        ["Fr","Ra","Ac","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"],
        # Lant/Act séparés (rangées « 4f » et « 5f »)
        ["", "", "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",""],   # Lanthanides
        ["", "", "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",""],   # Actinides
    ]

    def __init__(self, parent=None, preselected=None):
        super().__init__(parent)
        self.setWindowTitle("Tableau périodique – sélection d’éléments")
        self.setModal(True)
        self._buttons = {}  # symbole -> QToolButton

        main = QVBoxLayout(self)

        # Barre de recherche
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Filtrer (symbole) :"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Ex: Pt, C, He…")
        self.search_edit.textChanged.connect(self._apply_filter)
        search_row.addWidget(self.search_edit)
        main.addLayout(search_row)

        # Grille du tableau
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        main.addLayout(grid)

        for r, period in enumerate(self.PERIODS):
            for c, sym in enumerate(period):
                if not sym:
                    continue
                btn = QToolButton()
                btn.setText(sym)
                btn.setCheckable(True)
                btn.setMinimumWidth(36)
                btn.setMinimumHeight(28)
                btn.setToolTip(sym)
                grid.addWidget(btn, r, c)
                self._buttons[sym] = btn

        # Préselection éventuelle
        preselected = preselected or []
        for s in preselected:
            if s in self._buttons:
                self._buttons[s].setChecked(True)

        # Ligne d’actions
        actions = QHBoxLayout()
        self.clear_btn = QPushButton("Tout effacer")
        self.clear_btn.clicked.connect(self._clear_all)
        self.ok_btn = QPushButton("Valider")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.clicked.connect(self.reject)
        actions.addWidget(self.clear_btn)
        actions.addStretch(1)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.ok_btn)
        main.addLayout(actions)

        self.resize(700, 360)

    def _apply_filter(self, text: str):
        txt = text.strip().lower()
        for sym, btn in self._buttons.items():
            btn.setVisible((txt in sym.lower()) if txt else True)

    def _clear_all(self):
        for btn in self._buttons.values():
            btn.setChecked(False)

    def selected_elements(self):
        return [sym for sym, btn in self._buttons.items() if btn.isChecked()]
