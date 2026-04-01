from HBPy.Molecule.Atom import Z_from_elt  # Z_from_elt est une méthode de la classe Atom définie dans le fichier Atom.py se trouvant dans le répertoire Molecule
import HBPy.Molecule.Crystal as Crystal
import numpy as np
#from pathlib import Path
import subprocess
import shutil


class FEFF:
    configall={'RPATH': 5.0}
    def __init__(self):
        self.config={
            'TITLE':'FEFF INPUT FILE',
            'DEBYE_TEMP': 190.0,
            'SCF_RADIUS': 5.0,
            'RPATH': 5.0,     # typique 2.2xdistance plus proches voisins. changer pour étudier la cvg des spectres
            'EXAFS' : 20.0,   # xkmax - default 20 ang.^-1
            'EDGE':  "K",
            'RMAX':8.0,
            'feff_dir':  './JFEFF/feff90/unix/'
        }
    def create_input_file(self,
                          molecule:Crystal,
                          absorber_idx:int,
                          T:float=300.0, 
                          filename='feff.inp',
                          rpath=configall['RPATH']):


        
        # Positionner l'origine sur l'atome absorbeur
        tmp_molecule=molecule.duplicate()
        tmp_molecule.origin_at(origin=molecule.atoms[absorber_idx].q)
        absorber = tmp_molecule.atoms[absorber_idx]


        with open(filename, "w") as f:
            f.write(f"TITLE {self.config['TITLE']}\n")
            #     # *Cu at 190K, Debye temp 315K (Ashcroft & Mermin)
            #     # DEBYE 190 315 0
            f.write(f"DEBYE {T} {self.config['DEBYE_TEMP']} 0\n")
            f.write(f"SCF {self.config['SCF_RADIUS']}\n")
            f.write(f'EXAFS {self.config["EXAFS"]}\n')
            f.write(f"RPATH {rpath}\n")
            f.write(f"EDGE {self.config['EDGE']}\n")
            f.write(f'CONTROL\t1 1 1 1 1 1\n')            
            
            
            list_atm={}
            #     # Autres atomes
            for atm in tmp_molecule.atoms:
                if atm.idx != absorber.idx:
                    # Calculer la distance
                    R = atm.q - absorber.q
                    d = np.linalg.norm(R)
                    if d<self.config["RMAX"]: list_atm[atm.elt]=atm.idx


            # Section POTENTIALS
            f.write(f'\nPOTENTIALS\n')
            f.write(f' {0:>4d} {Z_from_elt[absorber.elt]:>5d} {absorber.elt:>7s}\n')
            #     # Liste des éléments uniques 
            for i, elt in enumerate(molecule.list_elt, start=1):
                if elt in list_atm.keys():
                    f.write(f' {i:>4d} {Z_from_elt[elt]:>5d} {elt:>7s}\n')
            # Section ATOMS
            f.write(f'\nATOMS\n')
            f.write(
                f' {absorber.q[0]:>10.6f} {absorber.q[1]:>10.6f} {absorber.q[2]:>10.6f} '
                f'{0:>4d} {absorber.elt:>5s} {0:>8.4f} (Absorbeur)\n'
                 )

            #     # Autres atomes
            for atm in tmp_molecule.atoms:
                if atm.idx != absorber.idx:
                    # Calculer la distance
                    R = atm.q - absorber.q
                    d = np.linalg.norm(R)
                    if d>self.config["RMAX"]: continue 
                    # Trouver l'indice du potentiel
                    ipot = molecule.list_elt.index(atm.elt) + 1
                    f.write(
                        f' {atm.q[0]:>10.6f} {atm.q[1]:>10.6f} {atm.q[2]:>10.6f} '
                        f'{ipot:>4d} {atm.elt:>5s} {d:>8.4f}\n'
                    )

            f.write(f'END\n')

        del tmp_molecule

    def run(self,filename,config=configall,list_pgm=['rdinp','pot']):
        shutil.copy2(filename,"feff.inp")
        for pgm in list_pgm:
            print(100*"#") ; print(f"### {pgm}") ; print(100*"#")
            subprocess.run([self.config["feff_dir"]+"/"+pgm],
                           capture_output=False, 
                           text=True, 
                           check=True)
        
    

####################################################################################################################    
#    def __init__(self):
#        """Constantes de configuration de l'application."""



# class FEFF:
#     def __init__(self):
#         self.FEFF_PGM = [
#             "rdinp", "atomic", "dmdw", "pot", "opconsat", 
#             "screen", "xsph", "fms", "mkgtr", "path", 
#             "genfmt", "ff2x", "sfconv", "compton", "eels", "ldos"
#         ]
#         self.config=self.default_config()

#     def default_config(self):
#         return {
#             'debye_temp_0':190.0,
#             'debye_temp'  :315.0,
#             'edge':  "K",
#             'scf_radius': 5.0,
#             'rpath': 5.0,     # typique 2.2xdistance plus proches voisins. changer pour étudier la cvg des spectres
#             'EXAFS' : 20.0,   # xkmax - default 20 ang.^-1
#             'feff_dir' : '/home/bulou/ownCloud/Notebooks/M2P2_HEA/Home/Modelisation/ATOMOD/JFEFF/feff90/unix/'
#             }
        
#     def create_input_file(self,
#                           molecule:Crystal,
#                           absorber_idx: int=0,
#                           filename:str="feff.inp",
#                           rmax:float=8.0,
#                           title: str = "FEFF Calculation",
#                           rpath=5.0,
#                           config=None):

#         if config is None:
#             #config=self.default_config()
#             config=self.config
        
#         # Positionner l'origine sur l'atome absorbeur
#         tmp_molecule=molecule.duplicate()
#         tmp_molecule.origin_at(origin=molecule.atoms[absorber_idx].q)
#         absorber = tmp_molecule.atoms[absorber_idx]


#         with open(filename, "w") as f:
#             # En-tête
#             f.write(f'TITLE {title} - Atome absorbeur : {absorber.elt}\n')
#             # *Cu at 190K, Debye temp 315K (Ashcroft & Mermin)
#             # DEBYE 190 315 0
#             f.write(f'DEBYE {config["debye_temp_0"]} {config["debye_temp"]} 0\n')
#             f.write(f'SCF {config["scf_radius"]}\n')

#             f.write(f'EXAFS {config["EXAFS"]}\n')
#             f.write(f'RPATH {rpath}\n')
#             f.write(f'EDGE {config["edge"]}\n')



#             f.write(f'CONTROL\t1 1 1 1 1 1\n')
#             # Section POTENTIALS
#             f.write(f'\nPOTENTIALS\n')
#             f.write(f' {0:>4d} {Z_from_elt[absorber.elt]:>5d} {absorber.elt:>7s}\n')
#             # Liste des éléments uniques 
#             for i, elt in enumerate(molecule.list_elt, start=1):
#                 f.write(f' {i:>4d} {Z_from_elt[elt]:>5d} {elt:>7s}\n')
#             # Section ATOMS
#             f.write(f'\nATOMS\n')
#             f.write(
#                 f' {absorber.q[0]:>10.6f} {absorber.q[1]:>10.6f} {absorber.q[2]:>10.6f} '
#                 f'{0:>4d} {absorber.elt:>5s} {0:>8.4f} (Absorbeur)\n'
#             )

#             # Autres atomes
#             for atm in tmp_molecule.atoms:
#                 if atm.idx != absorber.idx:
#                     # Calculer la distance
#                     R = atm.q - absorber.q
#                     d = np.linalg.norm(R)
#                     if d>rmax: continue
#                     # Trouver l'indice du potentiel
#                     ipot = molecule.list_elt.index(atm.elt) + 1


#                     f.write(
#                         f' {atm.q[0]:>10.6f} {atm.q[1]:>10.6f} {atm.q[2]:>10.6f} '
#                         f'{ipot:>4d} {atm.elt:>5s} {d:>8.4f}\n'
#                     )

#             f.write(f'END\n')
#             del tmp_molecule
#     def run(self,config,pgm='rdinp'):
#         shutil.copy2(filename,"feff.inp")
#         result=subprocess.run([self.config["feff_dir"]+"/"+pgm],
#                               capture_output=True, 
#                               text=True, 
#                               check=True)
#         print(result)
        
#     def run_old(self,feff_pgm):
#         # Liste ordonnée des programmes FEFF à exécuter
#         #conf={"rdinp":True,"atomic":False}
#         for pgm in feff_pgm.keys():
#             try:
#                 print(f"{pgm} -> {feff_pgm[pgm].isChecked()}")
#                 if feff_pgm[pgm].isChecked():
#                     print(f"### FEFF ### {pgm} is running...")
#                     result=subprocess.run([self.config["feff_dir"]+"/"+pgm],
#                                           capture_output=True, 
#                                           text=True, 
#                                           check=True)
#                     print(f"{result.stdout.strip()}")
#                     print(f"... -> DONE!")
#                     print(100*"-")
#             except subprocess.CalledProcessError as e:
#                 # e.stderr contient le message d'erreur officiel du programme
#                 print(f"La commande {pgm} a échoué avec l'erreur : {e.stderr}")
#         print(f"EXAFS calculation DONE!")
#     # def _on_feff_pgm_checkbox_changed(self):
#     #     pass

##############################################################################
        
# class FEFF_config:
#     def __init__(self):
#         self.title="title"
#         self.debye_temp_0=190.0
#         self.debye_temp  =315.0
#         self.edge: str = "K"
#         self.scf_radius: float = 5.0
#         self.rpath: float = 5.0
#         self.feff_dir = '/home/bulou/ownCloud/Notebooks/M2P2_HEA/Modelisation/ATOMOD/JFEFF/feff90/unix/'

# def FEFF_info(idx=0):
#     print(f"FEFF")
#     print(f"idx={idx}")

# def FEFF_create_parameter_file(
#         filename:str,
#         molecule: Crystal,
#         absorber_idx: int = 0,
#         config= None,
#         rmax:float=8.0,
#         title: str = "FEFF Calculation") -> None:

    
#     if config is None:
#         config = FEFF_config()

#     # Positionner l'origine sur l'atome absorbeur
#     tmp_molecule=molecule.duplicate()
#     tmp_molecule.origin_at(origin=molecule.atoms[absorber_idx].q)
#     absorber = tmp_molecule.atoms[absorber_idx]


#     for atm in molecule.atoms:
#         logger.info(f"###     MOLECULE ### {atm.idx} {atm.elt} {atm.q}")
#     for atm in tmpmolecule.atoms:
#         logger.info(f"### TMP_MOLECULE ### {atm.idx} {atm.elt} {atm.q}")
    
#     with open(filename, "w") as f:
#         # En-tête
#         f.write(f'TITLE {title} - Atome absorbeur : {absorber.elt}\n')
#         f.write(f'DEBYE {config.debye_temp_0} {config.debye_temp} 0\n')
#         f.write(f'EDGE {config.edge}\n')
#         f.write(f'SCF {config.scf_radius}\n')
#         f.write(f'RPATH {config.rpath}\n')
#         f.write(f'CONTROL\t1 1 1 1 1 1\n')
        
#         # Section POTENTIALS
#         f.write(f'\nPOTENTIALS\n')
#         f.write(f' {0:>4d} {Z_from_elt[absorber.elt]:>5d} {absorber.elt:>7s}\n')
#         # Liste des éléments uniques 
#         for i, elt in enumerate(molecule.list_elt, start=1):
#             f.write(f' {i:>4d} {Z_from_elt[elt]:>5d} {elt:>7s}\n')
#         # Section ATOMS
#         f.write(f'\nATOMS\n')
#         f.write(
#             f' {absorber.q[0]:>10.6f} {absorber.q[1]:>10.6f} {absorber.q[2]:>10.6f} '
#             f'{0:>4d} {absorber.elt:>5s} {0:>8.4f} (Absorbeur)\n'
#         )
            
#         # Autres atomes
#         for atm in tmp_molecule.atoms:
#             if atm.idx != absorber.idx:
#                 # Calculer la distance
#                 R = atm.q - absorber.q
#                 d = np.linalg.norm(R)
#                 if d>rmax: continue

#                 # Trouver l'indice du potentiel
#                 ipot = molecule.list_elt.index(atm.elt) + 1
#                 f.write(
#                     f' {atm.q[0]:>10.6f} {atm.q[1]:>10.6f} {atm.q[2]:>10.6f} '
#                     f'{ipot:>4d} {atm.elt:>5s} {d:>8.4f}\n'
#                 )
            
#         f.write(f'END\n')
#         del tmp_molecule
        
# class FEFF_calculator:
#     """Classe pour gérer les calculs FEFF"""
    
#     # Liste ordonnée des programmes FEFF à exécuter
#     FEFF_PROGRAMS = [
#         "rdinp", "atomic", "dmdw", "pot", "opconsat", 
#         "screen", "xsph", "fms", "mkgtr", "path", 
#         "genfmt", "ff2x", "sfconv", "compton", "eels", "ldos"
#     ]
#     def __init__(self, config: FEFF_config):
#         self.config = config
#         self._validate_feff_installation()
    
#     def _validate_feff_installation(self):
#         """Vérifie que FEFF est correctement installé"""
#         feff_path = Path(self.config.feff_dir)
#         if not feff_path.exists():
#             raise FileNotFoundError(
#                 f"Le répertoire FEFF n'existe pas: {self.config.feff_dir}\n"
#                 f"Définissez la variable d'environnement FEFF_DIR ou modifiez le script."
#             )
        
#         # Vérifier que rdinp existe au moins
#         rdinp_path = feff_path / "rdinp"
#         if not rdinp_path.exists():
#             raise FileNotFoundError(
#                 f"Programme rdinp introuvable dans {self.config.feff_dir}"
#             )

