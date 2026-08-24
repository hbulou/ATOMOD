import sys
import os
from datetime import datetime, timezone
import random
import numpy
from collections import defaultdict
import copy
import HBPy
from HBPy.Molecule.Crystal import Crystal,Atom
import pandas as pd
import json
from scipy.constants import physical_constants
kB_eV = physical_constants['Boltzmann constant in eV/K'][0]

import matplotlib.pyplot as plt
sys.path.append('./lib/')
import abtem
from pathlib import Path
import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ##################################################################################
def alloy_stability(config,T=300):
################################################################################
    """
    fonction alloy_stability()
    T : Temperature en Kelvin
    L'idée de la fonction alloy_stability est de procéder à des échanges d'atomes, de
    nature chimique différente, entre deux (pour l'instant) nanoparticules.
    POur l'instant la structure des deux particules initiales est déterminées au
    hasard (seed). (fonction init_configuration())
    [A FAIRE] lire les configurations initiales à partir d'un fichier xyz
    [A FAIRE] changer le nombre de particules 
    """
    # ----------------- Fonctions locales ------------------------
        
    def init_configurations(config,nparticle):
        for seed in range(config['Alloy stability']['MC']['seed'],config['Alloy stability']['MC']['seed']+nparticle):
            NP.append(Crystal())
            NP[-1].build(a=config['NP']['structure']['a'],
                         radius=config['NP']['structure']['radius'],
                         materials='NP')
            NP[-1].origin_at_mass_center()
            logger.info(f"Number of atoms={len(NP[-1].atoms)}")
            NP[-1].set_composition(config['NP']['structure']['composition'],seed=seed)
            NP[-1].optimize_ase(config)
            compute_EXAFS_spectrum(config,NP[-1],feff_dir=config['run_dir']/config['simul_dir']/"feff_input_files")
            NP[-1].get_element_distribution()
        NP.append(copy.deepcopy(NP[0]))
        NP.append(copy.deepcopy(NP[1]))
        return NP


    def save_configuration(NP,last_NP_idx,records,savedirxyz,f):
        """
        fonction 
          * sauvant la nanoparticule NPau format xyz dans le directory savedirxyz
          * 
        
        """

        NP.save(prefix=f"NP{last_NP_idx:08d}",fmt='xyz',savedir=savedirxyz);
        NP.chemical_formula=""
        for elt in sorted(NP.list_elt):
            NP.chemical_formula=f"{NP.chemical_formula}{elt}{len(NP.pos_elt[elt])}"
        f.write(f"{NP.chemical_formula} {NP.Epot:12.6f}\n")
        logger.info(f"NP {last_NP_idx:4d} - {NP.chemical_formula} - Epot={NP.Epot:8.3f} - lst_elt={NP.list_elt}")
        records=update_records(last_NP_idx,NP,records)
        
        last_NP_idx+=1
        return last_NP_idx,records
    
    def test_configuration(diff,T,i0,i1,i2,i3):
        exch=False
        if diff<0:
            itmp=i0
            i0=i2
            i2=itmp
            itmp=i1
            i1=i3
            i3=itmp
            exch=True
        else:
            if random.random()<1/(1+numpy.exp(diff/(kB_eV*T))):
                itmp=i0
                i0=i2
                i2=itmp
                itmp=i1
                i1=i3
                i3=itmp
                exch=True
        return i0,i1,i2,i3,exch


    def try_configurations(NP):
        elt_switch = random.sample(NP[i0].list_elt, 2)

        if elt_switch[0] in NP[i0].pos_elt and elt_switch[1] in NP[i1].pos_elt:
            is0=0
            is1=1
        else :
            if elt_switch[1] in NP[i0].pos_elt and elt_switch[0] in NP[i1].pos_elt:
                is0=1
                is1=0
            else:
                logger.error()
                exit()

        ielt0=random.randrange(len(NP[i0].pos_elt[elt_switch[is0]]))
        ielt1=random.randrange(len(NP[i1].pos_elt[elt_switch[is1]]))
        pos0=NP[i0].pos_elt[elt_switch[is0]][ielt0]
        pos1=NP[i1].pos_elt[elt_switch[is1]][ielt1]

        NP[i2]=copy.deepcopy(NP[i0])
        NP[i2].atoms[pos0].elt=elt_switch[is1]
        NP[i2].optimize_ase(config)
        NP[i2].get_element_distribution()
        NP[i3]=copy.deepcopy(NP[i1])
        NP[i3].atoms[pos1].elt=elt_switch[is0]
        NP[i3].optimize_ase(config)
        NP[i3].get_element_distribution()
        return NP

    
    # ---------------- Partie principale d'Alloy_stability ----------------------

    # --- initialisation des variables ----
    NP=[]
    nparticle=2
    nexch=0
    i0=0
    i1=1
    i2=2
    i3=3
    savedir=config['run_dir']/config['simul_dir']
    savedir.mkdir(parents=True, exist_ok=True)
    f=open(savedir/"conf.nrj", "w", encoding="utf-8")
    fMC=open(savedir/"MC.dat", "w", encoding="utf-8")
    last_NP_idx=0
    savedirxyz=savedir/"XYZ"
    savedirxyz.mkdir(parents=True, exist_ok=True)
    records=[]

    # ------------ Configurations initiales ----------------------------
    NP=init_configurations(config,nparticle)
    # ------------ Nouvelles configurations ----------------------------
    NP=try_configurations(NP)

    for i in range(2*nparticle):
        last_NP_idx,records=save_configuration(NP[i],last_NP_idx,records,savedirxyz,f)
        
    Dnrj=NP[i3].Epot+NP[i2].Epot-NP[i1].Epot-NP[i0].Epot
    i0,i1,i2,i3,exch=test_configuration(Dnrj,T,i0,i1,i2,i3)
    if exch:
        nexch+=1
    logger.info(f"{exch} {nexch} exchange(s) Dnrj={Dnrj}")

    for istep in range(config['Alloy stability']['MC']['nstep']):
        NP=try_configurations(NP)

        last_NP_idx,records=save_configuration(NP[i2],last_NP_idx,records,savedirxyz,f)
        last_NP_idx,records=save_configuration(NP[i3],last_NP_idx,records,savedirxyz,f)

        Dnrj=NP[i3].Epot+NP[i2].Epot-NP[i1].Epot-NP[i0].Epot
        i0,i1,i2,i3,exch=test_configuration(Dnrj,T,i0,i1,i2,i3)
        if exch:
            nexch+=1
        logger.info(f"{exch} {nexch} exchange(s) Dnrj={Dnrj}")
            
        fMC.write(f"{istep:8d} {NP[i0].chemical_formula}   {NP[i1].chemical_formula} {NP[i0].Epot+NP[i1].Epot:12.6f}\n")

        
    f.close()
    fMC.close()
    df = pd.DataFrame(records)
    print(df)
    comp_df = pd.json_normalize(df['local_composition'])
    comp_df.columns = [f'frac_{c}' for c in comp_df.columns]
    df = pd.concat([df.reset_index(drop=True), comp_df], axis=1)
    print(df)


    # Sérialiser local_composition (dict) en JSON string pour compatibilité Parquet
    df_save = df.copy()
    df_save['local_composition'] = df_save['local_composition'].apply(json.dumps)
    
    df_save.to_parquet('nanoparticules_HEA_atomes.parquet', engine='pyarrow', compression='snappy')

    
# ##################################################################################
def compute_EXAFS_spectrum(config,NP,feff_dir=Path("./")):
    #feff_dir=config['run_dir']/config['simul_dir']/"feff_input_files"
    base_dir=os.getcwd()
    for atm in NP.atoms:
        config['feff']['parameters']['input_save_dir']=f"{feff_dir}/{atm.elt}_{atm.idx}"
        os.makedirs(config['feff']['parameters']['input_save_dir'], exist_ok=True)
        os.chdir(f"{config['feff']['parameters']['input_save_dir']}")
        NP.FEFF_create_input_file(config['feff']['parameters'],absorber_idx=atm.idx)
        NP.FEFF_run(config['feff']['parameters'])
        dossier = Path("./")
        for element in dossier.iterdir():
            if element.is_file() and element.name != "xmu.dat":
                element.unlink()
        os.chdir(base_dir)


def update_records(particule_idx,particule,records,xmu_path=Path("./")):
    """
        records=[]
    """
    particule.chemical_formula=""
    for elt in sorted(particule.list_elt):
        particule.chemical_formula=f"{particule.chemical_formula}{elt}{len(particule.pos_elt[elt])}"

    for i, atm in enumerate(particule.atoms):
        records.append({
            'particule_id': particule_idx,
            'particule_composition': particule.chemical_formula,
            'particule_Epot': particule.Epot/len(particule.atoms),
            'particule_natom': len(particule.atoms),
            'atome_idx': i,
            'espece': atm.elt,
            'CN': atm.CN,
            'GCN': atm.GCN,
            'Esite': atm.Esite,
            'local_composition':atm.local_composition,
            'xmu_path':f"{xmu_path}/{atm.elt}_{i}/xmu.dat",
        })
    return records
def add_particle(jsonl_path, input_nfo, add_timestamp=True, ajouter_si_absent_id=True):
    """
    Ajoute une entrée (dict) à la fin du fichier JSONL, sans réécrire le fichier entier.
 
    Parameters
    ----------
    jsonl_path : str ou Path
        Chemin du fichier .jsonl (créé s'il n'existe pas).
    input : dict
        Métadonnées de la particule (voir schéma dans le docstring du module).
    add_timestamp : bool
        Si True et si la clé 'date_generation' est absente, l'ajoute automatiquement (UTC, ISO 8601).
    ajouter_si_absent_id : bool
        Si True, vérifie que 'particule_id' n'existe pas déjà dans le fichier avant d'ajouter
        (évite les doublons silencieux). Coûte une lecture complète du fichier existant.
    """
    input_nfo = dict(input_nfo)  # copie défensive
 
    if add_timestamp and "date_generation" not in input_nfo:
        input_nfo["date_generation"] = datetime.now(timezone.utc).isoformat()
 
    jsonl_path = Path(jsonl_path)
 
    if ajouter_si_absent_id and "particule_id" in input_nfo and jsonl_path.exists():
        ids_existants = _lister_ids(jsonl_path)
        if input_nfo["particule_id"] in ids_existants:
            raise ValueError(
                f"particule_id '{input_nfo['particule_id']}' already exists in {jsonl_path}"
            )
 
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(input_nfo, ensure_ascii=False,indent="\t") + "\n")

def _lister_ids(chemin_jsonl):
    """Lecture rapide des seuls particule_id déjà présents (pour la vérification de doublons)."""
    ids = set()
    with open(chemin_jsonl, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            obj = json.loads(ligne)
            if "particule_id" in obj:
                ids.add(obj["particule_id"])
    return ids
# ##################################################################################
def mk_in_silico_data_v2(config,idx):
    ################################################################################
    NP=[]
    records=[]
    #__________________________________
    # Construction of the nanoparticle
    #__________________________________
    NP.append(Crystal())
    NP[-1].build(elt=config['NP']['structure']['composition'][random.randrange(0, len(config['NP']['structure']['composition']))],
                 a=config['NP']['structure']['a'],
                 radius=config['NP']['structure']['radius'],
                 materials='NP')
    NP[-1].origin_at_mass_center()
    logger.info(f"Number of atoms={len(NP[-1].atoms)}")
    NP[-1].set_composition(config['NP']['structure']['composition'],
                           seed=config['NP']['seed'])
    #__________________________________
    # structural optimization
    #__________________________________
    NP[-1].optimize_ase(config)
    NP[-1].get_element_distribution()
    savedir=config['run_dir']/config['simul_dir']/str(idx)
    NP[-1].save(prefix="NP",fmt='xyz',savedir=savedir)
    #_________________________________________________________________________
    # calculation of EXAFS spectra for each of the atoms constituting the NP
    #_________________________________________________________________________
    compute_EXAFS_spectrum(config,NP[-1],feff_dir=savedir)


    records=update_records(idx,NP[-1],records,xmu_path=savedir)
    
    df = pd.DataFrame(records)
    comp_df = pd.json_normalize(df['local_composition'])
    comp_df.columns = [f'frac_{c}' for c in comp_df.columns]
    df = pd.concat([df.reset_index(drop=True), comp_df], axis=1)
    # Sérialiser local_composition (dict) en JSON string pour compatibilité Parquet
    df_save = df.copy()
    df_save['local_composition'] = df_save['local_composition'].apply(json.dumps)
    df_save.to_parquet(savedir/'NP.parquet', engine='pyarrow', compression='snappy')


 
    add_particle(f"{savedir}/NP.jsonl", {
        "particule_id": idx,
        'composition': NP[-1].chemical_formula,
        'Epot': NP[-1].Epot/len(NP[-1].atoms),
        'natom': len(NP[-1].atoms),
        "seed": config['NP']['seed'],
        "files": {
            "xyz_file": "NP.xyz",
            "parquet_file": "NP.parquet",
        },
    })

    

        
# ##################################################################################
def mk_in_silico_data(config):
    ################################################################################
    # _______________________________________
    #
    # Etape 1 : construire la nanoparticules
    # _______________________________________
    #   Etape 1.1 : la structure
    if config['NP']['status']:
        NP=Crystal()
        NP.build(a=config['NP']['structure']['a'],
                 radius=config['NP']['structure']['radius'],
                 materials='NP')
        NP.origin_at_mass_center()
        logger.info(f"min={NP.qmin} max={NP.qmax}")
        logger.info(f"Mass center={NP.MC}")
        logger.info(f"Number of atoms={len(NP.atoms)}")
        #   Etape 1.2 : la distribution chimique

        NP.set_composition(config['NP']['structure']['composition'],seed=config['NP']['seed'])
        savedir=config['run_dir']/config['simul_dir']/str(config['NP']['seed'])/config['train']['nfo_dir']/"XYZ"
        savedir.mkdir(parents=True, exist_ok=True)

        NP.save(prefix="NP",fmt='xyz',savedir=savedir)
    
        #   Etape 1.3 : (optionnelle) l'optimisation structurale et/ou chimique
        if config['NP']['structure']['optimization']:
            NP.optimize_ase(config)

    # ___________________________________________________________________
    #
    # Etape 2 : construire les cartes de probabilité de présence atomique
    # ___________________________________________________________________
    if config['atomic presence probability map']['status']:
        NP.xyz2slice(config,
                     savedir=config['run_dir']/config['simul_dir']/str(config['NP']['seed'])/config['train']['prob_maps_img_dir'])


        logger.info(f"{NP.qmin[0]} {NP.qmax[0]} {NP.qmin[1]} {NP.qmax[1]}")
        logger.info(f"{config['image']['xmin']} {config['image']['xmax']} {config['image']['ymin']} {config['image']['ymax']}")
    
    # ___________________________________________________________________
    # Etape 3 : construire l'image TEM
    #   abTEM : https://github.com/abTEM/abTEM
    #   https://abtem.readthedocs.io/en/latest/intro.html#
    # ___________________________________________________________________
    if config['abtem']['status']:
        NP.abTEM(config,
                 savedir=config['run_dir']/config['simul_dir']/str(config['NP']['seed'])/config['train']['TEM_img_dir'])

    # ____________________________________________________
    # Etape 4 : construire les spectres EXAFS
    # ____________________________________________________
    feff_dir=config['run_dir']/config['simul_dir']/str(config['NP']['seed'])/config['train']['nfo_dir']/"feff_input_files"
    if config['feff']['status']:
        feff_dir.mkdir(parents=True, exist_ok=True)
        base_dir=os.getcwd()
        if config['feff']['status']:
            for atm in NP.atoms:
                config['feff']['parameters']['input_save_dir']=f"{feff_dir}/{atm.elt}_{atm.idx}"
                os.makedirs(config['feff']['parameters']['input_save_dir'], exist_ok=True)
                
                os.chdir(f"{config['feff']['parameters']['input_save_dir']}")
                NP.FEFF_create_input_file(config['feff']['parameters'],absorber_idx=atm.idx)
                NP.FEFF_run(config['feff']['parameters'])
                dossier = Path("./")
                for element in dossier.iterdir():
                    if element.is_file() and element.name != "xmu.dat":
                        element.unlink()
                os.chdir(base_dir)
    
    # Initialise automatiquement avec une liste vide
    series = defaultdict(list)
    logger.info(NP.list_elt)
    base_dir=os.getcwd()
    logger.info(f"{base_dir}")
    for atm in NP.atoms:
        xmu_dir=f"{feff_dir}/{atm.elt}_{atm.idx}"
        try:
            k, chi = numpy.loadtxt(f"{xmu_dir}/xmu.dat", comments='#', usecols=(2, 5), unpack=True)
            series[atm.elt].append((k,chi))
        except:
            logger.error(f"file {xmu_dir}/xmu.dat not found!")
    exafs_dir=config['run_dir']/config['simul_dir']/str(config['NP']['seed'])/config['train']['EXAFS_dir']
    exafs_dir.mkdir(parents=True, exist_ok=True)


    for kexpo in range(4):
        if kexpo==0:
            ylbl=r"$\chi(E)$"
            savename="chi(k)"
            header="# k (A^-1)   chi(k)"
        elif kexpo==1:
            ylbl=r"$k\cdot\chi(E)$"
            savename="kchi(k)"
            header="# k (A^-1)   k.chi(k)"
        elif kexpo==2:
            ylbl=r"$k^2\cdot\chi(E)$"
            savename="k2chi(k)"
            header="# k (A^-1)   k^2.chi(k)"
        elif kexpo==3:
            ylbl=r"$k^3\cdot\chi(E)$"
            savename="k3chi(k)"
            header="# k (A^-1)   k^3.chi(k)"

        for elt in NP.list_elt:
            logger.info(f"{elt} : {len(series[elt])}")
            
            k,chiglob,dev=HBPy.Molecule.Tools.mk_mean(series[elt],expo=kexpo)
            logger.info(f"Saving {exafs_dir}/{savename}_{elt}.dat")
            numpy.savetxt(
                f"{exafs_dir}/{savename}_{elt}.dat",                       # Le nom du fichier
                numpy.column_stack((k,chiglob)),               # Le tableau 2D créé juste au-dessus
                fmt='%.6f',                                         # Le format (ici : 6 chiffres après la virgule)
                delimiter='    ',                                   # Le séparateur entre les colonnes (ici : 4 espaces)
                header=header,
                comments='# '                                       # (Optionnel) Le caractère pour commenter l'en-tête
            )
# #########################################################################################
if __name__ == "__main__":
    mk_in_silico_data()

