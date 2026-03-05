import sys
sys.path.append('/home/bulou/ownCloud/code/site-packages/')
from Molecule.Atom import Atom,COV_RADIUS,CPK_COLOR
from Molecule.Crystal import Crystal
from Molecule.ForceField import ForceField
import random
import pandas as pd



def HEAS(composition=['Pd','Pt','Rh','Ir','Ru'],
         radius=5.0,
         seed=0,
         N=7):

    Bulk=Crystal(Nx=N,Ny=N,Nz=N,elt=composition[0])
    Bulk.origin_at_mass_center()
    NP=Bulk.transform(radius=radius)
    NP.get_element_distribution()
    for elt in composition:
        if elt not in NP.list_elt:
            NP.pos_elt[elt]=[]
            print(elt,"->",0.0)
        else:
            print(elt,NP.pos_elt[elt],"->",len(NP.pos_elt[elt])/len(NP.atoms))

    
    stoechiometry=1.0/len(composition)
    nmin=len(NP.pos_elt[composition[0]])*stoechiometry
    #print(len(NP.pos_elt[composition[0]]),stoechiometry,"nmin=",nmin)
    random.seed(seed)
    idxfill=1
    while len(NP.pos_elt[composition[0]])>nmin:
        n = random.randrange(0, len(NP.pos_elt[composition[0]]))   # 0 à 10 (11 exclu)
        if len(NP.pos_elt[composition[idxfill]])>=nmin:
            idxfill=idxfill+1
        idx=NP.pos_elt[composition[0]].pop(n)
        NP.pos_elt[composition[idxfill]].append(idx)
        NP.atoms[idx].elt=composition[idxfill]
        print(n,idx,"->",NP.pos_elt[composition[0]],"# ",composition[idxfill],NP.pos_elt[composition[idxfill]])

    NP.get_element_distribution()
    NP.get_structure()

    return NP
