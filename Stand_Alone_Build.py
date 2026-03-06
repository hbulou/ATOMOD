from Molecule.Crystal import Crystal
import numpy as np



def main():
    radius_box=30 # ang
    radius_NP=8.0  # ang
    N=int(radius_box/3.92)  # 3.92 est le parametre de maille du Pt en angstreom
    Bulk=Crystal()
    r1=5.7
    r2=7.0
    r3=8.0
    radius_comp=[r1,r2,r3]
    composition=['Ir','Ru','Ni']
    Bulk.build(Nx=N,Ny=N,Nz=N,elt=composition[0])
    # découpe de la nanoparticule
    Bulk.origin_at_mass_center()
    molecule=Bulk.transform(radius=radius_NP)
    print(len(molecule.atoms))
    molecule.get_element_distribution()
    molecule.MassCenter()
    molecule.origin_at_mass_center()
    molecule.get_structure()
    for atm in molecule.atoms:
        d=atm.distance_from_(molecule.MC)
        if d<= radius_comp[0]:
            atm.elt=composition[0]
        elif d<= radius_comp[1]:
            atm.elt=composition[1]
        else:
            atm.elt=composition[-1]
        print(atm.elt,atm.q,d)
    print(f"natom={len(molecule.atoms)}")
    molecule.get_element_distribution()
    for elt in molecule.list_elt:
        print(elt,len(molecule.pos_elt[elt]),100.0*len(molecule.pos_elt[elt])/len(molecule.atoms))
    #for elt in composition:
    #    if elt not in molecule.pos_elt:
    #        molecule.pos_elt[elt]=[]
    #        #print(f"### {elt} {molecule.pos_elt[elt]} {len(molecule.pos_elt[elt])} {len(molecule.atoms)}")
    #        print(f"### {elt} {molecule.pos_elt[elt]} -> stoechiometry {len(molecule.pos_elt[elt])/len(molecule.atoms)}")
    molecule.save(prefix="NP")
# ##########################################################################################
# Point d’entrée du programme
if __name__ == "__main__":
    main()
