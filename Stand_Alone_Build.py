from Molecule.Crystal import Crystal
import numpy as np



def main():
    radius_box=30 # ang
    radius_NP=9.0  # ang
    N=int(radius_box/3.92)  # 3.92 est le parametre de maille du Pt en angstreom
    Bulk=Crystal()



    Bulk.build(Nx=N,Ny=N,Nz=N,elt='Pt')
    # découpe de la nanoparticule
    Bulk.origin_at_mass_center()
    molecule=Bulk.transform(radius=radius_NP)
    print(len(molecule.atoms))
    molecule.get_element_distribution()
    molecule.MassCenter()
    molecule.origin_at_mass_center()
    molecule.get_structure()
    composition=[
        (5.8,'Ir'),
        (7.8,'Ru'),
        (float('inf'),'Ni')
    ]
    molecule.core_shell(composition)
    print(f"natom={len(molecule.atoms)}")
    molecule.get_element_distribution()
    for elt in molecule.list_elt:
        print(elt,len(molecule.pos_elt[elt]),100.0*len(molecule.pos_elt[elt])/len(molecule.atoms))
    molecule.save(prefix="NP")
    
    molecule.mixing(nexchange=10*len(molecule.atoms),seed=0)
    molecule.save(prefix="NPmix")
    molecule.get_element_distribution()
    for elt in molecule.list_elt:
        print(elt,len(molecule.pos_elt[elt]),100.0*len(molecule.pos_elt[elt])/len(molecule.atoms))
    for elt in molecule.list_elt:
        for atm in molecule.atoms:
            atm.elt=elt
        molecule.save(prefix=f"NP{elt}")
        print(f"NP{elt}")
    print(f"natom={len(molecule.atoms)}")
# ##########################################################################################
# Point d’entrée du programme
if __name__ == "__main__":
    main()
