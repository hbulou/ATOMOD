import os
import sys
sys.path.append('/home/bulou/ownCloud/code/site-packages/')
from bulou.Crystal import Crystal
from bulou.ForceField import ForceField
import bulou.Atom
import random
import abtem
import ase
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

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
    
####################################################################################################################
def mk_TEM(seed,radius=5.0,composition=['Pt','Pd'],N=20,root_dir="data"):

    TEM_output_dir=root_dir+"/train/images"
    os.makedirs(TEM_output_dir, exist_ok=True)
    prob_output_dir = root_dir+"/train/prob_maps"
    os.makedirs(prob_output_dir, exist_ok=True)
    xyz_output_dir = root_dir+"/xyz"
    os.makedirs(xyz_output_dir, exist_ok=True)

    
    ##############################################################################################"
    #
    # generation in silico de la NP
    #
    ##############################################################################################"
    Bulk=Crystal(elt=composition[0],Nx=N,Ny=N,Nz=N)
    Bulk.origin_at_mass_center()
    molecule=Bulk.transform(radius=radius)
            
    molecule.get_element_distribution()
    print(len(molecule.atoms))
    for elt in composition:
        if elt not in molecule.pos_elt:
            molecule.pos_elt[elt]=[]
    print(f"### {elt} {molecule.pos_elt[elt]} -> stoechiometry {len(molecule.pos_elt[elt])/len(molecule.atoms)}")
    molecule.get_structure()

    print(100*"-","\n### Building alloy")
    print("### element(s) :",composition)
    print("### ",molecule.pos_elt)
    stoechiometry=1.0/len(composition)
    nmin=len(molecule.pos_elt[composition[0]])*stoechiometry
    random.seed(seed)
    idxfill=1
    while len(molecule.pos_elt[composition[0]])>nmin:
        # on choisit au hasard un des atomes de l'espèce en excés
        n = random.randrange(0, len(molecule.pos_elt[composition[0]]))   # 0 à 10 (11 exclu)
        if len(molecule.pos_elt[composition[idxfill]])>=nmin:
            idxfill=idxfill+1
        idx=molecule.pos_elt[composition[0]].pop(n)
        molecule.pos_elt[composition[idxfill]].append(idx)
        molecule.atoms[idx].elt=composition[idxfill]
        print(n,idx,"->",
              molecule.pos_elt[composition[0]],
              "# ",
              composition[idxfill],
              molecule.pos_elt[composition[idxfill]])

    molecule.get_element_distribution()
    molecule.get_structure()
    print("### elt=",molecule.pos_elt)
    
    molecule.save(prefix=f"{xyz_output_dir}/NP_{seed:04d}_unrlx",fmt='xyz')

    # -------------- STRUCTURAL OPTIMIZATION -------------------------------
    molecule.FF=ForceField()
    molecule.optimize(tol=1.0e-8)

    molecule.get_element_distribution()
    molecule.get_structure()
    print("### (xmin,ymin,zmin)=",molecule.qmin)
    print("### (xmax,ymax,zmax)=",molecule.qmax)
    
    ##############################################################################################"
    #
    # generation des images TEM
    #
    ##############################################################################################"

    # Crée une boîte vide de 10x10x10 Å
    # pour l'instant on passe par ASE pour fournir la structure à abtem
    cellsize=20.0 # taille de la zone à simuler
    atoms = ase.Atoms(cell=[cellsize,cellsize,cellsize], pbc=True)

    for atm in molecule.atoms:
        atoms += ase.Atom(bulou.Atom.Z_from_elt[atm.elt], (atm.q[0],atm.q[1],atm.q[2]))
    atoms.center()
    for atm in atoms:
        print(atm)
        molecule.atoms[atm.index].q[0]=atm.x
        molecule.atoms[atm.index].q[1]=atm.y
        molecule.atoms[atm.index].q[2]=atm.z
    molecule.get_structure()
    molecule.save(prefix=f"{xyz_output_dir}/NP_{seed:04d}",fmt='xyz')



    z_coords=[]
    for atm in molecule.atoms:
        z_coords.append(atm.q[2])
    zp,dzmean=get_z_plane(z_coords)    
        
    dz=dzmean
    dx=0.04
    potential = abtem.Potential(atoms,
                                slice_thickness=dz,
                                sampling= dx)
    
    #print(dir(potential))
    #print(dir(abtem.visualize))
    #potential.show()
    # fonction d'onde électronique qui est diffusée
    plane_wave = abtem.PlaneWave(sampling =0.01 , energy =300e3  )
    #exit_wave = plane_wave.multislice(atoms)
    exit_wave = plane_wave.multislice(potential)
    # exécution du calcul
    exit_wave.compute()
    # Dans les expériences HRTEM réalistes, les fonctions d'onde doivent être amplifiées par une
    # lentille d'objectif, ce qui introduit des aberrations et élimine de fait les grands angles
    # de diffusion.
    # ici on applique un flou de 50 angstreom et une ouverture d'objectif de 20 mrad
    #exit_wave.apply_ctf(defocus=-30,
    #                    focal_spread=40,
    #                    semiangle_cutoff=20)#.intensity()#.show(cbar=True);
    #image=exit_wave.intensity()#.show(common_color_scale=True, cbar=True);

        
    ctf = abtem.CTF(defocus =200 , focal_spread =40, semiangle_cutoff=20 )
    image_wave = ctf.apply(exit_wave) 
    image = image_wave.intensity()
    fig, axes = plt.subplots(1,1, figsize=(10,10), gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    image.show(ax=axes)
    print(f"### abtem : extent={exit_wave.extent}") # taille réelle en angstroems
    print(f"### abtem : gpts={exit_wave.gpts}")     # resolution en pixel
    print(f"### abtem : sampling={exit_wave.sampling}") # taille d'un pixel en angstroem
    print(f"### abtem : exit wave shape={exit_wave.shape}")  # dimension de la matrice
    print(f"### abtem : potential shape={potential.shape}")  # dimension de la matrice
    print(f"### abtem : potential extent={potential.extent}")  # dimension de la matrice
    plt.axis("off")
    # Sauvegarde en PNG (ou autre format suivant l’extension)
    #plt.savefig(self.TEM_img.name, dpi=300, bbox_inches="tight")
    
    filename = os.path.join(TEM_output_dir, f"img_{seed:04d}.png")
    plt.savefig(filename,
                dpi=150,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1,
                facecolor='white')
    
    #plt.savefig(filename, dpi=150, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

    ##############################################################################################"
    #
    # generation des cartes de probabilite de presence atomique
    #
    ##############################################################################################"
    nx=potential.shape[1]
    ny=potential.shape[2]
    nvacc=2
    nz=len(zp)+2*nvacc
    dy=dx
    print(f"(nx,ny,nz)=({nx},{ny},{nz} (dx,dy,dz)=({dx},{dy},{dz})")
    x = np.linspace(0.0, potential.extent[0], nx)
    y = np.linspace(0.0, potential.extent[1], ny)
    z = np.linspace(zp[0]-nvacc*dzmean, zp[-1]+nvacc*dzmean, nz)
    print(f"zp={zp}")
    print(f"z={z}")
    print(f"z/dz={z/dz}")
    volumes = {}  # dict: espèce -> volume 3D
    for sp in composition:
        volumes[sp] = np.zeros((nx, ny, nz), dtype=float)
        
    sigma = 0.6  # en Å, largeur de la gaussienne ~ rayon atomique ou un peu moins
    for atom in molecule.atoms:
        sp = atom.elt
        vol = volumes[sp]
        # Position de l’atome
        ax, ay, az = atom.q[0], atom.q[1], atom.q[2]
        #     # Indices du voisinage à affecter (±3 sigma)
        ix_center = int((ax) / dx)
        iy_center = int((ay) / dy)
        iz_center = int((az-z[0]) / dz)
        #print(ax,ay,az,ix_center,iy_center,iz_center)
        r = int(3 * sigma / dx)  # rayon en nombre de voxels
        rz = int(3 * sigma / dz)  # rayon en nombre de voxels

        ix_min = max(ix_center - r, 0)
        ix_max = min(ix_center + r + 1, nx)
        iy_min = max(iy_center - r, 0)
        iy_max = min(iy_center + r + 1, ny)
        iz_min = max(iz_center - rz, 0)
        iz_max = min(iz_center + rz + 1, nz)
        print(atom.elt,ix_min,ix_max,iy_min,iy_max,iz_min,iz_max)
        #     # Sous-grille locale
        Xsub = x[ix_min:ix_max]
        Ysub = y[iy_min:iy_max]
        Zsub = z[iz_min:iz_max]
        
        Xg, Yg, Zg = np.meshgrid(Xsub, Ysub, Zsub, indexing="ij")
        
        dx2 = (Xg - ax)**2
        dy2 = (Yg - ay)**2
        dz2 = (Zg - az+z[0])**2
        gauss = np.exp(-(dx2 + dy2 + dz2) / (2 * sigma**2))
        
        vol[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] += gauss



    xmin=0.0
    ymin=0.0
    zmin=z[0]
    xmax=potential.extent[1]
    ymax=potential.extent[1]
    zmax=z[-1]
    print(f"zmin={zmin} zmax={zmax}")
    for sp in composition:
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
            filename = os.path.join(prob_output_dir, f"img_{seed:04d}_{sp}_{k:04d}.png")
            plt.savefig(filename,
                        dpi=150,
                        bbox_inches='tight',
                        transparent=True,
                        pad_inches=0.1,
                        facecolor='white')
            
            #plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
    del molecule
    del atoms
####################################################################################################################
def main():
    config={
        'root_dir':  "data2",
        'idx_start': 2050,
        'idx_end':   2050
        }
    
    for seed in range(config['idx_start'],config['idx_end']+1):
        mk_TEM(seed,composition=['Rh','Ir'],root_dir=config['root_dir'])
# ##########################################################################################
# Point d’entrée du programme
if __name__ == "__main__":
    main()
