import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# 1. Charger vos données (remplacez par la lecture de votre fichier XYZ)
# Supposons que 'z_coords' est un array numpy contenant toutes vos cotes z
z_coords = np.loadtxt("data/xyz/NP_2050.xyz", skiprows=2, usecols=3) # Exemple pour format XYZ standard
print(z_coords)
# 2. Calculer le KDE (densité de probabilité)
density = gaussian_kde(z_coords, bw_method=0.05) # Ajuster bw_method selon le bruit
z_range = np.linspace(min(z_coords), max(z_coords), 1000)
z_density = density(z_range)

# 3. Trouver les pics
peaks, _ = find_peaks(z_density, height=np.max(z_density)*0.1)
z_planes = z_range[peaks]

# 4. Visualisation
plt.plot(z_range, z_density)
plt.plot(z_planes, z_density[peaks], "x")
plt.title(f"Cotes des plans détectées : {z_planes}")
plt.xlabel("z")
plt.ylabel("Densité")
plt.show()

print("Cotes des plans :", z_planes)
