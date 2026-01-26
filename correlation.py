import numpy as np
import cv2

# https://claude.ai/chat/234ac0b5-5f85-4358-a238-4e235481991e
# permet de vérifier si les cartes de probabilité de présence atomique pour les différentes espèces chimiques permettent
# la séparation chimique des éléments.
# Interprétation :
#
#  * < 0.7 → Séparation possible ✅
#  * 0.7-0.9 → Difficile mais faisable ⚠️
#  * > 0.9 → Les données sont trop corrélées, séparation quasi-impossible ❌



correlations = []
for i in range(50):  # Vérifier 50 images
    rh = cv2.imread(f'data/train/prob_maps/img_{i:04d}_Rh_0000.png', 0) / 255.0
    ir = cv2.imread(f'data/train/prob_maps/img_{i:04d}_Ir_0000.png', 0) / 255.0
    corr = np.corrcoef(rh.ravel(), ir.ravel())[0, 1]
    correlations.append(corr)

print(f"Corrélation moyenne Rh-Ir : {np.mean(correlations):.3f}")
