import numpy as np
import cv2

correlations = []
for i in range(50):  # Vérifier 50 images
    rh = cv2.imread(f'data/train/prob_maps/img_{i:04d}_Rh_0000.png', 0) / 255.0
    ir = cv2.imread(f'data/train/prob_maps/img_{i:04d}_Ir_0000.png', 0) / 255.0
    corr = np.corrcoef(rh.ravel(), ir.ravel())[0, 1]
    correlations.append(corr)

print(f"Corrélation moyenne Rh-Ir : {np.mean(correlations):.3f}")
