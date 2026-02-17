"""
Script pour calculer la différence entre deux images
"""

import cv2
import numpy as np
import argparse


def load_images(image1_path, image2_path):
    """
    Charge deux images depuis leurs chemins
    
    Args:
        image1_path: Chemin vers la première image
        image2_path: Chemin vers la deuxième image
    
    Returns:
        tuple: (image1, image2) ou (None, None) en cas d'erreur
    """
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None:
        print(f"Erreur: Impossible de charger l'image {image1_path}")
        return None, None
    
    if img2 is None:
        print(f"Erreur: Impossible de charger l'image {image2_path}")
        return None, None
    
    # Vérifier que les images ont la même taille
    if img1.shape != img2.shape:
        print(f"Attention: Les images n'ont pas la même taille")
        print(f"Image 1: {img1.shape}, Image 2: {img2.shape}")
        print("Redimensionnement de l'image 2 à la taille de l'image 1...")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    return img1, img2


def calculate_difference(img1, img2):
    """
    Calcule la différence absolue entre deux images
    
    Args:
        img1: Première image
        img2: Deuxième image
    
    Returns:
        Image de différence
    """
    return cv2.absdiff(img1, img2)


def normalize_image(image):
    """
    Normalise une image pour étendre les valeurs sur toute la plage [0, 255]
    
    Args:
        image: Image à normaliser
    
    Returns:
        Image normalisée
    """
    # Normaliser chaque canal séparément pour les images couleur
    if len(image.shape) == 3:
        normalized = np.zeros_like(image)
        for i in range(image.shape[2]):
            normalized[:, :, i] = cv2.normalize(
                image[:, :, i], 
                None, 
                alpha=0, 
                beta=255, 
                norm_type=cv2.NORM_MINMAX
            )
        return normalized
    else:
        # Image en niveaux de gris
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def invert_colors(image):
    """
    Inverse les couleurs d'une image
    
    Args:
        image: Image à inverser
    
    Returns:
        Image avec couleurs inversées
    """
    return 255 - image


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description='Calcule la différence entre deux images'
    )
    parser.add_argument('image1', help='Chemin vers la première image')
    parser.add_argument('image2', help='Chemin vers la deuxième image')
    parser.add_argument(
        '-o', '--output',
        default='difference.png',
        help='Chemin de sortie pour l\'image de différence (défaut: difference.png)'
    )
    parser.add_argument(
        '-i', '--invert',
        action='store_true',
        help='Inverser les couleurs de l\'image de différence'
    )
    parser.add_argument(
        '-n', '--normalize',
        action='store_true',
        help='Normaliser l\'image de différence pour améliorer le contraste'
    )
    
    args = parser.parse_args()
    
    # Charger les images
    print(f"Chargement des images...")
    img1, img2 = load_images(args.image1, args.image2)
    
    if img1 is None or img2 is None:
        return
    
    print(f"Images chargées avec succès!")
    print(f"Taille: {img1.shape}")
    
    # Calculer la différence
    print(f"Calcul de la différence...")
    diff = calculate_difference(img1, img2)
    
    # Afficher les valeurs min/max avant normalisation
    print(f"Avant traitement - Min: {np.min(diff)}, Max: {np.max(diff)}")
    
    # Normaliser si demandé
    if args.normalize:
        print("Normalisation de l'image...")
        diff = normalize_image(diff)
        print(f"Après normalisation - Min: {np.min(diff)}, Max: {np.max(diff)}")
    
    # Inverser les couleurs si demandé
    if args.invert:
        print("Inversion des couleurs...")
        diff = invert_colors(diff)
    
    # Sauvegarder le résultat
    cv2.imwrite(args.output, diff)
    print(f"Image de différence sauvegardée: {args.output}")
    
    # Calculer quelques statistiques
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mean_diff = np.mean(diff_gray)
    max_diff = np.max(diff_gray)
    min_diff = np.min(diff_gray)
    
    # Compter les pixels différents (en tenant compte de l'inversion)
    if args.invert:
        nb_pixels_diff = np.count_nonzero(diff_gray < 255)
    else:
        nb_pixels_diff = np.count_nonzero(diff_gray > 0)
    
    percent_diff = (nb_pixels_diff / diff_gray.size) * 100
    
    print(f"\nStatistiques de la différence:")
    print(f"  - Valeur minimale: {min_diff}")
    print(f"  - Valeur moyenne: {mean_diff:.2f}")
    print(f"  - Valeur maximale: {max_diff}")
    print(f"  - Pixels différents: {nb_pixels_diff} ({percent_diff:.2f}%)")


if __name__ == "__main__":
    main()
