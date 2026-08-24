import numpy
##########################################################################################################################
def resampling(filename,xmin=2.0,xmax=8.0,N=100,colx=0,coly=1):
    """
    Méthode pour rééchantilloner des données contenues dans le fichier filename, en N points régulièrement
    répartis entre [xmin,xmax].
    
    * Les 4 étapes de l'algorithme
      ÉTAPE 1 : Chargement
    ────────────────────
    Charge un fichier texte avec 2 colonnes (x, y)
    Les lignes commençant par '#' sont ignorées (commentaires)
    
    ÉTAPE 2 : Validation
    ─────────────────────
    Vérifie que [xmin, xmax] est dans les limites du fichier
    Si non → lève une erreur (ValueError)
    
    ÉTAPE 3 : Rééchantillonnage
    ────────────────────────────
    Crée N points x uniformément espacés entre xmin et xmax
    Interpole les valeurs y correspondantes
    
    ÉTAPE 4 : Retour
    ────────────────
    Retourne un array (N, 2) avec les données rééchantillonnées
    
    Args:
        filename (str): Chemin du fichier de données
        xmin (float): Valeur x minimale souhaitée (défaut: 2.0)
        xmax (float): Valeur x maximale souhaitée (défaut: 8.0)
        N (int): Nombre de points dans le nouvel échantillon (défaut: 100)
    
    Returns:
        numpy.ndarray: Array de shape (N, 2) avec [x_nouveau, y_nouveau]
    
    Raises:
        ValueError: Si xmin < min(x_original) ou xmax > max(x_original)
    
    Example: Charger et rééchantillonner un spectre XAS
        >>> data = resampling(
                      filename="spectrum.txt",
                      xmin=4.5,           # Énérgie min (keV)
                      xmax=5.5,           # Énérgie max (keV)
                      N=500)              # 500 points réguliers
        >>> print(data.shape)  # Résultat : array de shape (500, 2) contenant exactement 500 points espacés uniformément

    
    """
    # Chargement du fichier
    x_original, y_original = numpy.loadtxt(filename, comments='#', usecols=(colx,coly),unpack=True)
    # Test de sécurité
    # On vérifie si xmin est trop petit OU si xmax est trop grand
    if xmin < x_original.min() or xmax > x_original.max():
        raise ValueError(
            f"🚨 Erreur de bornes ! Vous demandez un intervalle [{xmin}, {xmax}] "
            f"qui déborde du fichier original.\n"
            f"👉 Plage réelle du fichier : [{x_original.min():.3f}, {x_original.max():.3f}]"
    )

    # Si le test passe, le script continue en toute sécurité
    x_nouveau = numpy.linspace(xmin, xmax, N)
    y_nouveau = numpy.interp(x_nouveau, x_original, y_original)
    return numpy.array([x_nouveau,y_nouveau])

