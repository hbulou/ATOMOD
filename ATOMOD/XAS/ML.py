import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import json

def load_particle(chemin_jsonl):
    """
    Relit un fichier contenant plusieurs objets JSON indentés (multi-lignes)
    concaténés à la suite (écrits avec json.dumps(..., indent="\t") + "\n").
    """
    with open(chemin_jsonl, "r", encoding="utf-8") as f:
        contenu = f.read()

    decoder = json.JSONDecoder()
    entrees = []
    pos = 0
    n = len(contenu)

    while pos < n:
        # ignorer les espaces/retours à la ligne entre deux objets
        while pos < n and contenu[pos].isspace():
            pos += 1
        if pos >= n:
            break
        obj, fin = decoder.raw_decode(contenu, pos)
        entrees.append(obj)
        pos = fin

    return pd.json_normalize(entrees, sep=".") 

def EXAFS_model(config):
    """
     entrainement du modèle
     ______________________________
     DONNES
     ______________________________
     - X_train : les descripteurs d'entrée (le vecteur de taille n_features) pour l'ensemble d'entraînement.
     - Le deuxième argument est un dictionnaire de cibles, une par sortie nommée du modèle :
         * 'spectre_exafs': Y_spectre_train → le spectre EXAFS de référence à reconstruire.
         * 'esite_predite': Y_esite_train → la valeur scalaire (énergie de site) à prédire en parallèle.
     - validation_data=(X_val, {...}) : même logique mais pour le jeu de validation, évalué à chaque epoch sans mettre à jour les poids. C'est ce qui alimente les métriques val_* utilisées par les callbacks.
     ______________________________
     HYPERPARAMETRES d'ENTRAINEMENT
     ______________________________
     - epochs=500 : jusqu'à 500 passages complets sur les données d'entraînement — mais en pratique l'arrêt anticipé (EarlyStopping) coupera probablement bien avant.
     - batch_size=64 : 64 échantillons par mise à jour de gradient.
     ______________________________
     CALLBACKS
     ______________________________
     1. EarlyStopping
       - monitor='val_spectre_exafs_loss' : surveille la perte de validation spécifique à la sortie spectre_exafs (pas la perte totale ni celle de esite_predite).
         C'est un choix important : il priorise la qualité de reconstruction du spectre EXAFS comme critère d'arrêt, même si le modèle est multi-tâche.
       - patience=30 : si cette métrique ne s'améliore pas pendant 30 epochs consécutives, l'entraînement s'arrête.
       - restore_best_weights=True : à la fin, le modèle récupère automatiquement les poids correspondant au meilleur epoch (celui avec la plus faible val_spectre_exafs_loss), pas les derniers poids calculés.
     2. ReduceLROnPlateau
       - Même métrique surveillée (val_spectre_exafs_loss).
       - patience=10 : si pas d'amélioration pendant 10 epochs, le taux d'apprentissage (learning rate) est réduit.
       - factor=0.5 : il est alors divisé par 2.
       - Cela permet d'affiner l'optimisation en fin d'entraînement quand la descente de gradient stagne, avant que l'EarlyStopping (patience plus longue, 30) ne déclenche l'arrêt complet.
    ______________________________
    Résultat
    ______________________________
    history est un objet Keras History qui contient, epoch par epoch, toutes les valeurs de perte (loss totale, spectre_exafs_loss, esite_predite_loss, et leurs équivalents val_*,
    plus toute métrique éventuellement définie dans model.compile). Utile ensuite pour tracer les courbes d'apprentissage.
    """

    # history = model.fit(
    #     X_train,
    #     {'spectre_exafs': Y_spectre_train, 'esite_predite': Y_esite_train},
    #     validation_data=(X_val, {'spectre_exafs': Y_spectre_val, 'esite_predite': Y_esite_val}),
    #     epochs=500,
    #     batch_size=64,
    #     callbacks=[
    #         tf.keras.callbacks.EarlyStopping(monitor='val_spectre_exafs_loss', patience=30, restore_best_weights=True),
    #         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_spectre_exafs_loss', factor=0.5, patience=10)
    #     ]
    # )

    #________________________________________________________________________________________________________________________
    #
    # Lecture des data pour l"entrainement du modèle
    #
    #________________________________________________________________________________________________________________________

    #df=load_particle("simulv3/0/NP.jsonl")
    df = pd.concat(
        [
            load_particle(f"simulv3/{idx}/NP.jsonl") for idx in range(8)
        ],
        ignore_index=True
    )
    print(df)
    exit()


    df = pd.concat([pd.read_parquet(config['run_dir']/config['simul_dir']/str(idx)/'NP.parquet', engine='pyarrow') for idx in [0,1,2,3,4]], ignore_index=True)
    print(df)
    particule_id=0
    particules_uniques = df['particule_id'].unique()
    print(particules_uniques)


    """
    On tire aléatoirement 80% des identifiants de particules pour l'entraînement et 20% pour la validation. (test_size=0.2)
    random_state=42 : graine fixe pour que le split soit reproductible d'une exécution à l'autre.
    Pourquoi c'est important : si on avait fait train_test_split directement sur les lignes de df, des lignes provenant de la même particule auraient pu se retrouver à la fois dans train et dans validation.
    Le modèle aurait alors "vu" indirectement des informations sur les particules de validation pendant l'entraînement (fuite de données), ce qui biaiserait l'évaluation en donnant une performance de validation
    artificiellement optimiste. En splittant par particule_id, on garantit qu'une particule donnée est soit entièrement en train, soit entièrement en validation, jamais les deux.
    """
    train_ids, val_ids = train_test_split(particules_uniques, test_size=0.2, random_state=42)
    print(f"train ids={train_ids}, validation ids={val_ids}")

    """
    Pour chaque ligne de df, on vérifie si son particule_id appartient à l'ensemble des IDs d'entraînement ou de validation. Cela donne deux tableaux booléens (True/False) de la même longueur que df, alignés
    ligne par ligne. .values extrait le tableau numpy sous-jacent (plutôt qu'une Series pandas), pratique pour indexer directement des arrays numpy comme X.
    """
    mask_train = df['particule_id'].isin(train_ids).values
    mask_val = df['particule_id'].isin(val_ids).values
    print(mask_train)
    print(mask_val)
    """
    On utilise ces masques pour découper de façon cohérente (même lignes conservées partout) :
       * X : les descripteurs d'entrée,
       * Y_spectre_norm : les spectres EXAFS cibles (déjà normalisés, vu le suffixe _norm),
       * Y_esite : les valeurs cibles pour la sortie annexe esite_predite.
    """
    #X_train, X_val = X[mask_train], X[mask_val]
    #Y_spectre_train, Y_spectre_val = Y_spectre_norm[mask_train], Y_spectre_norm[mask_val]
    #Y_esite_train, Y_esite_val = Y_esite[mask_train], Y_esite[mask_val]



    """
    df['espece'] contient l'élément chimique de l'atome central (le site autour duquel on calcule EXAFS/structure locale), par exemple 'Pt'.
    pd.get_dummies(...) transforme cette colonne catégorielle en plusieurs colonnes binaires (une par valeur unique rencontrée).
    [especes] réordonne/filtre explicitement les colonnes selon la liste especes définie plus haut — important pour garantir un ordre stable (pd.get_dummies seul trierait alphabétiquement ou selon l'ordre d'apparition, ce qui pourrait varier).
    .values extrait le résultat en tableau numpy.
    Résultat : une matrice (n_échantillons, 5) où chaque ligne a un 1 à la position de l'espèce centrale et des 0 ailleurs. C'est l'encodage classique "one-hot" pour une variable catégorielle.
    """
    especes=config['NP']['structure']['composition']
    espece_onehot = pd.get_dummies(df['espece'])[especes].values
    print(espece_onehot)

    frac_cols = [f'frac_{e}' for e in especes]
    X_composition = df[frac_cols].values
    print(type(X_composition),X_composition.shape,X_composition.ndim,X_composition.size)
    for i in range(len(X_composition)):
        print([elem for elem, condition in zip(especes, espece_onehot[i]) if condition],X_composition[i])
