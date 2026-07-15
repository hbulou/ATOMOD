from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, UpSampling1D, Concatenate
from tensorflow.keras.models import Model

def build_descripteur_vers_exafs(n_features, n_k_points, n_especes=5):
    """
    Ajustez n_start et le nombre de couches UpSampling1D pour que n_start * 2^(nb_upsampling) corresponde exactement à n_k_points 
    (ou proche, avec un recadrage/Cropping1D final si besoin).
    """
    inputs = Input(shape=(n_features,), name='descripteurs')
    
    # Encodeur dense : compresse les descripteurs en représentation latente
    x = Dense(64, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)

    # Point de départ pour le décodeur convolutif : petite séquence 1D
    n_start = n_k_points // 8  # à ajuster selon n_k_points et le facteur d'upsampling total
    x = Dense(n_start * 32, activation='relu')(x)
    x = Reshape((n_start, 32))(x)

    # Décodeur convolutif : upsampling progressif jusqu'à la résolution cible
    x = UpSampling1D(2)(x)
    x = Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, kernel_size=5, padding='same', activation='relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, kernel_size=5, padding='same', activation='relu')(x)

    # Sortie finale : un seul canal, le spectre k^n*chi(k)
    spectre = Conv1D(1, kernel_size=3, padding='same', activation=None, name='spectre_exafs')(x)

    # Sortie additionnelle si option multi-tâche (Esite)
    esite_pred = Dense(1, activation=None, name='esite_predite')(x if False else Dense(32, activation='relu')(inputs))

    return Model(inputs=inputs, outputs=[spectre, esite_pred])

def EXAFS_model(config):
    """
    Architecture : encodeur dense → décodeur convolutif 1D
    """
    pass
