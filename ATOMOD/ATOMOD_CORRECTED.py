import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.layers import Input,Conv2D,Dropout,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.models import Model,load_model

import cv2

def activation_function(x,name=None):
    act_fnct = tf.keras.layers.PReLU(shared_axes=[1,2],
                                     name=name+"-PReLU",
                                     alpha_initializer=tf.keras.initializers.Constant(0.01))
    return act_fnct(x)

def convolutional_layer(x, channels, kernel_size=3, name='convolutional_layer'):
    kernel_initializer="he_normal"      # ReLU / LeakyReLU
    bias_initializer="zeros"
    
    convolutional_layer = tf.keras.layers.Conv2D(channels,
                                                 kernel_size,
                                                 padding='same',
                                                 kernel_regularizer=regularization(1.0e-4),
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 name=name)

    x = activation_function(convolutional_layer(x),name=name)
    return x

def residual_block(x, channels, name='residual_block'):
    y = convolutional_layer(x, channels, name=name+'-convolution_in_residual_1')
    y = convolutional_layer(y, channels, name=name+'-convolution_in_residual_2')
    y = convolutional_layer(y, channels, name=name+'-convolution_in_residual_3')
    connection_x_y = skip_connection(x, y, name=name+'-element_wise_addition')
    return connection_x_y

def convolution_block(x, channels, name='convolution_block'):
    x = convolutional_layer(x, channels, name=name+"-convolution_1")
    x = residual_block(x, channels, name=name+"-residual_block")
    x = convolutional_layer(x,channels, name=name+"-convolution_2")
    return x

def regularization(weight_decay=1e-4):
    return tf.keras.regularizers.l2(weight_decay) if weight_decay else None

def skip_connection(x, y,name=None):
    return tf.keras.layers.add([x, y])

def max_pooling_layer(x, name='max_pooling'):
    return tf.keras.layers.MaxPooling2D(pool_size=2, padding='same', name=name)(x)

def upsampling_layer(x, channels, name='upsampling_layer'):
    x = tf.keras.layers.UpSampling2D(name=name+'-upsampling')(x)
    # kernel_size=3 pour lisser le redimensionnement
    x = convolutional_layer(x, channels, kernel_size=3, name=name+'-upsampling_convolutional')
    return x

def UNet(input_height, input_width, N):
    """
    Définition du Modèle UNet adapté à N sorties
    AVEC skip connections pour préserver les détails fins
    """
    inputs = Input(shape=(input_height, input_width, 1))

    # === ENCODEUR (Downsampling) ===
    channels = 32
    c1 = convolution_block(inputs, channels, name="c1")
    p1 = max_pooling_layer(c1, name="p1")
    
    c2 = convolution_block(p1, channels*2, name="c2")
    p2 = max_pooling_layer(c2, name="p2")

    c3 = convolution_block(p2, channels*4, name="c3")
    p3 = max_pooling_layer(c3, name="p3")

    # Bridge layer
    bridge_layer = convolution_block(p3, channels*8, name="bridge_layer")

    # === DÉCODEUR (Upsampling) AVEC SKIP CONNECTIONS ===
    c4 = upsampling_layer(bridge_layer, channels*4, name="c4")
    c4_connected = skip_connection(c4, c3, name='skip4')  # ← IMPORTANT
    c4_convolution = convolution_block(c4_connected, channels*4, name="c4_convolution")

    c5 = upsampling_layer(c4_convolution, channels*2, name="c5")
    c5_connected = skip_connection(c5, c2, name='skip5')  # ← IMPORTANT
    c5_convolution = convolution_block(c5_connected, channels*2, name="c5_convolution")

    c6 = upsampling_layer(c5_convolution, channels, name="c6")
    c6_connected = skip_connection(c6, c1, name='skip6')  # ← IMPORTANT
    c6_convolution = convolution_block(c6_connected, channels, name="c6_convolution")

    # === COUCHE FINALE ===
    # ✅ CORRECTION CRITIQUE : activation='sigmoid' obligatoire avec BCE
    kernel_size = 1
    outputs = tf.keras.layers.Conv2D(N,
                                     kernel_size,
                                     activation='sigmoid',  # ← AJOUTÉ
                                     padding='same',
                                     kernel_regularizer=regularization(1.0e-4),
                                     kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
                                     bias_initializer='zeros',
                                     name="final")(c6_convolution)
    
    return Model(inputs=[inputs], outputs=[outputs])

#############################################################################################################################
class ImageSamplingCallback(tf.keras.callbacks.Callback):
    """
    Callback pour visualiser la prédiction du UNet sur une image fixe 
    de validation à la fin de chaque époque.
    """
    def __init__(self,
                 sample_input_image,
                 class_channel_index,
                 val_IDs,
                 nz,
                 composition,
                 output_dir="epoch_samples",
                 H=128,
                 W=128,
                 freq_img_save=1):
        
        self.sample_input = sample_input_image
        self.class_channel_index = class_channel_index 
        self.output_dir = output_dir
        self.H = H
        self.W = W
        self.val_IDs = val_IDs
        self.nz = nz
        self.composition = composition
        self.freq_img_save = freq_img_save
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def on_train_begin(self, logs=None):
        """Sauvegarde les masques de référence au début de l'entraînement"""
        for sp in self.composition:
            for j in range(self.nz):
                image_path = f"data/train/prob_maps/{self.val_IDs[0]}_{sp}_{j:04d}.png"
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️ Impossible de charger : {image_path}")
                    continue
                    
                if img.shape[0] != self.H or img.shape[1] != self.W:
                    img = cv2.resize(img, (self.W, self.H))

                filename = os.path.join(self.output_dir, f"REFERENCE_{self.val_IDs[0]}_{sp}_{j:04d}.png")
                cv2.imwrite(filename, img)
            
    def on_epoch_end(self, epoch, logs=None):
        """Génère les prédictions à la fin de chaque époque"""
        freq_img_save = self.freq_img_save
        
        # Charger l'image d'entrée
        image_path = f"data/train/images/{self.val_IDs[0]}.png"
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Impossible de charger : {image_path}")
            return

        # Redimensionnement
        if img.shape[0] != self.H or img.shape[1] != self.W:
            img = cv2.resize(img, (self.W, self.H))
        
        # Normalisation [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Correction des dimensions
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        
        # Prédiction (training=False pour désactiver Dropout)
        prediction = self.model(img, training=False).numpy()
        
        prediction_map = prediction[0]
        idx_channel = 0
        
        # Sauvegarder chaque canal
        for sp in self.composition:
            for j in range(self.nz):
                pred_channel = prediction_map[..., idx_channel]
                pred_img = (pred_channel * 255).astype(np.uint8)
                
                filename = os.path.join(
                    self.output_dir, 
                    f"EPOCH_{(epoch+1):06d}_CHANNEL{idx_channel:02d}_{sp}_{j:04d}.png"
                )
                cv2.imwrite(filename, pred_img)
                idx_channel += 1

        # Sauvegarde périodique du modèle
        if epoch % 10 == 0:
            self.model.save('unet_atomod_trained_last.keras')
                
###########################################################################################################################
class CustomDataGenerator(tf.keras.utils.Sequence):
    """
    Générateur de données pour l'entraînement du UNet
    AVEC VÉRIFICATIONS ET DIAGNOSTICS
    """
    def __init__(self,
                 list_IDs,
                 data_path,
                 target_size,
                 batch_size=32,
                 shuffle=True,
                 composition=['Pt'],
                 nz=10,
                 debug_mode=False,
                 **kwargs):

        super().__init__(**kwargs)
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.composition = composition
        self.nz = nz
        self.debug_mode = debug_mode
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y
        
    def __data_generation(self, list_IDs_temp):
        H, W = self.target_size
        
        # Initialisation
        X = np.empty((self.batch_size, H, W, 1), dtype=np.float32)
        Y = np.empty((self.batch_size, H, W, len(self.composition)*self.nz), dtype=np.float32)

        # Créer le dossier de vérification si nécessaire
        if self.debug_mode and not os.path.exists("data/check"):
            os.makedirs("data/check")
        
        # Générer les données
        for i, ID in enumerate(list_IDs_temp):
            # === PARTIE X (Image d'Entrée) ===
            img_path = os.path.join(self.data_path, 'images', f'{ID}.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise FileNotFoundError(f"❌ Image non trouvée : {img_path}")
            
            # Redimensionnement
            img = cv2.resize(img, (W, H))
            
            # ✅ NORMALISATION [0, 1]
            img = img.astype(np.float32) / 255.0
            
            X[i,] = np.expand_dims(img, axis=-1)
            
            # === PARTIE Y (Masques Empilés) ===
            all_masks = []
            for sp in self.composition:
                for j in range(self.nz):
                    mask_path = os.path.join(self.data_path, 'prob_maps', f'{ID}_{sp}_{j:04d}.png')
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if mask is None:
                        raise FileNotFoundError(f"❌ Masque non trouvé : {mask_path}")
                    
                    # Redimensionnement (INTER_NEAREST pour préserver les valeurs binaires)
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    # ✅ NORMALISATION [0, 1]
                    mask = mask.astype(np.float32) / 255.0
                    
                    # 🔍 DEBUG : Afficher les statistiques du premier masque
                    if self.debug_mode and i == 0 and len(all_masks) == 0:
                        print(f"\n📊 STATS MASQUE {sp}_{j:04d}:")
                        print(f"   Min: {mask.min():.4f}")
                        print(f"   Max: {mask.max():.4f}")
                        print(f"   Mean: {mask.mean():.4f}")
                        print(f"   Unique values: {np.unique(mask)[:10]}")
                        
                        # Sauvegarder pour inspection visuelle
                        debug_path = f"data/check/MASK_{sp}_{j:04d}_normalized.png"
                        cv2.imwrite(debug_path, (mask * 255).astype(np.uint8))
                    
                    all_masks.append(mask)

            Y[i,] = np.stack(all_masks, axis=-1)
        
        # 🔍 DEBUG : Statistiques globales du batch
        if self.debug_mode:
            print(f"\n📊 STATS BATCH:")
            print(f"   X - Min: {X.min():.4f}, Max: {X.max():.4f}, Mean: {X.mean():.4f}")
            print(f"   Y - Min: {Y.min():.4f}, Max: {Y.max():.4f}, Mean: {Y.mean():.4f}")
            print(f"   Y shape: {Y.shape}")

        return X, Y
