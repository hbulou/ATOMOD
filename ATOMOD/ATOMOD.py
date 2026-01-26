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
def  convolutional_layer(x, channels, kernel_size=3, name='convolutional_layer'):

    #kernel_initializer="RandomNormal"
    #kernel_initializer="glorot_uniform" # avec tanh / sigmoid
    kernel_initializer="he_normal"      # ReLU / LeakyReLU
    
    #bias_initializer=tf.keras.initializers.Constant(0.1),
    bias_initializer="zeros"
    convolutional_layer = tf.keras.layers.Conv2D(channels,
                                                 kernel_size,
                                                 padding='same',
                                                 kernel_regularizer=regularization(1.0e-4),
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 name=name)

    x = activation_function(convolutional_layer(x),name=name)
    
    #x = tf.keras.layers.BatchNormalization(name=name+'_batch_normalization')(x)

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
#def regularization():
#    weight_decay=True
#    if weight_decay:
#        return tf.keras.regularizers.l2(weight_decay)
#    else:
#        return None

def regularization(weight_decay=1e-4):
    return tf.keras.regularizers.l2(weight_decay) if weight_decay else None

def skip_connection(x, y,name=None):
    return tf.keras.layers.add([x, y])

def max_pooling_layer(x, name='max_pooling'):
    return tf.keras.layers.MaxPooling2D(pool_size=2, padding='same', name=name)(x)

#def upsampling_layer(x, channels, name='upsampling_layer'):
#    x = tf.keras.layers.UpSampling2D(name=name+'-upsampling')(x)
#    x = convolutional_layer(x, channels, kernel_size=1, name=name+'-upsampling_convolutional')
#    return (x)
def upsampling_layer(x, channels, name='upsampling_layer'):
    x = tf.keras.layers.UpSampling2D(name=name+'-upsampling')(x)
    # CHANGEMENT ICI : kernel_size passe de 1 à 3 pour lisser le redimensionnement
    x = convolutional_layer(x, channels, kernel_size=3, name=name+'-upsampling_convolutional')
    return x
def UNetv2(input_height,input_width,N):
    """
    Définition du Modèle UNet adapté à N sorties
    """
    # La taille du lot (Batch) est implicite ici, donc la forme est (H, W, Canaux)
    # L'entrée est une image en niveaux de gris : (H, W, 1)
    inputs = Input(shape=(input_height, input_width, 1))

    # === ENCODEUR (Downsampling) ===
    # Contraction Block 1
    # convolution layer(channels,kernel_size,...)
    # channels=32, 64, 128  --> nombre de filtres. Il détermine la profondeur de la sortie. Si on choisit 32 filtres, la sortie apprendra 32 motifs
    # différents et produira 32 cartes de caractéristiques en sortie. Plus ce nombre est élevé, plus le réseau peut apprendre de caractéristiques
    # complexes, mais en retour le calcul est lourd.
    # kernel_size=(3,3), (5,5). C'est la taille de la fenètre glissante qui se déplace sur l'image.

    channels=32
    c1 = convolution_block(inputs, channels, name="c1")
    p1 = max_pooling_layer(c1, name="p1")
    
    # Contraction Block 2
     # block 2
    c2 = convolution_block(p1, channels*2, name="c2")
    p2 = max_pooling_layer(c2, name="p2")

    # ... (Ajoutez plus de blocs c3, c4, p3, p4 ici pour un modèle plus profond)
    # block 3
    c3 = convolution_block(p2, channels*4, name="c3")
    p3 = max_pooling_layer(c3, name="p3")

    # bridge layer between the encoder and the decoder
    bridge_layer = convolution_block(p3, channels*8, name="bridge_layer")

    c4 = upsampling_layer(bridge_layer,              channels*4, name="c4")
    c4_convolution = convolution_block(c4, channels*4, name="c4_convolution")

    c5 = upsampling_layer(c4_convolution           , channels*2, name="c5")
    c5_convolution = convolution_block(c5, channels*2, name="c5_convolution")

    c6 = upsampling_layer(c5_convolution,              channels, name="c6")
    c6_convolution = convolution_block(c6, channels, name="c6_convolution")

    kernel_size=1
    #bias_initializer=tf.keras.initializers.Constant(0.1),
    outputs = tf.keras.layers.Conv2D(N,
                                     kernel_size,
                                     activation='sigmoid',
                                     padding='same',
                                     kernel_regularizer=regularization(1.0e-4),
                                     kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
                                     bias_initializer='zeros',
                                     name="final")(c6_convolution)
    
    #final_activation = tf.keras.layers.Activation('relu', dtype='float32')
    #outputs=final_activation(final_convolution(c6_convolution))
    #outputs=final_convolution(c6_convolution)

    
    return  Model(inputs=[inputs], outputs=[outputs])
def UNet(input_height,input_width,N):
    """
    Définition du Modèle UNet adapté à N sorties
    """
    # La taille du lot (Batch) est implicite ici, donc la forme est (H, W, Canaux)
    # L'entrée est une image en niveaux de gris : (H, W, 1)
    inputs = Input(shape=(input_height, input_width, 1))

    # === ENCODEUR (Downsampling) ===
    # Contraction Block 1
    # convolution layer(channels,kernel_size,...)
    # channels=32, 64, 128  --> nombre de filtres. Il détermine la profondeur de la sortie. Si on choisit 32 filtres, la sortie apprendra 32 motifs
    # différents et produira 32 cartes de caractéristiques en sortie. Plus ce nombre est élevé, plus le réseau peut apprendre de caractéristiques
    # complexes, mais en retour le calcul est lourd.
    # kernel_size=(3,3), (5,5). C'est la taille de la fenètre glissante qui se déplace sur l'image.

    channels=32
    c1 = convolution_block(inputs, channels, name="c1")
    p1 = max_pooling_layer(c1, name="p1")
    
    # Contraction Block 2
     # block 2
    c2 = convolution_block(p1, channels*2, name="c2")
    p2 = max_pooling_layer(c2, name="p2")

    # ... (Ajoutez plus de blocs c3, c4, p3, p4 ici pour un modèle plus profond)
    # block 3
    c3 = convolution_block(p2, channels*4, name="c3")
    p3 = max_pooling_layer(c3, name="p3")

    # bridge layer between the encoder and the decoder
    bridge_layer = convolution_block(p3, channels*8, name="bridge_layer")

    c4 = upsampling_layer(bridge_layer,              channels*4, name="c4")
    c4_connected = skip_connection(c4, c3,                       name='skip4')
    c4_convolution = convolution_block(c4_connected, channels*4, name="c4_convolution")

    c5 = upsampling_layer(c4_convolution           , channels*2, name="c5")
    c5_connected = skip_connection(c5, c2,                       name='skip5')
    c5_convolution = convolution_block(c5_connected, channels*2, name="c5_convolution")

    c6 = upsampling_layer(c5_convolution,              channels, name="c6")
    c6_connected = skip_connection(c6, c1,                     name='skip6')
    c6_convolution = convolution_block(c6_connected, channels, name="c6_convolution")

    kernel_size=1
    #bias_initializer=tf.keras.initializers.Constant(0.1),
    outputs = tf.keras.layers.Conv2D(N,
                                     kernel_size,
                                     activation='sigmoid',
                                     padding='same',
                                     kernel_regularizer=regularization(1.0e-4),
                                     kernel_initializer='he_normal',
                                     bias_initializer='zeros',
                                     name="final")(c6_convolution)
    
    #final_activation = tf.keras.layers.Activation('relu', dtype='float32')
    #outputs=final_activation(final_convolution(c6_convolution))
    #outputs=final_convolution(c6_convolution)

    
    return  Model(inputs=[inputs], outputs=[outputs])

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

        
        # L'image d'entrée doit être dans le format (1, H, W, C)
        self.sample_input = sample_input_image
        # L'indice du canal (classe) que vous souhaitez visualiser (e.g., Atome A = 0)
        self.class_channel_index = class_channel_index 
        self.output_dir = output_dir
        self.H = H
        self.W = W
        self.val_IDs=val_IDs
        self.nz=nz
        self.composition=composition
        self.freq_img_save=freq_img_save
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def on_train_begin(self,logs=None):
        for sp in self.composition:
            for j in range(self.nz):
                image_path=f"data/train/prob_maps/{self.val_IDs[0]}_{sp}_{j:04d}.png"
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")
                #img = img.astype(np.float32) / 255.0
                if img.shape[0] != self.H or img.shape[1] != self.W:
                    img = cv2.resize(img, (self.W, self.H))

                filename = os.path.join(self.output_dir, f"{self.val_IDs[0]}_{sp}_{j:04d}.png")
                cv2.imwrite(filename, img)

            
    def on_epoch_end(self, epoch, logs=None):
        freq_img_save=self.freq_img_save

        
        print(self.val_IDs[0])
        image_path=f"data/train/images/{self.val_IDs[0]}.png"
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")
        #img = img.astype(np.float32) / 255.0

        # --- REDIMENSIONNEMENT ---
        if img.shape[0] != self.H or img.shape[1] != self.W:
            img = cv2.resize(img, (self.W, self.H))

        #filename = os.path.join(self.output_dir, f"epoch_{epoch+1:03d}_{img.min()}_{img.max()}.png")
        #filename = os.path.join(self.output_dir, f"INPUT_{self.val_IDs[0]}.png")
        #cv2.imwrite(filename, img)
        
        # --- NORMALISATION (Important pour les réseaux de neurones) ---
        # On passe de [0, 255] (entiers) à [0.0, 1.0] (flottants)
        img = img.astype(np.float32) / 255.0
        
        # --- CORRECTION DES DIMENSIONS (SHAPE) ---
        # 1. Si l'image est en 2D (64, 64), on ajoute le canal -> (64, 64, 1)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            
        # 2. On ajoute la dimension du Batch -> (1, 64, 64, 1)
        img = np.expand_dims(img, axis=0)

        print(f"Debug: Shape envoyée au predict: {img.shape}") # Doit afficher (1, 64, 64, 1)
        #img=np.expand_dims(img,axis=-1)
        #img=np.expand_dims(img,axis=0)
        
        # Appel direct (call) au lieu de predict. C'est plus rapide pour 1 image et évite le bug de distribution.
        # training=False est important pour désactiver le Dropout/BatchNorm si vous en avez.
        prediction = self.model(img, training=False).numpy()


        
        #prediction = self.model.predict(img)
        prediction_map=prediction[0]
        idx_channel=0
        for sp in self.composition:
            for j in range(self.nz):
                img=prediction_map[...,idx_channel]
                img=  (img * 255).astype(np.uint8)
                filename = os.path.join(self.output_dir, f"EPOCH_{(epoch+1)%freq_img_save:03d}_CHANNEL{idx_channel}_{sp}_{j}.png")
                #cv2.imwrite(filename, seg_visualization)
                cv2.imwrite(filename, img)

                idx_channel+=1

        if epoch%10 == 0:
            self.model.save('unet_atomod_trained_last.h5')
                
        # # 1. Effectuer la prédiction
        # # self.model est automatiquement disponible dans le callback après l'initialisation
        # prediction = self.model.predict(self.sample_input, verbose=0)
        # print(f"len(prediction)={len(prediction)}")
        # print(prediction.min(), prediction.max(), prediction.mean())
        # # Le résultat est de forme (1, H, W, N_classes). On prend l'échantillon 0
        # prediction_map = prediction[0] 
        
        # # 2. Extraire la classe d'intérêt (e.g., Atome A, canal 0)
        # img = prediction_map[..., self.class_channel_index]
        
        # # 3. Créer la carte de segmentation binaire (Seuillage à 0.5)
        # # On multiplie par 255 pour la visualisation (0=Noir, 255=Blanc)
        # #seg_visualization = (img > 0.5).astype(np.uint8) * 255
        # img=  (img * 255).astype(np.uint8)

        
        # # 4. Sauvegarder l'image
        # filename = os.path.join(self.output_dir, f"epoch_{epoch+1:03d}_{img.min()}_{img.max()}.png")
        # #cv2.imwrite(filename, seg_visualization)
        # cv2.imwrite(filename, img)
        
        # print(f"\n[Callback] Image de segmentation sauvegardée pour l'époque {epoch+1}.")
        
###########################################################################################################################
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 list_IDs,
                 data_path,
                 target_size,
                 batch_size=32,
                 shuffle=True,
                 composition=['Pt'],
                 nz=10,
                 **kwargs):

        super().__init__(**kwargs)
        self.list_IDs = list_IDs       # Liste des noms de fichiers de base (ex: ['img_001', 'img_002', ...])
        self.data_path = data_path     # Chemin racine vers les dossiers 'images' et 'masks'
        self.target_size = target_size # (H, W)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.composition=composition
        self.nz=nz
    def __len__(self):
        # Nombre de lots par époque
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        # Mettre à jour l'ordre des indices après chaque époque si shuffle=True
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Générer un lot d'indices. Fabriquer un "batch" (un lot) de données.
        # Lorsqu'on lance model.fit() dans Stand_Alone_ATOMOD.py, Keras ne charge pas
        # toutes les données d'un coup. Il appelle ls générateur en boucle :
        #   Keras : "J'ai besoin du batch numéro 0."
        #   la méthode __getitem__(0) s'exécute.
        #   Keras : "J'ai besoin du batch numéro 1."
        #   ls méthode __getitem__(1) s'exécute.

        # identification des fichiers à charger
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Générer les données pour ce lot
        # Charge et prépare les données (X=images, Y=labels)
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y
    def __data_generation(self, list_IDs_temp):
        H, W = self.target_size
        
        
        # Initialisation des tableaux de lots
        # X: (Batch, H, W, 1) ; Y: (Batch, H, W, 10)
        X = np.empty((self.batch_size, H, W, 1), dtype=np.float32)
        Y = np.empty((self.batch_size, H, W, len(self.composition)*self.nz), dtype=np.float32)

        if not os.path.exists("data/check"):
            os.makedirs("data/check")

        
        # Générer les données
        for i, ID in enumerate(list_IDs_temp):

            # --- Partie X (Image d'Entrée - Niveaux de Gris) ---
            img_path = os.path.join(self.data_path, 'images', f'{ID}.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Charge en (H, W)
            
            # Prétraitement de X
            img = cv2.resize(img, (W, H))
            #cv2.imwrite(f"data/check/X{ID}_{img.min()}_{img.max()}.png", img)

            img = img / 255.0  # Normalisation
            
            # Ajouter la dimension du canal (1) et du lot (sera ajouté par Keras)
            X[i,] = np.expand_dims(img, axis=-1) # Forme (H, W, 1)
            
            # --- Partie Y (10 Masques Empilés - Binaires) ---
            all_masks = []
            for sp in self.composition:
                for j in range(self.nz):
                    mask_path = os.path.join(self.data_path, 'prob_maps', f'{ID}_{sp}_{j:04d}.png')

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    #_info_img(mask)
                    #_display_img(mask,title=f"{mask_path}")
                    #exit()
                
                    # Prétraitement de Y
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(f"data/check/Y{sp}_{j}_{mask.min()}_{mask.max()}.png", mask)
                    mask = mask / 255.0 # Normalisation binaire (0 ou 1)
                    all_masks.append(mask)

            # Empiler les 10 masques dans l'axe des canaux
            Y[i,] = np.stack(all_masks, axis=-1) # Forme (H, W, 10)

        return X, Y
