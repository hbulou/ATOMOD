import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import mixed_precision

# --- OPTIMISATION 1 : MIXED PRECISION POUR H100 ---
# Utilise float16 pour les calculs, float32 pour la stabilité.
# Sur H100, cela booste énormément les perfs.
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from ATOMOD.ATOMOD import UNet, ImageSamplingCallback 
# Note : On n'utilise plus CustomDataGenerator pour l'entraînement, mais tf.data

class ATOMODTrainer:
    def __init__(self, config=None):
        self.config = {
            'batch_size': 1024, # H100 aime les gros batchs
            'epochs': 200000,
            'height': 64,
            'width': 64,
            'composition': ['Rh', 'Ir'],
            'nz': 10,
            'restart': True,
            'checkpoint_path': 'unet_atomod_trained_last.h5',
            'initial_epoch': 0,
            'learning_rate': 1e-4,
            'data_root': 'data/train',
            'output_dir': 'data/train/intermediate',
            'logs_dir': 'logs',
            'save_best_only': True
        }
        if config:
            self.config.update(config)
        
        self.model = None
        self.strategy = None
        self.global_batch_size = None

    def setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) détecté(s)")
            self.strategy = tf.distribute.MirroredStrategy()
            # Ajustement du batch size global
            self.global_batch_size = self.config['batch_size'] * self.strategy.num_replicas_in_sync
            print(f"🚀 Global Batch Size: {self.global_batch_size}")
        else:
            self.strategy = tf.distribute.get_strategy() # Default (CPU)
            self.global_batch_size = self.config['batch_size']

    def load_image_tf(self, file_path):
        """
        Fonction de chargement optimisée TensorFlow (remplace __getitem__ du générateur).
        À ADAPTER : Cette fonction suppose que l'image est l'entrée ET la sortie (Autoencoder).
        Si vos labels sont différents, modifiez cette fonction.
        """
        # 1. Lecture du fichier
        img_raw = tf.io.read_file(file_path)
        # 2. Décodage (adaptez channels=1 ou 3 selon vos images)
        img = tf.io.decode_png(img_raw, channels=1) 
        # 3. Conversion float32 et normalisation
        img = tf.image.convert_image_dtype(img, tf.float32) # Convertit [0,255] -> [0.0, 1.0]
        # 4. Resize (au cas où)
        img = tf.image.resize(img, [self.config['height'], self.config['width']])
        
        # --- LOGIQUE DE CIBLE (LABEL) ---
        # Si votre réseau doit prédire autre chose que l'image d'entrée,
        # vous devez générer la cible ici. Pour l'instant, je suppose Input = Target.
        # Si vous avez besoin de générer des canaux spécifiques comme dans CustomDataGenerator,
        # vous devrez peut-être utiliser tf.py_function (un peu plus lent mais flexible).
        return img, img 

    def create_tf_dataset(self, file_patterns, is_training=True):
        """Crée un pipeline tf.data haute performance."""
        # Récupération de la liste des fichiers
        files = glob.glob(file_patterns)
        
        # Création du dataset de base
        dataset = tf.data.Dataset.from_tensor_slices(files)
        
        # Chargement et prétraitement parallèle (C++ multithreading)
        dataset = dataset.map(self.load_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
        
        # --- OPTIMISATION 2 : CACHING ---
        # Vos images sont petites (64x64). On garde TOUT en RAM après la 1ère epoch.
        # Plus aucune lecture disque après ça.
        dataset = dataset.cache()
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=10000)
        
        # Batching
        dataset = dataset.batch(self.global_batch_size)
        
        # --- OPTIMISATION 3 : PREFETCHING ---
        # Prépare le batch suivant pendant que le GPU travaille
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset, len(files)

    def build_model(self):
        print(f"🔧 Distribution sur {self.strategy.num_replicas_in_sync} GPU(s)")
        
        with self.strategy.scope():
            output_channels = len(self.config['composition']) * self.config['nz']
            
            if self.config['restart'] and os.path.exists(self.config['checkpoint_path']):
                print(f"📥 Chargement du modèle depuis {self.config['checkpoint_path']}")
                self.model = load_model(self.config['checkpoint_path'], compile=False)
            else:
                print("🆕 Création d'un nouveau modèle UNet")
                self.model = UNet(
                    self.config['height'],
                    self.config['width'],
                    output_channels
                )
            
            # Note: Avec mixed_precision, l'output doit être float32 pour la stabilité numérique
            # Assurez-vous que la dernière couche de UNet a dtype='float32' si vous utilisez softmax/sigmoid
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
            )
        return self.model

    def train(self):
        self.setup_gpu()
        
        # Chemins des données (Adaptez le pattern *.png selon votre structure)
        # On suppose ici que data_root contient toutes les images
        all_files_pattern = os.path.join(self.config['data_root'], "images", "*.png")
        
        # On divise manuellement train/val car tf.data ne le fait pas nativement comme Keras Gen
        all_files = glob.glob(all_files_pattern)
        val_split = int(len(all_files) * 0.1) # 10% validation par exemple
        train_files = all_files[val_split:]
        val_files = all_files[:val_split]
        
        print(f"📊 Entraînement: {len(train_files)} images")
        print(f"📊 Validation: {len(val_files)} images")

        # Création des datasets optimisés
        # Note : On passe la liste des fichiers directement
        train_ds, n_train = self.create_tf_dataset(os.path.join(self.config['data_root'], "images", "*.png"), is_training=True)
        val_ds, n_val = self.create_tf_dataset(os.path.join(self.config['data_root'], "images", "*.png"), is_training=False)
        # ATTENTION: Ci-dessus j'ai remis le glob pour l'exemple, mais idéalement passez les listes train_files/val_files
        # Pour faire simple avec votre structure actuelle, je recrée un dataset à partir de slices
        train_ds = tf.data.Dataset.from_tensor_slices(train_files)
        train_ds = train_ds.map(self.load_image_tf, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(10000).batch(self.global_batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices(val_files)
        val_ds = val_ds.map(self.load_image_tf, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(self.global_batch_size).prefetch(tf.data.AUTOTUNE)

        self.build_model()
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath='checkpoints/unet_epoch_{epoch:04d}.h5',
                save_best_only=self.config['save_best_only'],
                monitor='val_loss', mode='min', verbose=1
            ),
            TensorBoard(log_dir=self.config['logs_dir'], update_freq='epoch')
        ]
        
        # Pour ImageSamplingCallback, il faut extraire une image du dataset
        # C'est un peu plus délicat avec tf.data, on prend le premier batch
        try:
            for img_batch, _ in val_ds.take(1):
                sample_input = img_batch[0:1] # Garde dims (1, 64, 64, 1)
                # Note: val_IDs n'est plus pertinent ici, on passe des IDs fictifs ou on adapte le callback
                dummy_ids = [os.path.basename(f) for f in val_files[:1]]
                
                callbacks.append(ImageSamplingCallback(
                    sample_input_image=sample_input.numpy(), # Convertir en numpy pour le callback
                    class_channel_index=0,
                    val_IDs=dummy_ids,
                    nz=self.config['nz'],
                    composition=self.config['composition'],
                    output_dir=self.config['output_dir'],
                    H=self.config['height'], W=self.config['width']
                ))
        except Exception as e:
            print(f"⚠️ Callback visualisation désactivé: {e}")

        print("🏋️  Début de l'entraînement...")
        history = self.model.fit(
            train_ds,
            epochs=self.config['epochs'],
            initial_epoch=self.config['initial_epoch'],
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model.save('unet_atomod_trained_final.h5')
        return history

def main():
    # Config adaptée pour H100
    custom_config = {
        'batch_size': 256, # Par GPU -> Total = 1024 sur 4 GPU
        'epochs': 200000,
        'height': 64, 'width': 64,
        'composition': ['Rh', 'Ir'], 'nz': 10,
        'restart': False, # Pour test propre
        'checkpoint_path': 'unet_atomod_trained_last16.h5',
        'initial_epoch': 0,
        'learning_rate': 1e-4,
        'data_root': 'data/train',
        'output_dir': 'data16/train/intermediate',
        'logs_dir': 'logs16',
        'save_best_only': True
    }

    trainer = ATOMODTrainer(config=custom_config)
    trainer.train()

if __name__ == "__main__":
    main()
