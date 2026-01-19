import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from ATOMOD.ATOMOD import CustomDataGenerator, UNet, ImageSamplingCallback

def weighted_bce_loss(y_true, y_pred):
    # Poids pour les pixels positifs (les atomes). 
    # Si les atomes sont rares, on augmente ce poids (ex: 10 ou 50)
    pos_weight = 10.0 
    
    # Évite log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Calcul de la perte pondérée
    loss = - (pos_weight * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

class ATOMODTrainer:
    """Entraîneur pour le modèle ATOMOD UNet."""
    
    def __init__(self, config=None):
        """
        Initialise le trainer avec une configuration.
        
        Args:
            config (dict, optional): Dictionnaire de configuration personnalisée
        """
        # Configuration par défaut
        self.config = {
            'batch_size': 32,
            'epochs': 200000,
            'height': 64,
            'width': 64,
            'composition': ['Rh', 'Ir'],
            'nz': 10,
            'n_train_images': 2024,  # Première moitié pour train
            'n_val_images': 2024,     # Deuxième moitié pour val
            'restart': True,
            'checkpoint_path': 'unet_atomod_trained_last.h5',
            'initial_epoch': 0,
            'learning_rate': 1e-4,
            'data_root': 'data/train',
            'output_dir': 'data/train/intermediate',
            'logs_dir': 'logs',
            'checkpoint_dir': 'checkpoints',
            'save_best_only': False,
            'early_stopping_patience': 2000,
            'checkpoint_freq': 100
        }
        
        # Mise à jour avec la config personnalisée si fournie
        if config:
            self.config.update(config)
        
        # Création des dossiers nécessaires
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['logs_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        self.model = None
        self.device = None
        self.strategy = None
    
    def setup_gpu(self):
        """Configure et détecte les GPU disponibles avec optimisations."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) détecté(s):")
            for gpu in gpus:
                print(f"   - {gpu}")
            self.device = "cuda"
            
            # OPTIMISATION 1: Croissance mémoire dynamique
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("   ✓ Memory growth activé")
            except RuntimeError as e:
                print(f"   ⚠️  Erreur lors de la configuration GPU: {e}")
            
            # OPTIMISATION 2: XLA (Accelerated Linear Algebra)
            # Compile les graphes TensorFlow en kernels optimisés
            # Gain typique: 10-30% sur GPU
            tf.config.optimizer.set_jit(True)
            print("   ✓ XLA JIT compilation activée")
            
            # OPTIMISATION 3: Parallelisme intra-op et inter-op
            # Pour mieux utiliser les CPU threads pendant les transferts GPU
            tf.config.threading.set_intra_op_parallelism_threads(8)
            tf.config.threading.set_inter_op_parallelism_threads(4)
            print("   ✓ Threading optimisé (intra=8, inter=4)")
            
        else:
            print("⚠️  Aucun GPU détecté, utilisation du CPU")
            self.device = "cpu"
        
        return self.device
    
    def create_data_generators(self):
        """Crée les générateurs de données pour l'entraînement et la validation."""
        batch_size = self.config['batch_size']
        n_train = self.config['n_train_images']
        n_val = self.config['n_val_images']
        
        # Génération des IDs pour l'entraînement (première moitié)
        # img_0000.png à img_2023.png
        train_IDs = [f'img_{i:04d}' for i in range(n_train)]
        
        # Génération des IDs pour la validation (deuxième moitié)
        # img_2024.png à img_4047.png
        val_IDs = [f'img_{i:04d}' for i in range(n_train, n_train + n_val)]
        
        print(f"📊 Entraînement: {len(train_IDs)} images (img_0000 à img_{n_train-1:04d})")
        print(f"📊 Validation: {len(val_IDs)} images (img_{n_train:04d} à img_{n_train+n_val-1:04d})")
        print(f"📦 Batch size: {batch_size}")
        print(f"🔄 Steps per epoch train: {len(train_IDs) // batch_size}")
        print(f"🔄 Steps per epoch val: {len(val_IDs) // batch_size}")
        
        # Générateur d'entraînement
        train_generator = CustomDataGenerator(
            train_IDs,
            self.config['data_root'],
            (self.config['height'], self.config['width']),
            batch_size,
            shuffle=True,
            composition=self.config['composition'],
            nz=self.config['nz']
        )
        
        # Générateur de validation
        val_generator = CustomDataGenerator(
            val_IDs,
            self.config['data_root'],
            (self.config['height'], self.config['width']),
            batch_size,
            shuffle=False,
            composition=self.config['composition'],
            nz=self.config['nz']
        )
        
        return train_generator, val_generator, val_IDs
    
    def build_model(self):
        """Construit ou charge le modèle UNet."""
        # Configuration multi-GPU
        self.strategy = tf.distribute.MirroredStrategy()
        print(f"🔧 Distribution sur {self.strategy.num_replicas_in_sync} GPU(s)")
        
        with self.strategy.scope():
            # Calcul du nombre de canaux de sortie
            output_channels = len(self.config['composition']) * self.config['nz']
            print(f"📐 Output channels: {output_channels} ({len(self.config['composition'])} espèces × {self.config['nz']} couches)")
            
            # Chargement ou création du modèle
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
            
            # Compilation du modèle
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config['learning_rate'],
                    clipnorm=1.0
                ),
                # Notez from_logits=False car on a ajouté l'activation Sigmoid dans le modèle
                # Si vous utilisez la weighted_bce_loss ci-dessus : loss=weighted_bce_loss
                # Sinon, pour commencer simple :
                #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                loss=weighted_bce_loss,
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
                #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                #metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
            )
        
        self.model.summary()
        return self.model
    
    def create_callbacks(self, val_generator, val_IDs):
        """Crée la liste des callbacks pour l'entraînement."""
        callbacks = []
        
        # CALLBACK 1: Sauvegarde régulière
        # Note: On ne peut pas utiliser {val_loss} avec save_freq car val_loss
        # n'est disponible qu'à la fin de l'epoch, pas pendant
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(
                self.config['checkpoint_dir'],
                'unet_epoch_{epoch:06d}.h5'
            ),
            save_best_only=False,
            save_freq=self.config['checkpoint_freq'],  # Tous les N epochs
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # CALLBACK 2: Sauvegarde du meilleur modèle séparément
        best_checkpoint = ModelCheckpoint(
            filepath='best_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        callbacks.append(best_checkpoint)
        
        # CALLBACK 3: TensorBoard optimisé pour long training
        tensorboard_callback = TensorBoard(
            log_dir=self.config['logs_dir'],
            histogram_freq=0,  # Désactivé pour performance
            write_graph=False,  # Graph déjà connu, pas besoin
            update_freq='epoch',
            profile_batch=0  # Pas de profiling (très lourd)
        )
        callbacks.append(tensorboard_callback)
        
        # CALLBACK 4: CSV Logger pour analyse facile
        csv_logger = CSVLogger(
            filename=os.path.join(self.config['logs_dir'], 'training_history.csv'),
            separator=',',
            append=True
        )
        callbacks.append(csv_logger)
        
        # CALLBACK 5: Reduce LR on Plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=500,  # Adapté pour long training
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # CALLBACK 6: Early stopping avec patience adaptée
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # CALLBACK 7: Visualisation personnalisée
        try:
            sample_batch_input, _ = val_generator[0]
            sample_input_for_callback = sample_batch_input[0:1]
            
            image_sampling_callback = ImageSamplingCallback(
                sample_input_image=sample_input_for_callback,
                class_channel_index=0,
                val_IDs=val_IDs[:10],  # Prend seulement les 10 premiers pour économiser du temps
                nz=self.config['nz'],
                composition=self.config['composition'],
                output_dir=self.config['output_dir'],
                H=self.config['height'],
                W=self.config['width']
            )
            callbacks.append(image_sampling_callback)
        except Exception as e:
            print(f"⚠️  Impossible de créer ImageSamplingCallback: {e}")
        
        return callbacks
    
    def evaluate_initial_performance(self, val_generator):
        """Évalue les performances initiales du modèle."""
        print("\n" + "="*50)
        print("📊 ÉVALUATION PRÉ-ENTRAÎNEMENT")
        print("="*50)
        
        try:
            initial_metrics = self.model.evaluate(
                val_generator,
                steps=min(5, len(val_generator)),
                verbose=1
            )
            print(f"✅ Métriques initiales: Loss={initial_metrics[0]:.4f}, "
                  f"Accuracy={initial_metrics[1]:.4f}")
        except Exception as e:
            print(f"⚠️  Erreur lors de l'évaluation initiale: {e}")
        
        print("="*50 + "\n")
    
    def train(self):
        """Lance l'entraînement complet du modèle."""
        print("\n" + "="*60)
        print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT ATOMOD")
        print("="*60 + "\n")
        
        # 1. Configuration GPU
        self.setup_gpu()
        
        # 2. Création des générateurs de données
        train_gen, val_gen, val_IDs = self.create_data_generators()
        
        # 3. Construction du modèle
        self.build_model()
        
        # 4. Évaluation initiale
        self.evaluate_initial_performance(val_gen)
        
        # 5. Création des callbacks
        callbacks = self.create_callbacks(val_gen, val_IDs)
        
        # 6. Entraînement
        print("🏋️  Début de l'entraînement...")
        print(f"   - Sauvegarde tous les {self.config['checkpoint_freq']} epochs")
        print(f"   - Early stopping après {self.config['early_stopping_patience']} epochs sans amélioration\n")
        
        history = self.model.fit(
            train_gen,
            steps_per_epoch=len(train_gen),
            epochs=self.config['epochs'],
            initial_epoch=self.config['initial_epoch'],
            validation_data=val_gen,
            validation_steps=len(val_gen),
            callbacks=callbacks,
            verbose=1
        )
        
        # 7. Sauvegarde finale
        final_model_path = 'unet_atomod_trained_final.h5'
        self.model.save(final_model_path)
        print(f"\n✅ Entraînement terminé!")
        print(f"💾 Modèle sauvegardé: {final_model_path}")
        print(f"📊 Historique CSV: {os.path.join(self.config['logs_dir'], 'training_history.csv')}")
        
        return history


def main():
    """Point d'entrée principal du programme."""
    
    # Configuration optimisée pour 4 GPU P100 (batch_size=64)
    config = {
        'batch_size': 64,  # 2024/64 = 31.6 → 31 steps par epoch (16 images/GPU)
        'epochs': 200000,
        'height': 128,
        'width': 128,
        'composition': ['Rh', 'Ir'],
        'nz': 10,
        'n_train_images': 2024,  # Première moitié
        'n_val_images': 2024,     # Deuxième moitié
        'restart': False,
        'checkpoint_path': 'unet_atomod_trained_last4.h5',
        'initial_epoch': 0,
        'learning_rate': 1e-4,
        'data_root': 'data/train',
        'output_dir': 'data4/train/intermediate',
        'logs_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'save_best_only': False,
        'early_stopping_patience': 2000,
        'checkpoint_freq': 100
    }
    
    # Alternative batch size=128 pour utilisation GPU maximale (si mémoire suffisante)
    # config_max_gpu = {
    #     'batch_size': 128,  # 2024/128 = 15.8 → 15 steps/epoch (32 images/GPU)
    #     'n_train_images': 2024,
    #     'n_val_images': 2024,
    #     ...
    # }
    # ATTENTION: Avec batch=128, risque d'OOM sur P100 (16GB). Commencer avec 64.
    
    # Création et lancement de l'entraîneur
    trainer = ATOMODTrainer(config=config)
    history = trainer.train()
    
    print("\n🎉 PROCESSUS TERMINÉ!")


if __name__ == "__main__":
    main()
