import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from ATOMOD.ATOMOD import CustomDataGenerator, UNet, ImageSamplingCallback


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
            'batch_size': 64,
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
        
        # Mise à jour avec la config personnalisée si fournie
        if config:
            self.config.update(config)
        
        self.model = None
        self.device = None
        self.strategy = None
    
    def setup_gpu(self):
        """Configure et détecte les GPU disponibles."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) détecté(s):")
            for gpu in gpus:
                print(f"   - {gpu}")
            self.device = "cuda"
            
            # Configuration de la croissance mémoire pour éviter les OOM
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"⚠️  Erreur lors de la configuration GPU: {e}")
        else:
            print("⚠️  Aucun GPU détecté, utilisation du CPU")
            self.device = "cpu"
        
        return self.device
    
    def create_data_generators(self):
        """Crée les générateurs de données pour l'entraînement et la validation."""
        batch_size = self.config['batch_size']
        
        # Génération des IDs pour l'entraînement
        train_IDs = [f'img_{i+1:04d}' for i in range(batch_size)]
        
        # Génération des IDs pour la validation
        val_IDs = [f'img_{i+1:04d}' for i in range(batch_size, 2 * batch_size)]
        
        print(f"📊 Entraînement: {len(train_IDs)} images")
        print(f"📊 Validation: {len(val_IDs)} images")
        
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
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
            )
        
        self.model.summary()
        return self.model
    
    def create_callbacks(self, val_generator, val_IDs):
        """Crée la liste des callbacks pour l'entraînement."""
        callbacks = []
        
        # Callback de sauvegarde des meilleurs modèles
        checkpoint_callback = ModelCheckpoint(
            filepath='checkpoints/unet_epoch_{epoch:04d}.h5',
            save_best_only=self.config['save_best_only'],
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Callback TensorBoard
        tensorboard_callback = TensorBoard(
            log_dir=self.config['logs_dir'],
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Callback de visualisation personnalisé
        try:
            sample_batch_input, _ = val_generator[0]
            sample_input_for_callback = sample_batch_input[0:1]
            
            image_sampling_callback = ImageSamplingCallback(
                sample_input_image=sample_input_for_callback,
                class_channel_index=0,
                val_IDs=val_IDs,
                nz=self.config['nz'],
                composition=self.config['composition'],
                output_dir=self.config['output_dir'],
                H=self.config['height'],
                W=self.config['width']
            )
            callbacks.append(image_sampling_callback)
        except Exception as e:
            print(f"⚠️  Impossible de créer ImageSamplingCallback: {e}")
        
        # Early stopping (optionnel)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
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
        
        return history


def main():
    """Point d'entrée principal du programme."""
    # Configuration personnalisée (optionnel)
    custom_config = {
        'composition': ['Rh', 'Ir'],
        'batch_size': 64,
        'epochs': 200000,
        'learning_rate': 1e-4
    }
    
    # Création et lancement de l'entraîneur
    trainer = ATOMODTrainer(config=custom_config)
    history = trainer.train()
    
    print("\n🎉 PROCESSUS TERMINÉ!")


if __name__ == "__main__":
    main()
