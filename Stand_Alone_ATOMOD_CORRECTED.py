import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from ATOMOD.ATOMOD_CORRECTED import CustomDataGenerator, UNet, ImageSamplingCallback


# ========================================
# FONCTIONS DE LOSS AMÉLIORÉES
# ========================================

def weighted_bce_loss(y_true, y_pred):
    """
    Binary Cross Entropy pondérée
    ✅ CORRECTION : pos_weight réduit de 20 à 5 pour plus de stabilité
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # Poids réduit pour éviter l'instabilité
    pos_weight = 5.0  # Réduit de 20.0 à 5.0
    
    loss = - (pos_weight * y_true * tf.math.log(y_pred) + 
              (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss pour gérer le déséquilibre de classes
    Focus sur les exemples difficiles à classer
    
    Args:
        alpha: Balance entre classes positives/négatives (0.25 = 25% poids sur positives)
        gamma: Focus sur exemples difficiles (2.0 = standard)
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Cross entropy standard
    cross_entropy = - (y_true * tf.math.log(y_pred) + 
                       (1 - y_true) * tf.math.log(1 - y_pred))
    
    # Terme focal : réduit la perte sur les exemples bien classés
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_term = tf.pow(1 - p_t, gamma)
    
    # Poids alpha pour balance positif/négatif
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    loss = alpha_factor * focal_term * cross_entropy
    return tf.reduce_mean(loss)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss : excellente pour la segmentation
    Mesure le chevauchement entre prédiction et vérité terrain
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice


def combined_loss(y_true, y_pred):
    """
    Combinaison de Focal Loss et Dice Loss
    Recommandé pour la segmentation avec déséquilibre
    """
    focal = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
    dice = dice_loss(y_true, y_pred)
    return 0.7 * focal + 0.3 * dice


# ========================================
# CLASSE TRAINER
# ========================================

class ATOMODTrainer:
    """Entraîneur pour le modèle ATOMOD UNet - VERSION CORRIGÉE"""
    
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
            'n_train_images': 2048,
            'n_val_images': 2048,
            'restart': False,
            'checkpoint_path': 'unet_atomod_trained_last.keras',
            'initial_epoch': 0,
            'learning_rate': 1e-4,
            'data_root': 'data/train',
            'output_dir': 'model/intermediate',
            'logs_dir': 'model/logs',
            'checkpoint_dir': 'model/checkpoints',
            'save_best_only': False,
            'early_stopping_patience': 2000,
            'checkpoint_freq': 100,
            'debug_mode': False,  # ✅ AJOUTÉ
            'loss_function': 'combined'  # ✅ AJOUTÉ : 'weighted_bce', 'focal', 'dice', 'combined'
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
                print(f"   ⚠️ Erreur lors de la configuration GPU: {e}")
            
            # OPTIMISATION 2: XLA (Accelerated Linear Algebra)
            tf.config.optimizer.set_jit(True)
            print("   ✓ XLA JIT compilation activée")
            
            # OPTIMISATION 3: Parallelisme
            tf.config.threading.set_intra_op_parallelism_threads(8)
            tf.config.threading.set_inter_op_parallelism_threads(4)
            print("   ✓ Threading optimisé (intra=8, inter=4)")
            
        else:
            print("⚠️ Aucun GPU détecté, utilisation du CPU")
            self.device = "cpu"
        
        return self.device
    
    def create_data_generators(self):
        """Crée les générateurs de données pour l'entraînement et la validation."""
        batch_size = self.config['batch_size']
        n_train = self.config['n_train_images']
        n_val = self.config['n_val_images']
        
        # Génération des IDs
        train_IDs = [f'img_{i:04d}' for i in range(n_train)]
        val_IDs = [f'img_{i:04d}' for i in range(n_train, n_train + n_val)]
        
        print(f"\n📊 CONFIGURATION DES DONNÉES:")
        print(f"   Entraînement: {len(train_IDs)} images (img_0000 à img_{n_train-1:04d})")
        print(f"   Validation: {len(val_IDs)} images (img_{n_train:04d} à img_{n_train+n_val-1:04d})")
        print(f"   Batch size: {batch_size}")
        print(f"   Steps/epoch train: {len(train_IDs) // batch_size}")
        print(f"   Steps/epoch val: {len(val_IDs) // batch_size}")
        
        # ✅ Générateur avec debug_mode
        train_generator = CustomDataGenerator(
            train_IDs,
            self.config['data_root'],
            (self.config['height'], self.config['width']),
            batch_size,
            shuffle=True,
            composition=self.config['composition'],
            nz=self.config['nz'],
            debug_mode=self.config['debug_mode']  # ✅ AJOUTÉ
        )
        
        val_generator = CustomDataGenerator(
            val_IDs,
            self.config['data_root'],
            (self.config['height'], self.config['width']),
            batch_size,
            shuffle=False,
            composition=self.config['composition'],
            nz=self.config['nz'],
            debug_mode=False  # Pas de debug en validation
        )
        
        return train_generator, val_generator, val_IDs
    
    def build_model(self):
        """Construit ou charge le modèle UNet."""
        # Configuration multi-GPU
        self.strategy = tf.distribute.MirroredStrategy()
        print(f"\n🔧 CONFIGURATION MODÈLE:")
        print(f"   Distribution sur {self.strategy.num_replicas_in_sync} GPU(s)")
        
        with self.strategy.scope():
            # Calcul du nombre de canaux de sortie
            output_channels = len(self.config['composition']) * self.config['nz']
            print(f"   Output channels: {output_channels} ({len(self.config['composition'])} espèces × {self.config['nz']} couches)")
            
            # Chargement ou création du modèle
            if self.config['restart'] and os.path.exists(self.config['checkpoint_path']):
                print(f"   📥 Chargement du modèle depuis {self.config['checkpoint_path']}")
                self.model = load_model(self.config['checkpoint_path'], compile=False)
            else:
                print("   🆕 Création d'un nouveau modèle UNet (avec skip connections)")
                self.model = UNet(
                    self.config['height'],
                    self.config['width'],
                    output_channels
                )
            
            # ✅ Sélection de la fonction de loss
            loss_functions = {
                'weighted_bce': weighted_bce_loss,
                'focal': focal_loss,
                'dice': dice_loss,
                'combined': combined_loss
            }
            
            selected_loss = loss_functions.get(
                self.config['loss_function'], 
                combined_loss
            )
            
            print(f"   Loss function: {self.config['loss_function']}")
            
            # Compilation du modèle
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config['learning_rate'],
                    clipnorm=1.0
                ),
                loss=selected_loss,
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )
        
        print("\n📋 ARCHITECTURE DU MODÈLE:")
        self.model.summary()
        return self.model
    
    def create_callbacks(self, val_generator, val_IDs):
        """Crée la liste des callbacks pour l'entraînement."""
        callbacks = []
        
        # CALLBACK 1: Sauvegarde régulière
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(
                self.config['checkpoint_dir'],
                'unet_epoch_{epoch:06d}.keras'
            ),
            save_best_only=False,
            save_freq=self.config['checkpoint_freq'],
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # CALLBACK 2: Sauvegarde du meilleur modèle
        best_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.config['checkpoint_dir'], 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        callbacks.append(best_checkpoint)
        
        # CALLBACK 3: TensorBoard
        tensorboard_callback = TensorBoard(
            log_dir=self.config['logs_dir'],
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch',
            profile_batch=0
        )
        callbacks.append(tensorboard_callback)
        
        # CALLBACK 4: CSV Logger
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
            patience=500,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # CALLBACK 6: Early stopping
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
                val_IDs=val_IDs[:5],  # 5 images de validation
                nz=self.config['nz'],
                composition=self.config['composition'],
                output_dir=self.config['output_dir'],
                H=self.config['height'],
                W=self.config['width'],
                freq_img_save=10  # Sauvegarde tous les 10 epochs
            )
            callbacks.append(image_sampling_callback)
        except Exception as e:
            print(f"⚠️ Impossible de créer ImageSamplingCallback: {e}")
        
        return callbacks
    
    def evaluate_initial_performance(self, val_generator):
        """Évalue les performances initiales du modèle."""
        print("\n" + "="*60)
        print("📊 ÉVALUATION PRÉ-ENTRAÎNEMENT")
        print("="*60)
        
        try:
            # Test sur un petit échantillon
            sample_X, sample_Y = val_generator[0]
            
            # Statistiques des données
            print(f"\n📈 STATISTIQUES DES DONNÉES:")
            print(f"   X (entrée) - Min: {sample_X.min():.4f}, Max: {sample_X.max():.4f}, Mean: {sample_X.mean():.4f}")
            print(f"   Y (masque) - Min: {sample_Y.min():.4f}, Max: {sample_Y.max():.4f}, Mean: {sample_Y.mean():.4f}")
            print(f"   Shape X: {sample_X.shape}, Shape Y: {sample_Y.shape}")
            
            # Prédiction initiale
            pred = self.model.predict(sample_X[:1], verbose=0)
            print(f"\n🔮 PRÉDICTION INITIALE:")
            print(f"   Min: {pred.min():.4f}, Max: {pred.max():.4f}, Mean: {pred.mean():.4f}")
            print(f"   Shape: {pred.shape}")
            
            # Évaluation complète
            initial_metrics = self.model.evaluate(
                val_generator,
                steps=min(5, len(val_generator)),
                verbose=1
            )
            
            metric_names = ['Loss', 'Accuracy', 'Precision', 'Recall']
            print(f"\n✅ MÉTRIQUES INITIALES:")
            for name, value in zip(metric_names, initial_metrics):
                print(f"   {name}: {value:.4f}")
                
        except Exception as e:
            print(f"⚠️ Erreur lors de l'évaluation initiale: {e}")
        
        print("="*60 + "\n")
    
    def train(self):
        """Lance l'entraînement complet du modèle."""
        print("\n" + "="*60)
        print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT ATOMOD - VERSION CORRIGÉE")
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
        print("🏋️ DÉBUT DE L'ENTRAÎNEMENT...")
        print(f"   - Sauvegarde tous les {self.config['checkpoint_freq']} epochs")
        print(f"   - Early stopping après {self.config['early_stopping_patience']} epochs sans amélioration")
        print(f"   - Loss function: {self.config['loss_function']}\n")
        
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
        final_model_path = os.path.join(self.config['checkpoint_dir'], 'unet_atomod_trained_final.keras')
        self.model.save(final_model_path)
        
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ!")
        print(f"💾 Modèle sauvegardé: {final_model_path}")
        print(f"📊 Historique CSV: {os.path.join(self.config['logs_dir'], 'training_history.csv')}")
        
        return history


# ========================================
# FONCTION MAIN
# ========================================

def main():
    """Point d'entrée principal du programme."""
    
    # Configuration recommandée pour démarrer
    batch_size = 64  # Commencer avec 64, augmenter si GPU sous-utilisé
    save_dir = "model_corrected_64x64_" + str(batch_size)
    
    config = {
        'batch_size': batch_size,
        'epochs': 200000,
        'height': 64,
        'width': 64,
        'composition': ['Rh', 'Ir'],
        'nz': 10,
        'n_train_images': 2048,
        'n_val_images': 2048,
        'restart': False,
        'checkpoint_path': 'unet_atomod_trained.keras',
        'initial_epoch': 0,
        'learning_rate': 1e-4,
        'data_root': 'data/train',
        'output_dir': save_dir + '/intermediate',
        'logs_dir': save_dir + '/logs',
        'checkpoint_dir': save_dir + '/checkpoints',
        'save_best_only': False,
        'early_stopping_patience': 2000,
        'checkpoint_freq': 100,
        'debug_mode': True,  # ✅ ACTIVER pour le premier batch
        'loss_function': 'combined'  # ✅ 'combined' recommandé (focal + dice)
    }
    
    # ⚠️ IMPORTANT : Après le premier batch, désactiver debug_mode
    # config['debug_mode'] = False
    
    print("\n📝 CONFIGURATION:")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Loss function: {config['loss_function']}")
    print(f"   Debug mode: {config['debug_mode']}")
    print(f"   Save directory: {save_dir}")
    
    # Création et lancement de l'entraîneur
    trainer = ATOMODTrainer(config=config)
    history = trainer.train()
    
    print("\n🎉 PROCESSUS TERMINÉ!")


if __name__ == "__main__":
    main()
