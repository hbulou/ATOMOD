import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from ATOMOD.ATOMOD import CustomDataGenerator,UNet,ImageSamplingCallback
class NN:
    def __init__(self):
        self.batch_size=16
        self.epochs=200000
        #self.H=256
        #self.W=256
        self.H=64
        self.W=64
        self.composition=['Fe','Pt']
        self.nz=10
        self.restart=False
        self.ATOMOD_Training_starting_model="unet_atomod_trained_last.h5"
        self.initial_epochs=1
        self.device=None
    def ATOMOD_training(self):
        print("ATOMOD_training")
        #self.image_paths = sorted(glob(os.path.join(image_dir, "*")))
        #if len(self.image_paths) == 0:
        #    raise ValueError(f"Aucune image trouvée dans {image_dir}")
        device=self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Utilisation de l'appareil :", device)
        
                    

        # # Liste fictive des IDs d'images
        # # En réalité, ceci serait généré par os.listdir(os.path.join(DATA_ROOT, 'images'))

        BATCH_SIZE=self.batch_size
        EPOCHS=self.epochs
        

        # sélection des images/masks pour l'entrainement
        train_IDs=[]
        for i in range(BATCH_SIZE):
            train_IDs.append(f'img_{i+1:04d}')

        # sélection des images/masks pours la validation
        val_IDs=[]
        for i in range(BATCH_SIZE,2*BATCH_SIZE):
            val_IDs.append(f'img_{i+1:04d}')
        print(train_IDs)
        print(val_IDs)
        #for sp in self.composition:
        #    f"{sp}_{i:04d}_{k:04d}.png"

        DATA_ROOT="data/train"

        # # --- 1. Initialisation des Générateurs ---
        train_generator = CustomDataGenerator(
             train_IDs, 
             DATA_ROOT, 
             (self.H, self.W), 
             BATCH_SIZE, 
             shuffle=True,
            composition=self.composition,
            nz=self.nz
         )
        
        validation_generator = CustomDataGenerator(
            val_IDs, 
            DATA_ROOT, 
            (self.H, self.W), 
            BATCH_SIZE, 
            shuffle=False,
            composition=self.composition,
            nz=self.nz
        )

        print("🆕 Création d'un nouveau modèle")
        self.model=UNet(self.H,self.W,len(self.composition)*self.nz)
                # --- 3. Compilation du Modèle ---
        if self.restart:
            model_path = self.ATOMOD_Training_starting_model
            if os.path.exists(model_path):
                print("🔄 Reprise de l'entraînement depuis", model_path)
                self.model = load_model(model_path, compile=False)

                
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
            # Perte pour N classifications binaires indépendantes (H, W, N)
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)],
            run_eagerly=True
        )
        initial_epochs=self.initial_epochs
        self.model.summary() # Décommenter pour voir l'architecture et la forme de sortie (None, H, W, 10)

        # --- 4. Entraînement ---
        print("Démarrage de l'entraînement...")

        # --- Préparation du Callback pour la Visualisation ---
        # 1. Extraire un échantillon du générateur de validation
        # On récupère le premier lot (batch) du générateur de validation
        # Assurez-vous que votre CustomDataGenerator supporte l'indexation ([0]) ou utilisez next(iter(validation_generator))
        try:
            sample_batch_input, _ = validation_generator[0] 
        except Exception:
            # Si l'indexation n'est pas supportée, utilisez un itérateur :
            sample_batch_input, _ = next(iter(validation_generator))

        # 2. Sélectionner la première image du lot (Batch=1 pour la prédiction)
        # La forme doit être (1, H, W, C) pour la prédiction UNet

        intermediate_dir="data/train/intermediate"

        
        sample_input_for_callback = sample_batch_input[0:1]
        callbacks_list=[]
        callbacks_list.append(
            ImageSamplingCallback(
                sample_input_image=sample_input_for_callback, 
                class_channel_index=0,
                val_IDs=val_IDs,
                nz=self.nz,
                composition=self.composition,
                output_dir=intermediate_dir,
                H=self.H, 
                W=self.W)
        )
            
        #        Ce qui se passe à l’intérieur de fit#
        #
        #Pour chaque époque :
        #
        #for epoch in epochs:
        #    for batch in data:
        #        y_pred = model(x_batch, training=True)
        #        loss = loss_fn(y_batch, y_pred)
        #        gradients = tape.gradient(loss, weights)
        #        optimizer.apply_gradients(...)
        #        update_metrics()
        #    run_validation()
        #    callbacks.on_epoch_end()
        #
        # fit() encapsule tout cela automatiquement.

        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=EPOCHS,
            initial_epoch=initial_epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks_list
        )

        print("Entraînement terminé.")
        self.model.save('unet_atomod_trained.h5')
        print("Modèle sauvegardé sous 'unet_atomod_trained.h5'")
        
        # Ce fichier contient :
        #    l’architecture
        #    les poids
        #    l’optimizer avec son état interne (Adam, etc.)
        
        print("DONE!")

# ##########################################################################################
# Point d’entrée du programme
if __name__ == "__main__":
    FePt=NN()
    FePt.ATOMOD_training()
