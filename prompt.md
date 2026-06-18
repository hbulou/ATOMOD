---
type: Type
---
# Prompt

Peux tu me donner le code de train_generator, qui à chaque itération

- lit BATCH_SIZE images TEM (une par nanoparticules),
- lit BATCH_SIZE×N_ESPECES spectres EXAFS,
- lit BATCH_SIZE×N_ESPECES volumes 3D des probabilités de présence atomique cibles,

puis assemble toutes ces données dans des tableaux NumPy et le envoie au GPU. A chaque itération, train_generator doit fournir un couple strict de données sous la forme ([batch_tem, batch_exafs], batch_volume_target ) :

- batch_tem : Les BATCH_SIZE images TEM de forme (BATCH_SIZE, WIDTH_HEIGHT,HEIGHT_WIDTH, 1),
- batch_exafs : Les BATCH_SIZE matrices de spectres normalisés de forme (BATCH_SIZE, N_points, N_ESPECES),
- batch_volume_target : Les BATCH_SIZE volumes 3D cibles de forme (BATCH_SIZE, WITH_IMAGE, HEIGHT_IMAGE, N_ESPECE * N_Z_PLANS).
