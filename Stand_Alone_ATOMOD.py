# On traverse le dossier ATOMOD, on ouvre le fichier ATOMOD, et on importe la fonction
#from ATOMOD.data_generation import mk_in_silico_data
from ATOMOD.MachineLearning import train
from ATOMOD.data_generation import mk_in_silico_data

def main():
    mk_in_silico_data()
    #train()
# #########################################################################################
if __name__ == "__main__":
    main()


    # J'ai une méthode mk_in_silico_data() définie dans un fichier ATOMOD.py qui est dans le répertoire /home/bulou/src/ATOMOD/ATOMOD.  je veux appeler cette méthode dans le script Stand_Alone_ATOMOD.py qui est dans le répertoire /home/bulou/src/ATOMOD. Que me conseilles tu ?
