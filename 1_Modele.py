# **************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 2 
# GBEGAN HUGUES 
# GBEH24279505
# ===========================================================================

# #===========================================================================
# Ce modèle est un classifieur (un CNN) entrainé sur l'ensemble de données MNIST afin de distinguer entre les images des chiffres 2 et 7.
# MNIST est une base de données contenant des chiffres entre 0 et 9 Ècrits à la main en noire et blanc de taille 28x28 pixels
# Pour des fins d'illustration, nous avons pris seulement deux chiffres 2 et 7
#
# Données:
# ------------------------------------------------
# entrainement : classe '2': 4 000 images | classe '7': images 4 000 images
# validation   : classe '2': 1 000 images | classe '7': images 1 000 images
# test         : classe '2': 1 000 images | classe '7': images 1 000 images 
# ------------------------------------------------

#>>> Ce code fonctionne sur MNIST. 
#>>> Vous devez donc intervenir sur ce code afin de l'adapter aux données du TP. 
#>>> À cette fin repérer les section QUESTION et insérer votre code et modification à ces endroits

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
import time
from keras.preprocessing.image import ImageDataGenerator

# Le Type de notre modéle (séquentiel)
from keras.models import Model
from keras.models import Sequential

# Le type d'optimisateur utilisé dans notre modèle (RMSprop, adam, sgd, adaboost ...)
# L'optimisateur ajuste les poids de notre modèle par descente du gradient
# Chaque optimisateur a ses propres paramètres
# Note: Il faut tester plusieurs et ajuster les paramètres afin d'avoir les meilleurs résultats
from keras.optimizers import Adam

# Les types des couches utlilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, Dense, LeakyReLU

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Sauvegarde du modèle
from keras.models import load_model

# Affichage des graphes 
import matplotlib.pyplot as plt

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);

# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - training_batch_size
# - validation_batch_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les données
mainDataPath = "donnees/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"


import os
# Le dossier contenant les images de validation
validationPath = mainDataPath + "validation"
if not os.path.exists(validationPath):
    os.makedirs(validationPath)
    for i in os.listdir(trainPath):
            os.makedirs(validationPath  + "/" + i)
            for j in os.listdir(trainPath + "/" +  i):
                if int(j.split(".")[0]) > 3000:
                    os.rename(trainPath + "/" + i + "/" + j, validationPath + "/" + i + "/" + j)
else:
    print("Validation folder already exists")


# Le dossier contenant les images de test
testPath = mainDataPath + "test"

# Le nom du fichier du modèle à sauvegarder
modelsPath = "Model.hdf5"


# Le nombre d'images d'entrainement et de validation
# Il faut en premier lieu identifier les paramètres du CNN qui permettent d’arriver à des bons résultats. À cette fin, la démarche générale consiste à utiliser une partie des données d’entrainement et valider les résultats avec les données de validation. Les paramètres du réseaux (nombre de couches de convolutions, de pooling, nombre de filtres, etc) devrait etre ajustés en conséquence.  Ce processus devrait se répéter jusqu’au l’obtention d’une configuration (architecture) satisfaisante. 
# Si on utilise l’ensemble de données d’entrainement en entier, le processus va être long car on devrait ajuster les paramètres et reprendre le processus sur tout l’ensemble des données d’entrainement.

training_batch_size = 3000  # total 18000 (3000 pour chaque classe)
validation_batch_size = 1000  # total 6000 (1000 pour chaque classe)

# Configuration des  images 
image_scale = 224 # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs 
image_shape = (image_scale, image_scale, image_channels) # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 32 # le nombre d'images entrainées ensemble: un batch
fit_epochs = 50 # Le nombre d'époques 

# ==========================================
# ==================MODÈLE==================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS DU TP
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajuster les deux fonctions:
# 2) feature_extraction
# 3) fully_connected
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)

# Partie feature extraction (ou cascade de couches d'extraction des caractéristiques)
def feature_extraction(input):
    # 1-couche de convolution avec nombre de filtre  (exp 32)  avec la taille de la fenetre de ballaiage exp : 3x3 
    # 2-fonction d'activation exp: sigmoid, relu, tanh ...
    # 3-couche d'echantillonage (pooling) pour reduire la taille avec la taille de la fenetre de ballaiage exp :2x2  

    # **** On répète ces étapes tant que nécessaire ****
    x = Conv2D(32, (3, 3), padding='same')(input) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Dropout(0.2)(x) 

    x = Conv2D(64, (3, 3), padding='same')(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Dropout(0.2)(x) 

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x) 
    # # x = Dropout(0.3)(x) 

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # # x = Dropout(0.3)(x)  

    # x = Conv2D(512, (3, 3), padding='same')(x)
    # x = Activation("relu")(x)
    # x = Dropout(0.4)(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # L'ensemble des features/caractéristiques extraits
    
    return encoded


# Partie complètement connectée (Fully Connected Layer)
def fully_connected(encoded):
    # Flatten: pour convertir les matrices en vecteurs pour la couche MLP
    # Dense: une couche neuronale simple avec le nombre de neurone (exemple 64)
    # fonction d'activation exp: sigmoid, relu, tanh ...
    # Dropout: pour éviter le sur-apprentissage

    x = Flatten(input_shape=image_shape)(encoded)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x) 

    x = Dense(64)(x)
    x = Activation("relu")(x)

    # La dernière couche est une couche de sortie avec le nombre de neurone égale au nombre de classes
    # puisqu'on a une classification multiclasse (6 classes), la dernière couche doit être formée de 6 neurones avec une fonction d'activation softmax
    x = Dense(6)(x)
    sortie = Activation('softmax')(x)
    return sortie


# Déclaration du modèle:
# La sortie de l'extracteur des features sert comme entrée à la couche complétement connectée
model = Model(input_layer, fully_connected(feature_extraction(input_layer)))

# Affichage des paramétres du modèle
# Cette commande affiche un tableau avec les détails du modèle 
# (nombre de couches et de paramétrer ...)
model.summary()

# Compilation du modèle :
# On définit l'optimisateur et la fonction de perte (loss) et la métrique à afficher durant l'entrainement (categorical_crossentropy, accuracy, ...)
# L'optimisateur utilisé avec ses paramétres (Exemple : optimizer=adam(learning_rate=0.001) )
# La valeur à afficher durant l'entrainement, metrics=['accuracy'] 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les données d'entrainement en mémoire
# quand il charge les images, il les ajuste (change la taille, les dimensions, la direction ...) 
# aléatoirement afin de rendre le modèle plus robuste à la position du sujet dans les images
# Note: On peut utiliser cette méthode pour augmenter le nombre d'images d'entrainement (data augmentation)
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# validation_data_generator: charge les données de validation en memoire
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size=training_batch_size, # nombre d'images à entrainer (batch size)
    class_mode="categorical", # classement binaire problème de 2 classes et categorical pour le multiclasse
    shuffle=True) # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage
# seed=42

# validation_generator: indique la méthode de chargement des données de validation
validation_generator = validation_data_generator.flow_from_directory(
    validationPath, # Place des images de validation
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=validation_batch_size,  # nombre d'images à valider
    class_mode="categorical",  # classement binaire problème de 2 classes et categorical pour le multiclasse
    shuffle=True) # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage
# seed=42

# On imprime l'indice de chaque classe (Keras numerote les classes selon l'ordre des dossiers des classes)
print(training_generator.class_indices)
print(validation_generator.class_indices)

# On charge les données d'entrainement et de validation
# x_train: Les données d'entrainement
# y_train: Les Ètiquettes des données d'entrainement
# x_val: Les données de validation
# y_val: Les Ètiquettes des données de validation
(x_train, y_train) = training_generator.next()
(x_val, y_val) = validation_generator.next()

# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Savegarder le modèle avec la meilleure validation accuracy ('val_acc') 
# Note: on sauvegarder le modèle seulement quand la précision de la validation s'améliore
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

# entrainement du modèle
classifier = model.fit(x_train, y_train,
                       epochs=fit_epochs, # nombre d'époques
                       batch_size=fit_batch_size, # nombre d'images entrainées ensemble
                       validation_data=(x_val, y_val), # données de validation
                       verbose=1, # mets cette valeur ‡ 0, si vous voulez ne pas afficher les détails d'entrainement
                       callbacks=[modelcheckpoint], # les fonctions à appeler à la fin de chaque époque (dans ce cas modelcheckpoint: qui sauvegarde le modèle)
                       shuffle=True)# shuffle les images 

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 4) Afficher le temps d'execution
# ***********************************************
# Plot accuracy over epochs (precision par époque)
print(classifier.history.keys())
start_time = time.time()
end_time = time.time()
print("Le temps d'exécution est de :", end_time - start_time, " secondes")


# ***********************************************
#                    QUESTION
# ***********************************************
#
# 5) Afficher la courbe d’exactitude par époque (Training vs Validation) ainsi que la courbe de perte (loss)
# ***********************************************

plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.show()
