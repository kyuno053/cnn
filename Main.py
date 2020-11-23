import tensorflow as tf
import numpy as np
from DataSets import DataSet, DataSetVerif
from Models import ConvNeuralNet
from Train import train_model
import os
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



date = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

if "\\" in __file__:
	__dir__ = __file__[:-__file__[::-1].index("\\")]
else:
	__dir__ = __file__[:-__file__[::-1].index("/")]

logFileName = __dir__ + './logs/%s' % date

def loadModel(p_modelFolder: str, p_modelNum: int, p_optimizer: tf.optimizers, p_model: tf.Module):
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=p_optimizer, net=p_model)
	ckpt.restore(__dir__ + "./models/%s/saved_model-%d" % (p_modelFolder, p_modelNum))



#----------------------------------------------------------------------------------#
#-------------------------------------- MAIN --------------------------------------#
#----------------------------------------------------------------------------------#


# Format des images
img_height, img_width, img_channels = 80, 80, 3



# Initialisation du modèle et de l'optimizer
optimizer = tf.optimizers.Adam(1e-3)  # Choix de l'optimizer et de son pas : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers | https://www.youtube.com/watch?v=mdKjMPmcWjY
simple_cnn = ConvNeuralNet(img_height, img_width, img_channels)  # Initialise le modèle



# Chargement du modèle (non par défaut)
action_load = input("Charger un modèle ? (O/n) : ")
if action_load == "O":
	modelFolder = input("  Dossier du modèle : ")
	modelNum = int(input("  Numéro du modèle : "))
	loadModel(modelFolder, modelNum, optimizer, simple_cnn)



# Entraînement du modèle (oui par défaut)
action_train = input("Entraîner le modèle ? (O/n) : ")
if action_train != "n":
	dataset = DataSet("B_train_label", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	dataset.prep_load_fromBIN_lab01(filename_data_lab0 = "",
									filename_data_lab1 = "",
									nbLab0 = 30517,
									nbLab1 = 88777,
									nbImDiffClasse0_train = 20000,
									nbUtilisations0_train = 2,
									nbImDiffClasse0_val = 5000,
									rapport10_val = 88777./30517.)
	# dataset = DataSet("data_10k", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	# dataset.prep_load_fromBIN_lab01(filename_data_lab0 = __dir__ + "DataBases/data_10k_label0.bin",
	# 								filename_data_lab1 = __dir__ + "DataBases/data_10k_label1.bin",
	# 								nbLab0 = 4768,
	# 								nbLab1 = 5232,
	# 								nbImDiffClasse0_train = 4000,
	# 								nbUtilisations0_train = 1,
	# 								nbImDiffClasse0_val = 750,
	# 								rapport10_val = 5232./4768.)
	dataset.load_fromBIN_lab01()
	train_model(dataset, simple_cnn, optimizer, logFileName, betaL2=0.01, interv_reload=1000,
				nbIterMax=10000, min_delta=0.02, patience=5000, nbElemMoyGlissante=25, verbose=2, interv_accuracy=200)
	dataset.vider()



# Sauvegarde du modèle (oui par défaut)
action_save = input("Sauvegarder le modèle ? (O/n) : ")
if action_save != "n":
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=simple_cnn)
	ckpt.save(__dir__ + "/models/saved_model_%s/saved_model" % date)
	print("Modèle sauvegardé dans 'models/saved_model_%s/saved_model'" % date)



# Vérification des performances du modèle (non par défaut)
action_verif = input("Vérifier les performances du modèle ? (O/n) : ")
if action_verif == "O":
	# datasetVerif = DataSetVerif("A_train_part2", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	# datasetVerif.load_fromBIN_dataLab(filename_data = "",
	# 								  filename_label = "",
	# 								  nbImages = 59647)
	datasetVerif = DataSetVerif("data_test100k", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	datasetVerif.load_fromBIN_dataLab(filename_data = __dir__ + "DataBases/data_100k.bin",
									  filename_label = __dir__ + "DataBases/gender_100k.bin",
									  nbImages = 100000)
	datasetVerif.get_mean_accuracy(simple_cnn)
	datasetVerif.vider()



# Prédiction des labels
action_predict = input("Prédire les performances du modèle ? (O/n) : ")
if action_predict == "O":
	datasetPredict = DataSetVerif("C_test", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	datasetPredict.load_fromBIN_dataLab(filename_data = __dir__ + "../DataBases/test.bin",
										filename_label = False,
										nbImages = 21974)
	datasetPredict.predict(simple_cnn, date)
	datasetPredict.vider()
