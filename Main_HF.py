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

logFileName = __dir__ + './logsHF/%s' % date

def loadModel(p_modelFolder: str, p_modelNum: int, p_optimizer: tf.optimizers, p_model: tf.Module):
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=p_optimizer, net=p_model)
	ckpt.restore(__dir__ + "./modelsHF/%s/saved_model-%d" % (p_modelFolder, p_modelNum))



#----------------------------------------------------------------------------------#
#-------------------------------------- MAIN --------------------------------------#
#----------------------------------------------------------------------------------#


# Format des images
img_height, img_width, img_channels = 48, 48, 1



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
	dataset = DataSet("HF_data_10k_label", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	dataset.prep_load_fromBIN_lab01(filename_data_lab0 = "",
									filename_data_lab1 = "",
									nbLab0 = 4768,
									nbLab1 = 5232,
									nbImDiffClasse0_train = 4000,
									nbUtilisations0_train = 1,
									nbImDiffClasse0_val = 750,
									rapport10_val = 5232./4768.)
	dataset.load_fromBIN_lab01()
	mean = np.mean(dataset.data_train)
	std = np.std(dataset.data_train)
	print("mean = %.2f // std = %.2f" % (mean, std))
	train_model(dataset, simple_cnn, optimizer, logFileName, betaL2=0.01, interv_reload=200,
				nbIterMax=3500, min_delta=0.02, patience=5000, nbElemMoyGlissante=25, verbose=2, interv_accuracy=200)
	dataset.vider()



# Sauvegarde du modèle (oui par défaut)
action_save = input("Sauvegarder le modèle ? (O/n) : ")
if action_save != "n":
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=simple_cnn)
	ckpt.save(__dir__ + "/modelsHF/saved_model_%s/saved_model" % date)
	print("Modèle sauvegardé dans 'modelsHF/saved_model_%s/saved_model'" % date)



# Vérification des performances du modèle (non par défaut)
action_verif = input("Vérifier les performances du modèle ? (O/n) : ")
if action_verif == "O":
	datasetVerif = DataSetVerif("HF_dataset_100k", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	datasetVerif.load_fromBIN_dataLab(filename_data = "",
									filename_label = "",
									nbImages = 100000,
									fLabelType = np.float32)
	datasetVerif.get_mean_accuracy(simple_cnn)
	datasetVerif.vider()
	
	datasetVerif = DataSetVerif("HF_dataset_test10k", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	datasetVerif.load_fromBIN_dataLab(filename_data = "",
									filename_label = "",
									nbImages = 10000,
									fLabelType = np.float32)
	datasetVerif.get_mean_accuracy(simple_cnn)
	datasetVerif.vider()



# Prédiction des labels
action_predict = input("Prédire les performances du modèle ? (O/n) : ")
if action_predict == "O":
	datasetPredict = DataSetVerif("HF_dataset_test10k", dimsImages=[img_height, img_width, img_channels], batchSize=128)
	datasetPredict.load_fromBIN_dataLab(filename_data = "",
										filename_label = False,
										nbImages = 10000)
	datasetPredict.predict(simple_cnn, date)
	datasetPredict.vider()
