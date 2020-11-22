import tensorflow as tf
import numpy as np
from DataSets import DataSet
from Models import ConvNeuralNet, train_one_iter



def train_model(p_dataset: DataSet, p_model: tf.Module, p_optimizer: tf.optimizers, logFileName: str, betaL2: float, interv_reload: int,
				nbIterMax: int, min_delta: int, patience: int, nbElemMoyGlissante: int, interv_accuracy: int = 500, verbose: int = 1):
	""" Fonction qui gère l'entraînement du modèle, dont le nombre d'itérations max et l'early-stopping """
	train_summary_writer = tf.summary.create_file_writer(logFileName)  # Crée le fichier de logs pour pouvoir suivre l'évolution dans la tensorboard

	# Gère les niveaux de verbose (0 à 3)
	if verbose <= 0:
		interv_print = nbIterMax
	else:
		interv_print = int(1000/(10**verbose))  # 1 => 100 // 2 => 10 // 3 => 1

	# Fait des itérations d'entraînement
	earlyStopping_counter = 0
	max_earlyStopping_counter = 0
	l_lastLosses = np.full(shape=(nbElemMoyGlissante), fill_value=999999, dtype=np.float32)
	minSumLosses = sum(l_lastLosses)
	for numIter in range(nbIterMax):
		tf.summary.experimental.set_step(numIter)

		# Affiche et enregistre l'accuracy, la précision, le rappel et la matrice de confusion toutes les 500 itérations
		if numIter % interv_accuracy == 0:
			with train_summary_writer.as_default():
				p_dataset.get_mean_accuracy(p_model, numIter)

		# Entraîne le modèle
		ima, lab = p_dataset.NextTrainingBatch()  # Récupère les données (valeurs des pixels et labels) des images du batch suivant
		with train_summary_writer.as_default():  # Active l'enregistrement des logs
			loss = train_one_iter(p_model, p_optimizer, betaL2, ima, lab, numIter % 10 == 0)  # Fait une itération d'entraînement, en enregistrant les logs toutes les 10 iter
			loss += betaL2 * p_model.get_L2_loss()

		# Affiche la perte toutes les *interv_print* itérations
		if numIter % interv_print == 0:
			print("numIter = %6d - loss = %.3f - max_earlyStopping_counter = %d" % (numIter, loss, max_earlyStopping_counter))
			max_earlyStopping_counter = 0  # Affiche la valeur maximale de l'earlyStopping_counter depuis le dernier print

		# Early-stopping
		l_lastLosses[numIter % nbElemMoyGlissante] = loss.numpy()
		if minSumLosses - sum(l_lastLosses) < min_delta:
			earlyStopping_counter += 1
			if earlyStopping_counter > max_earlyStopping_counter:
				max_earlyStopping_counter = earlyStopping_counter
		else:
			earlyStopping_counter = 0
			minSumLosses = sum(l_lastLosses)
		if earlyStopping_counter > patience:
			print("\n----- EARLY STOPPING : numIter = %6d - loss = %f - earlyStopping_counter = %d -----" % (numIter, loss, earlyStopping_counter))
			break

		# Vide puis recharge le dataset
		if numIter > 0 and numIter % interv_reload == 0:
			p_dataset.reload_fromBIN_lab01()
			p_dataset.get_mean_accuracy(p_model, -1)


	# On finit en beauté par un calcul de l'accuracy
	with train_summary_writer.as_default():
		p_dataset.get_mean_accuracy(p_model, numIter)
