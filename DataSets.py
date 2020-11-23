import numpy as np
import tensorflow as tf
from random import random
from time import time



#----------------------------------------------------------------------------------------------------------#
#-------------------------------------- DATASET POUR L'ENTRAINEMENT ---------------------------------------#
#----------------------------------------------------------------------------------------------------------#


class DataSet(object):
	""" Classe pour faciliter l'import des images et de leurs labels, gérer le passage d'un batch au batch suivant et calculer l'accuracy
		Remarque : un objet contient à la fois la base de test et la base de validation """


	def __init__(self, name: str, dimsImages: [int, int, int], batchSize: int = 128):
		self.name = name
		self.dims = dimsImages  # dimensions de l'image : [nbPixels hauteur, nbPixels largeur, nbCanaux (ex:RGB)]
		self.dim = dimsImages[0] * dimsImages[1] * dimsImages[2]  # nb de valeurs pour l'ensemble des pixels : 80*80*3=19200
		self.batchSize = batchSize

		self.curPos = 0
		self.nbImages_train, self.nbImages_val = 0, 0
		self.data_train, self.label_train = None, None
		self.data_val, self.label_val = None, None
		self.lastAcc_train, self.lastF1_train = 0, 0
		self.lastAcc_val, self.lastF1_val = 0, 0
		self.shuffle = True
		self.dataVal_loaded = False



	def prep_load_fromBIN_lab01(self, filename_data_lab0: str, filename_data_lab1: str, nbLab0: int, nbLab1: int, nbImDiffClasse0_train: int,
								nbUtilisations0_train: int, nbImDiffClasse0_val: int, rapport10_val: float = 88777./30517):
		""" Enregistre les paramètres de construction du dataset

			Parameters
			----------
			filename_data_lab0 : str
				Chemin du fichier .bin contenant toutes les images avec un label 0
			filename_data_lab1 : str
				Chemin du fichier .bin contenant toutes les images avec un label 1
			nbLab0 : int
				Nombre d'images dans le .bin des labels 0
			nbLab1 : int
				Nombre d'images dans le .bin des labels 1
			nbImDiffClasse0_train : int
				Nombre d'images de la classe 0 différentes qui vont être utilisées en train
			nbUtilisations0_train : int
				Nombre d'utilisation en train de chaque image de la classe 0
				(1 => chaque image est utilisée une seule fois, 2 => chaque image est copiée pour être utilisée 2 fois, ...)
			nbImDiffClasse0_val : int
				Nombre d'images de la classe 0 différentes qui vont être utilisées en validation (aucune duplication ensuite)
			rapport10_val : float
				Rapport entre le nombre d'images de la classe 1 et celui de la classe 0, dans la base validation
				(dans les données de train, il est de 2.9)
		"""

		# Enregistre les noms des fichiers et le nombre d'images qu'ils contiennent
		if filename_data_lab0 == "":
			filename_data_lab0 = "./DataBases/%s0.bin" % self.name
		if filename_data_lab1 == "":
			filename_data_lab1 = "./DataBases/%s1.bin" % self.name
		self.filename_data_lab0 = filename_data_lab0
		self.filename_data_lab1 = filename_data_lab1
		
		# Enregistre les nombres d'images utilisées de chacun des classes en train et test
		self.nbLab0 = nbLab0
		self.nbLab1 = nbLab1
		self.nbUtilisations0_train = nbUtilisations0_train
		self.nbImDiffClasse0_train = nbImDiffClasse0_train
		self.nbImDiffClasse1_train = nbImDiffClasse0_train * nbUtilisations0_train  # La base d'entraînement sera équilibrée
		self.nbImDiffClasse0_val = nbImDiffClasse0_val
		self.nbImDiffClasse1_val = round(nbImDiffClasse0_val * rapport10_val)  # La base de validation sera déséquilibrée, comme les données

		# Calcule le nombre d'images en entraînement et en validation
		self.nbImages_train = self.nbImDiffClasse1_train * 2
		self.nbImages_val = self.nbImDiffClasse0_val + self.nbImDiffClasse1_val

		# Alloue la mémoire pour les listes data et label, en train et en validation
		self.data_train = np.empty(shape=[self.nbImages_train, self.dim], dtype=np.float32)
		self.data_val = np.empty(shape=[self.nbImages_val, self.dim], dtype=np.float32)
		self.label_train = np.empty(shape=[self.nbImages_train, 2], dtype=np.float32)
		self.label_val = np.empty(shape=[self.nbImages_val, 2], dtype=np.float32)

		# Construit les listes pour sélectionner les images à utiliser en validation, sur l'ensemble des images disponibles
		# Les listes sont enregistrées comme attributs pour reprendre les mêmes éléments en validation si la partie de train est renouvelée
		self.l_selectPourVal_label0 = [True] * self.nbImDiffClasse0_val  +  [False] * (nbLab0 - self.nbImDiffClasse0_val)
		self.l_selectPourVal_label1 = [True] * self.nbImDiffClasse1_val  +  [False] * (nbLab1 - self.nbImDiffClasse1_val)
		if self.shuffle:
			np.random.shuffle(self.l_selectPourVal_label0)
			np.random.shuffle(self.l_selectPourVal_label1)



	def load_fromBIN_lab01(self):
		""" Charge un dataset à partir de deux fichiers .bin, un contenant les images de la classe 0 et un autre contenant les images labellisées 1.
			Permet de construire un dataset en train équilibré et un dataset en validation déséquilibré. """
		t = time()
		print("Chargement du dataset en cours...")

		# Construit deux listes pour choisir parmi les images non sélectionnées pour la validation, celles qui vont être utilisées en train
		l_selectPourTrain_label0 = ( [True] * self.nbImDiffClasse0_train + [False] * (self.nbLab0 - self.nbImDiffClasse0_val - self.nbImDiffClasse0_train) ) * self.nbUtilisations0_train
		l_selectPourTrain_label1 = [True] * self.nbImDiffClasse1_train + [False] * (self.nbLab1 - self.nbImDiffClasse1_val - self.nbImDiffClasse1_train)
		if self.shuffle:
			np.random.shuffle(l_selectPourTrain_label0)
			np.random.shuffle(l_selectPourTrain_label1)
		iter_selectTrain_lab0 = iter(l_selectPourTrain_label0)
		iter_selectTrain_lab1 = iter(l_selectPourTrain_label1)

		# Construit les listes contenant les indices de la liste de train mélangés, pour enregistrer les images à des positions aléatoires pour l'entraînement
		indices_train_melanges = np.arange(self.nbImages_train)
		if self.shuffle:
			np.random.shuffle(indices_train_melanges)

		# Parcourt l'ensemble du fichier contenant les images avec un label 0 et les enregistre soit en validation, soit en train, soit les ignore
		f = open(self.filename_data_lab0, 'rb')
		iTrain, iVal, offset = 0, 0, 0
		for selectPourVal in self.l_selectPourVal_label0:
			if selectPourVal:  # L'image est utilisée en validation
				if not self.dataVal_loaded:
					self.data_val[iVal, :] = (np.fromfile(f, dtype=np.uint8, count=self.dim) - 128.) / 256.
					self.label_val[iVal, :] = [1., 0.]
					iVal += 1
				else:
					offset += self.dim
			else:  # L'image est utilisée en train => on la considère plusieurs fois
				pixImages = np.fromfile(f, dtype=np.uint8, count=self.dim)
				for _ in range(self.nbUtilisations0_train):
					if next(iter_selectTrain_lab0):  # Regarde s'il faut l'enregistrer en train ou ne pas l'utiliser
						self.data_train[indices_train_melanges[iTrain], :] = self.__transfo_image(pixImages)
						self.label_train[indices_train_melanges[iTrain], :] = [1., 0.]
						iTrain += 1
		f.close()

		# Parcourt l'ensemble du fichier contenant les images avec un label 1 et les enregistre soit en validation, soit en train, soit les ignore
		f = open(self.filename_data_lab1, 'rb')
		offset = 0
		for selectPourVal in self.l_selectPourVal_label1:
			if selectPourVal:  # L'image est utilisée en validation
				if not self.dataVal_loaded:
					self.data_val[iVal, :] = (np.fromfile(f, dtype=np.uint8, count=self.dim, offset=offset) - 128.) / 256.
					self.label_val[iVal, :] = [0., 1.]
					iVal += 1
					offset = 0
				else:
					offset += self.dim
			else:  # L'image est utilisée en train => pas de duplication pour la classe 1
				if next(iter_selectTrain_lab1):  # Regarde s'il faut l'enregistrer en train ou ne pas l'utiliser
					self.data_train[indices_train_melanges[iTrain], :] = self.__transfo_image(np.fromfile(f, dtype=np.uint8, count=self.dim, offset=offset))
					self.label_train[indices_train_melanges[iTrain], :] = [0., 1.]
					iTrain += 1
					offset = 0
				else:  # Si l'image n'est pas utilisée, incrémente l'offset pour la passer lors de la prochaine lecture
					offset += self.dim
		f.close()
		self.dataVal_loaded = True

		# Affiche la composition du dataset
		nbLab1Train = sum(np.argmax(self.label_train, axis=1))
		nbLab1Val = sum(np.argmax(self.label_val, axis=1))
		print("Dataset chargé ! (%.1fs)" % (time() - t))
		print("Nb data en train =\t %d images : nb label 0 = %d | nb label 1 = %d" % (self.nbImages_train, self.nbImages_train - nbLab1Train, nbLab1Train))
		print("Nb data en validation =\t %d images : nb label 0 = %d | nb label 1 = %d" % (self.nbImages_val, self.nbImages_val - nbLab1Val, nbLab1Val))



	def reload_fromBIN_lab01(self):
		""" Supprime les données enregistrées et recharge à nouveau le dataset, en conservant les mêmes images pour la validation """
		self.curPos = 0
		self.load_fromBIN_lab01()



	def __transfo_image(self, tenseurImage: np.array):
		""" Applique aléatoirement une transformation à l'image pour faire du data augmentation """
		probaTransfo = random()

		if probaTransfo > 0.6:			# 50% => ne rien faire
			tenseurImage = tenseurImage.reshape(self.dims)
			tenseurImage = tf.image.flip_left_right(tenseurImage)
		# 	if probaTransfo < 0.7:		# 20% => symétrie horizontale
		# 		tenseurImage = tf.image.flip_left_right(tenseurImage)
		# 	elif probaTransfo < 0.85:	# 15% => luminosité
		# 		tenseurImage = tf.image.random_brightness(tenseurImage, 0.25)
		# 	else:						# 15% => saturation
		# 		tenseurImage = tf.image.random_saturation(tenseurImage, 0.8, 1.5)
			tenseurImage = tenseurImage.numpy().reshape([self.dim])
		return (tenseurImage - 128.) / 256.  # Applique une transformation pour ramener la valeur des pixels dans [-0.5 ; 0496]



	def load_fromBIN_dataLab(self, filename_data: str, filename_label: str, nbImages: int, trainSize: float, fLabelType: np.dtype = np.uint8):
		""" Charge un dataset à partir d'un fichier bin contenant les images et un second contenant les labels """

		if filename_data == "":
			filename_data = "./DataBases/%s.bin" % self.name
		if filename_label == "":
			filename_label = "./DataBases/%s_labels.bin" % self.name

		# Construit les listes contenant les indices mélangés, pour enregistrer les images à des positions aléatoires
		indice_melanges = np.arange(nbImages)
		if self.shuffle:
			np.random.shuffle(indice_melanges)

		# Créer le tableau "data" contenant *nbImages* vecteurs (un par image) mélangés, dans lequel se trouvent les 19200 valeurs des pixels (sur une seule ligne)
		f = open(filename_data, 'rb')
		data = np.empty(shape=[nbImages, self.dim], dtype=np.float32)
		for i in range(nbImages):
			data[indice_melanges[i], :] = (np.fromfile(f, dtype=np.uint8, count=self.dim) - 128.) / 256.  # Applique une transformation aux valeurs des pixels => [-0.5 ; 0.496]
		f.close()

		# Créer le tableau "label" contenant le vecteur label ([1,0] ou [0,1]) de chacune des *nbImages* images, mélangées
		if filename_label:
			f = open(filename_label, 'rb')
			label = np.empty(shape=[nbImages, 2], dtype=np.float32)
			for i in range(nbImages):
				label[indice_melanges[i], :] = np.fromfile(f, dtype=fLabelType, count=2)  # /!\ Attention au format de lecture
			f.close()

		# Divise en base d'entraînement et de validation
		self.nbImages_train = round(nbImages * trainSize)
		self.nbImages_val = nbImages - self.nbImages_train
		self.data_train = data[:self.nbImages_train]
		self.data_val = data[self.nbImages_train:]
		if filename_label:
			self.label_train = label[:self.nbImages_train]
			self.label_val = label[self.nbImages_train:]

		# Affiche la composition du dataset
		if filename_label:
			nbLab1Train = sum(np.argmax(self.label_train, axis=1))
			nbLab1Val = sum(np.argmax(self.label_val, axis=1))
			print("Nb data en train :\t nb label 0 = %d | nb label 1 = %d" % (len(self.label_train) - nbLab1Train, nbLab1Train))
			print("Nb data en validation :\t nb label 0 = %d | nb label 1 = %d" % (len(self.label_val) - nbLab1Val, nbLab1Val))



	def NextTrainingBatch(self):
		""" Retourne les valeurs des pixels et les labels des images du prochain du batch """
		if self.curPos + self.batchSize > self.nbImages_train:
			self.curPos = 0
		xs = self.data_train[self.curPos:self.curPos + self.batchSize, :]
		ys = self.label_train[self.curPos:self.curPos + self.batchSize, :]
		self.curPos += self.batchSize
		return xs, ys
	


	def __calc_mean_accuracy(self, model: tf.Module, p_data: np.array, p_label: np.array, p_numIter: int, previousAcc: float, previousF1: float, name: str):
		""" Calcule l'accuracy du modèle donné sur la partie du dataset choisie """

		confusion_matrix = tf.convert_to_tensor([ [0,0], [0,0] ])
		# Parcourt les données par paquet de batchsize
		for i in range(0, len(p_label), self.batchSize):
			curBatchSize = min(self.batchSize, len(p_label) - i)
			curImages = p_data[i:i+curBatchSize,:]
			curLabels = p_label[i:i+curBatchSize,:]
			y = model(curImages, log_summary=False, training=False)  # Utilise le modèle et calcule la matrice de sortie, en ne loggant pas cette itération et en désactivant le dropout
			curLabels_argmax, y_argmax = tf.argmax(curLabels, 1), tf.argmax(y, 1)  # Pour chaque ligne de la matrice, argmax retourne l'index de la valeur la plus grande ([1,0] => 0)
			confusion_matrix += tf.math.confusion_matrix(curLabels_argmax, y_argmax, num_classes=2)  # Calcule la matrice de confusion, qui permettra de calculer l'accuracy, la précision et le recall

		# Calcule les métriques d'exactitude
		confusion_matrix = confusion_matrix.numpy()
		if len(p_label) != np.sum(confusion_matrix):
			print("y a un pb")
		accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / len(p_label)
		precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
		recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
		scoreF1 = 2 * precision * recall / (precision + recall)
		with tf.name_scope(name):  # Log ces métriques
			tf.summary.scalar('Accuracy %s'%name, accuracy)
			tf.summary.scalar('Score F1 %s'%name, scoreF1)
			tf.summary.scalar('Precision %s'%name, precision)
			tf.summary.scalar('Recall %s'%name, recall)
		if p_numIter > 0:
			deltaAcc = "(%+.2f) " % ((accuracy - previousAcc)*100)
			deltaF1 = "(%+.2f) " % ((scoreF1 - previousF1)*100)
		else:
			deltaAcc, deltaF1 = "", ""

		# Affiche la matrice de confusion
		print("%s : Accuracy = %.2f%% %s// Score F1 = %.2f%% %s// Precision = %.2f%% // Recall = %.2f%%"
				% (name, accuracy*100, deltaAcc, scoreF1*100, deltaF1, precision*100, recall*100))
		print("  {:<7} | {:>6} | {:>6}".format("", "pred 0", "pred 1"))
		print("  {:<7} | {:>6} | {:>6}".format("label 0", confusion_matrix[0][0], confusion_matrix[0][1]))
		print("  {:<7} | {:>6} | {:>6}\n".format("label 1", confusion_matrix[1][0], confusion_matrix[1][1]))

		return accuracy, scoreF1

	

	def get_mean_accuracy(self, model: tf.Module, numIter: int):
		""" Affiche l'accuracy du modèle donné sur la base de train et la base de validation """
		if numIter == -1:
			print("    ----- Calcul des performances après le changement de modèle -----")
		else:
			print("    ----- Calcul des performances après %d itérations -----" % numIter)
		self.lastAcc_train, self.lastF1_train = self.__calc_mean_accuracy(model, self.data_train, self.label_train, numIter, self.lastAcc_train, self.lastF1_train, "Train")
		self.lastAcc_val, self.lastF1_val = self.__calc_mean_accuracy(model, self.data_val, self.label_val, numIter, self.lastAcc_val, self.lastF1_val, "Validation")



	def vider(self):
		""" Vide le dataset """
		# self.__dict__.clear()
		self.curPos = 0
		self.data_train, self.label_train = None, None
		self.data_val, self.label_val = None, None
		self.lastAcc_train, self.lastF1_train = 0, 0
		self.lastAcc_val, self.lastF1_val = 0, 0
		print("Dataset vidé !")





#----------------------------------------------------------------------------------------------------------#
#----------------------------- DATASET POUR LA VERIFICATION ET LA PREDICTION ------------------------------#
#----------------------------------------------------------------------------------------------------------#


class DataSetVerif(DataSet):
	""" Dataset spécifique pour la vérification et la prédiction. Il ne contient que des données en validation, pas en train """

	def __init__(self, name: str, dimsImages: [int, int, int], batchSize: int = 128):
		""" Crée un dataset non mélangé, avec toutes les données en validation """
		DataSet.__init__(self, name, dimsImages, batchSize)
		self.shuffle = False

	
	def load_fromBIN_lab01(self, filename_data_lab0: str, filename_data_lab1: str, nbLab0: int, nbLab1: int, nbImDiffClasse0_val: int, rapport10_val: float):
		""" Charge des données dans le dataset à partir du nombre d'images de la classe 0 à utiliser et le rapport entre nb1/nb0 """
		DataSet.prep_load_fromBIN_lab01(self, filename_data_lab0, filename_data_lab1, nbLab0, nbLab1, 0, 0, nbImDiffClasse0_val, rapport10_val)
		DataSet.load_fromBIN_lab01(self)


	def load_fromBIN_dataLab(self, filename_data: str, filename_label: str, nbImages: int, fLabelType: np.dtype = np.uint8):
		""" Charge les données à partir de deux fichiers .bin, un avec les images et l'autre avec les labels """
		DataSet.load_fromBIN_dataLab(self, filename_data, filename_label, nbImages, 0, fLabelType)


	def get_mean_accuracy(self, model: tf.Module):
		""" Affiche l'accuracy du modèle donné sur le dataset """
		print("    ----- Vérification des performances du modèle -----")
		DataSet._DataSet__calc_mean_accuracy(self, model, self.data_val, self.label_val, 0, 0, 0, "Vérification")


	def predict(self, model: tf.Module, date: str):
		""" Crée un fichier .txt contenant les prédictions du modèle pour le dataset donné """
		print("    ----- Prédit les labels des données à partir du modèle -----")
		f = open("%s_%s.txt" % (self.name, date), "w")
		for i in range(0, self.nbImages_val, self.batchSize):
			curBatchSize = min(self.batchSize, self.nbImages_val - i)
			curImages = self.data_val[i:i+curBatchSize,:]
			y = model(curImages, log_summary=False, training=False)
			for lab in tf.argmax(y, 1).numpy():
				f.write("%d\n" % lab)
		f.close()
