import tensorflow as tf


class Couche(tf.Module):
	def get_L2_loss(self):
		return 0



#----------------------------------------------------------------------------------------------------------#
#-------------------------------------------- FULLY-CONNECTED ---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class FC(Couche):
	""" Couche fully-connected classique """

	def __init__(self, name: str, output_dim: int, dropout_rate: float = 0.):
		""" Lors de la création de la couche, enregistre ses attributs et crée seulement le vecteur des biais """
		self.scope_name = name  # Enregistre le nom du scope pour ajouter un préfixe aux opérations et mieux l'identifier
		self.output_dim = output_dim  # Nb de neurones en sortie
		self.b = tf.Variable(tf.constant(0.0, shape=[self.output_dim]), name='b_'+name)  # Initialise le vecteur des biais (tous à 0)
		self.rate = dropout_rate  # Taux de dropout


	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		""" Lors de l'appel de la couche, applique la transformation x*w+b (et crée la matrice w si elle n'existe pas) """

		# Si la matrice des poids w n'a pas encore été définie, la crée en regardant les dimensions du x fourni par la couche précédente pour connaître l'input_dim
		# Rmq : x.shape[0] = batchSize  //  x.shape[1] = nb de neurones en sortie de la couche précédente
		if not hasattr(self, 'w'):
			w_init = tf.random.truncated_normal(shape=[x.shape[1], self.output_dim], stddev=0.1, dtype=tf.float32)
			self.w = tf.Variable(w_init, name='w_'+self.scope_name)
			print('build fc %s  %d => %d' % (self.scope_name,  x.shape[1], self.output_dim))

		# Si log_summary est activé, active le scope et calcule les métriques pour la tensorboard
		if log_summary:
			with tf.name_scope(self.scope_name):
				tf.summary.scalar("mean w", tf.reduce_mean(self.w))
				tf.summary.scalar("max w", tf.reduce_max(self.w))
				tf.summary.histogram("w", self.w)
				tf.summary.scalar("mean b", tf.reduce_mean(self.b))
				tf.summary.scalar("max b", tf.reduce_max(self.b))
				tf.summary.histogram("b", self.b)

		# Calcule le vecteur résultat de x*w+b qui correspond au passage entre la couche N-1 et N
		x = tf.matmul(x, self.w) + self.b

		# Si on est en entraînement, met une certaine proportion des sorties à 0 pour éviter l'overfitting
		# La valeur des autres sorties est multipliée par 1/(1-rate) pour conserver une valeur globale constante
		if training:
			x = tf.nn.dropout(x, rate=self.rate)

		return x
	

	def get_L2_loss(self):
		return tf.nn.l2_loss(self.w)  # Retourne la somme de tous les w², divisée par 2



#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------- CONVOLUTION -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class Conv(Couche):
	""" Couche de convolution : applique plusieurs filtres sur tous les canaux en entrée (avec des images sous forme de matrices, pas en lignes) puis applique une ReLU """

	def __init__(self, name: str, output_dim: int, filterSize: int, stride: int = 1, dropout_rate: float = 0., useBias: bool = True):
		""" Lors de la création de la couche, enregistre les paramètres de la couche et crée le vecteur b """
		self.scope_name = name  # Enregistre le nom du scope pour ajouter un préfixe aux opérations et mieux l'identifier
		self.filterSize = filterSize  # Taille du filtre (côté du petit carré)
		self.output_dim = output_dim  # Nb de filtres, donc de canaux en sortie
		self.stride = stride  # Pas (en pixels) selon lequel le filtre se décale sur une image à chaque fois
		if useBias:  # S'il y a une batch normalization ensuite le biais ici est inutile
			self.b = tf.Variable(tf.constant(0.0, shape=[self.output_dim]), name='b_'+self.scope_name)  # Vecteur b, contenant un biais par filtre
		self.rate = dropout_rate  # Taux de dropout


	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		""" Lors de l'appel de la couche, calcule l'opération de convolution puis le passage dans une fonction ReLU ########################## """

		# Si la matrice des filtres w n'a pas encore été définie, la crée en regardant les dimensions du x fourni par la couche précédente pour connaître l'input_dim
		# Rmq : x.shape[0] = batchSize  //  x.shape[1] = hauteur de l'image  //  x.shape[2] = largeur de l'image  //  x.shape[3] = nb de canaux ou de filtres
		if not hasattr(self, 'w'):
			w_init = tf.random.truncated_normal([self.filterSize, self.filterSize, x.shape[3], self.output_dim], stddev=0.1, dtype=tf.float32)
			self.w = tf.Variable(w_init, name='w_'+self.scope_name)
			print('build conv %s %dx%d  %d => %d'%(self.scope_name,self.filterSize,self.filterSize, x.shape[3], self.output_dim))

		# Si log_summary est activé, active le scope et calcule les métriques pour la tensorboard
		if log_summary:
			with tf.name_scope(self.scope_name):
				tf.summary.scalar("mean w", tf.reduce_mean(self.w))
				tf.summary.scalar("max w", tf.reduce_max(self.w))
				tf.summary.histogram("w", self.w)
				if hasattr(self, 'b'):
					tf.summary.scalar("mean b", tf.reduce_mean(self.b))
					tf.summary.scalar("max b", tf.reduce_max(self.b))
					tf.summary.histogram("b", self.b)

		# Calcule x*w+b dans le cadre d'une convolution
		# https://cs231n.github.io/assets/conv-demo/index.html (source : https://cs231n.github.io/convolutional-networks/)	
		x = tf.nn.conv2d(input=x,
						 filters=self.w,
						 strides=[1, self.stride, self.stride, 1],  # Pas selon lequel est faite la convolution, pour chaque dim de l'entrée en NHWC (= batch, width, height, channels)
						 padding='SAME'  # On utilise le zero-padding pour conserver la même dimension des images en entrée et en sortie
						)
		if hasattr(self, 'b'):
			x += self.b

		# Si on est en entraînement, met une certaine proportion des sorties à 0 pour éviter l'overfitting
		# La valeur des autres sorties est multipliée par 1/(1-rate) pour conserver une valeur globale constante
		if training:
			x = tf.nn.dropout(x, rate=self.rate)

		# ReLU ou pas ReLU ?
		return x
		# return tf.nn.relu(x)
	

	def get_L2_loss(self):
		return tf.nn.l2_loss(self.w)  # Retourne la somme de tous les w², divisée par 2



#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------- MAX POOLING -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class Maxpool(Couche):
	""" Couche de max pooling : pour chaque fenêtre de l'entrée, retourne en sortie la plus grande valeur présente : https://cs231n.github.io/assets/cnn/maxpool.jpeg """

	def __init__(self, name: str, poolSize: int):
		""" Lors de la création, initialise les attributs, notamment le facteur de pooling """
		self.scope_name = name
		self.poolSize = poolSize

	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		""" Lors de l'appel, fait l'opération de max pooling """
		return tf.nn.max_pool2d(x,
								ksize=(1, self.poolSize, self.poolSize, 1),    # Taille de la fenêtre de pooling, selon chaque dimension de l'entrée en NHWC
								strides=(1, self.poolSize, self.poolSize, 1),  # Pas selon lequel est décalée à chaque étape la fenêtre, pour toutes les dim
								padding='SAME')  # On utilise le zero-padding pour conserver la même dimension des images en entrée et en sortie



#----------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- FLAT --------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class Flat(Couche):
	""" Couche de flatten : prend en entrée un tenseur x en 4D contenant un batch d'images, et retourne une matrice 2D après avoir regroupé les infos de chaque image sur une ligne """
	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		inDimH = x.shape[1]  # hauteur des images
		inDimW = x.shape[2]  # largeur des images
		inDimD = x.shape[3]  # nb de canaux des images
		# Retourne une matrice contenant une ligne par image du batch (image ou objet similaire), en mettant, pour chaque image, bout à bout les valeurs du tenseur 3D qui la représente
		# Rmq : L'opération est faite sans toucher aux données contenues dans le tenseur, elle est donc très efficace et rapide peu importe la taille du tenseur.
		return tf.reshape(x, [-1, inDimH * inDimW * inDimD])  # (-1 pour que la batchSize, la shape[0], soit gérée automatiquement)



#----------------------------------------------------------------------------------------------------------#
#------------------------------------------------- UNFLAT -------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class Unflat(Couche):
	""" Couche de unflatten : convertit une matrice 2D, contenant un batch d'images avec une image par ligne, en une matrice 4D qui reconstitue lignes/colonnes/canaux """

	def __init__(self, name: str, outDimH: int, outDimW: int, outDimD: int):
		""" Lors de l'initialisation, paramètre la transformation avec les dimensions du tenseur de sortie """
		self.scope_name = name
		self.new_shape = [-1, outDimH, outDimW, outDimD]
		print('def unflat %s ? => %d %d %d' % (self.scope_name, outDimH, outDimW, outDimD))

	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		""" Lors de l'appel, applique la transformation et reconstitue les 3 dimensions lignes/colonnes/canaux des images """
		# Rmq : L'opération est faite sans toucher aux données contenues dans le tenseur, elle est donc très efficace et rapide peu importe la taille du tenseur.
		x = tf.reshape(x, self.new_shape)
		# if log_summary:
		# 	with tf.name_scope(self.scope_name):
		# 		tf.summary.image('input', x, 5)
		return x



#----------------------------------------------------------------------------------------------------------#
#------------------------------------------------ SIGMOID -------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class Sigmo(Couche):
	""" Couche qui applique la fonction sigmoide """
	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		return tf.nn.sigmoid(x)



#----------------------------------------------------------------------------------------------------------#
#------------------------------------------------ RELU -------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class ReLU(Couche):
	""" Couche qui applique la fonction ReLU """
	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		return tf.nn.relu(x)



#----------------------------------------------------------------------------------------------------------#
#------------------------------------------ BATCH NORMALIZATION -------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class BN(Couche):
	""" Couche de batch normalization

		Sources
		-------
		Documentation officielle de tensorflow
			https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
		Article sur la batch normalization dans tensorflow
			https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
		Exemple de code sur GitHub avec tensorflow 1.1
			https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412#file-mnist_cnn_bn-py-L22
	"""

	def __init__(self, name:str, decay: int = 0.999):
		self.scope_name = name
		self.decay = decay
		self.epsilon = 1e-3


	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		if not hasattr(self, 'beta'):
			# Paramètres sur lesquels le réseau peut jouer pour ajuster l'effet de la normalisation
			self.beta = tf.Variable(tf.zeros(shape=x.shape[-1], dtype=tf.float32), name='beta_'+self.scope_name)  # offset => modification de la moyenne
			self.gamma = tf.Variable(tf.ones(shape=x.shape[-1], dtype=tf.float32), name='gamma_'+self.scope_name)  # scale => modification de l'écart-type

			# Estimateurs de la moyenne et de la variance de la population, qui seront utilisés pendant la phase de test
			# Ces variables seront mis à jour manuellement, pas par l'optimizer
			self.shadow_pop_mean = tf.Variable(tf.zeros(shape=x.shape[-1], dtype=tf.float32), name='pop_mean'+self.scope_name, trainable=False)
			self.shadow_pop_var = tf.Variable(tf.ones(shape=x.shape[-1], dtype=tf.float32), name='pop_var'+self.scope_name, trainable=False)


		# Pendant la phase d'entraînement, utilise la moyenne et la variance observées, et met à jour les estimateurs globaux
		if training:

			# Calcule la moyenne et la variance, pour chaque canal du batch d'entrée (pour chaque filtre par exemple)
			batch_mean, batch_var = tf.nn.moments(x, range(len(x.shape)-1))  # range(len(x.shape)-1) = [0,1,2] s'il y a 4 dimensions (BHWC)

			# Met à jour les estimateurs de la moyenne et de la variance en utilisant une moyenne mobile exponentielle
			# C'est ce que fait la classe ExponentialMovingAverage : https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
			self.shadow_pop_mean = self.shadow_pop_mean * self.decay + batch_mean * (1 - self.decay)
			self.shadow_pop_var = self.shadow_pop_var * self.decay + batch_var * (1 - self.decay)

			if log_summary:
				with tf.name_scope(self.scope_name):
					tf.summary.scalar("mean beta (offset)", tf.reduce_mean(self.beta))
					tf.summary.scalar("max beta (offset)", tf.reduce_max(self.beta))
					tf.summary.histogram("beta (offset)", self.beta)
					tf.summary.scalar("mean gamma (scale)", tf.reduce_mean(self.gamma))
					tf.summary.scalar("max gamma (scale)", tf.reduce_max(self.gamma))
					tf.summary.histogram("gamma (scale)", self.gamma)
					
			return tf.nn.batch_normalization(x, batch_mean, batch_var, self.beta, self.gamma, self.epsilon)
		

		# En phase de test, utilise les estimateurs de la moyenne et de la variance
		else:
			return tf.nn.batch_normalization(x, self.shadow_pop_mean, self.shadow_pop_var, self.beta, self.gamma, self.epsilon)



#----------------------------------------------------------------------------------------------------------#
#---------------------------------------- RESIDUAL LEARNING BLOCK -----------------------------------------#
#----------------------------------------------------------------------------------------------------------#

class ResLearningBlock(Couche):
	""" Bloc d'apprentissage résiduel composé d'une convolution avec un filtre de taille 1x1 sur le chemin direct
		et de deux convolutions suivies d'une batch normalization sur l'autre chemin """

	def __init__(self, name: str, output_dim: int, filterSize: int, stride: int = 1, dropout_rate: float = 0.):
		self.lin = Conv(name+"_lin", output_dim, 1, stride, dropout_rate)
		self.conv1 = Conv(name+"_conv1", output_dim, filterSize, stride, dropout_rate)
		self.conv2 = Conv(name+"_conv2", output_dim, filterSize, stride, dropout_rate)  #, useBias=False)
		# self.bn = BN(name+"_BN")
	
	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		xlin = self.lin(x, log_summary, training)
		x = self.conv1(x, log_summary, training)
		x = self.conv2(x, log_summary, training)
		# x = self.bn(x, log_summary, training)
		return xlin+x	

	def get_L2_loss(self):
		return self.lin.get_L2_loss() + self.conv1.get_L2_loss() + self.conv2.get_L2_loss()
