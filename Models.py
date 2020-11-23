import tensorflow as tf
import Layers



#----------------------------------------------------------------------------------------------------------#
#------------------------------ CLASSE CONVOLUTIONAL NEURAL NETWORK ---------------------------------------#
#----------------------------------------------------------------------------------------------------------#


class ConvNeuralNet(tf.Module):
	""" Réseau de neurones convolutif assez classique, avec 3 couches de convolution, entrecoupées par 3 maxpooling, et qui se termine par une FC """
	
	def __init__(self, img_height: int, img_width: int, img_channels: int):
		""" A l'initialisation, crée les différentes couches pour former la structure suivante :
				- Couche de unflatten : 19200 => 80x80x3
				- Couche de convolution :	=> 80x80x6
				- Couche de convolution :	=> 80x80x6
				- Couche de maxpooling :	=> 40x40x6
				- Couche de convolution :	=> 40x40x12
				- Couche de convolution :	=> 40x40x12
				- Couche de maxpooling :	=> 20x20x12
				- Couche de convolution :	=> 20x20x24
				- Couche de convolution :	=> 20x20x24
				- Couche de maxpooling :	=> 10x10x24
				- Couche de convolution :	=> 10x10x48
				- Couche de convolution :	=> 10x10x48
				- Couche de maxpooling :	=> 5x5x48
				- Couche de flatten :		=> 1200
				- Couche fully-connected :	=> 50
				- Couche fully-connected :	=> 2
		"""

		lCouches = []
		lCouches.append( Layers.Unflat('unflat', img_height, img_width, img_channels) )

		nbfilter = 6
		for i in range(4):
			lCouches.append( Layers.ResLearningBlock("resBlock_%d"%i, output_dim=nbfilter, filterSize=3, stride=1, dropout_rate = 0.1) )
			lCouches.append( Layers.Maxpool('pool', 2) )
			nbfilter *= 2
		lCouches.append(Layers.Flat())
		lCouches.append(Layers.FC('fc', 50))
		lCouches.append(Layers.FC('fc', 2))
		# lCouches.append(Layers.Sigmo())

		self.lCouches = lCouches


	def __call__(self, x: tf.Tensor, log_summary: bool, training: bool):
		""" Lors de l'appel, fait passer la matrice x dans les différentes couches du réseau """
		for couche in self.lCouches:
			x = couche(x, log_summary, training)
		return x


	def get_L2_loss(self):
		L2_loss = 0
		for couche in self.lCouches:
			L2_loss += couche.get_L2_loss()
		return L2_loss  # Somme de tous les w², divisée par 2



#----------------------------------------------------------------------------------------------------------#
#---------------------------------------- FONCTION TRAIN_ONE_ITER -----------------------------------------#
#----------------------------------------------------------------------------------------------------------#

def train_one_iter(model: tf.Module, optimizer: tf.optimizers, betaL2: float, image: tf.Tensor, label: tf.Tensor, log_summary: bool):
	""" Algorithme appliqué à chaque itération de l'entraînement, qui optimise les coefficients du cnn en utilisant les gradients sur la loss de cross-entropy """

	# Active le GradientTape qui va enregistrer les modifications faites sur des tf.Variables pour calculer les gradients
	# https://www.tensorflow.org/guide/autodiff?hl=en | https://medium.com/analytics-vidhya/tf-gradienttape-explained-for-keras-users-cc3f06276f22
	with tf.GradientTape() as tape:
		y = model(image, log_summary, training=True)  # Fait passer le batch d'images dans le cnn
		y = tf.nn.log_softmax(y)  # Applique une log_softmax à l'ensemble des vecteurs de sortie (softmax => proba entre [0;1] avec somme=1 // log(softmax) => envoie dans [-inf;0]),
								  # donc pour chaque image on passe par la proba pour chacune des deux classes, et on récupère le log de chacune, cad deux valeurs entre [-inf;0].
		diff = label * y  # Calcule la différence pour chaque image du batch en faisant le produit matriciel => diff = 1 x val, avec val ϵ [-inf ; 0 (si proba correcte à 1)]
		loss = -tf.reduce_sum(diff)  # Somme les différences et prend l'inverse. On vient de finir le calcul de la cross-entropy : https://en.wikipedia.org/wiki/Cross_entropy
		if log_summary:  # Log la valeur de la loss si demandée
			tf.summary.scalar('cross entropy', loss)
		grads = tape.gradient(loss, model.trainable_variables)  # Calcule les gradients de la loss pour chacune des variables du modèle
		optimizer.apply_gradients(zip(grads, model.trainable_variables))  # Modifie les coefficients du réseau en utilisant les gradients selon la méthode de l'optimizer choisi
	return loss
