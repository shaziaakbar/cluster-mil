from keras.datasets import mnist

import time
import os
import numpy as np
import random
import math

import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

from keras.models import Model
from keras.layers import Input, Lambda, Dense, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, \
	BatchNormalization, Flatten
from keras.utils import to_categorical

from keras import backend as K

K.set_image_data_format('channels_first')
K.set_session(sess)

from loss import LossFunction as my_loss
import latent

FV_LENGTH = 64

from sklearn.cluster import KMeans


def get_model(loss_type, alpha):
	"""Builds simple AlexNet-like architecture

    Returns:
      Keras Model with adaptive loss function incorporated
    """

	# Initialize Input parameters
	input_img = Input(shape=(1, 28, 28), name='input_data')
	input_feature = Input(shape=(10,), dtype='float32', name='input_feature')
	input_y = Input(shape=(10,), dtype='float32', name='input_y')

	# Build simple CNN for digits classification
	x = Conv2D(64, (3, 3), activation='relu')(input_img)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2))(x)
	x = Conv2D(32, (3, 3), activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2))(x)
	x = Dropout(rate=0.2)(x)
	x = Flatten()(x)
	cnn_model = Dense(128, activation='relu')(x)
	cnn_model_output = Dense(10, activation='softmax', name='p_out')(cnn_model)

	loss_function = my_loss(loss_type, alpha, 10)

	# Use Lambda layer to implement our custom loss function
	# todo: implement custom Layer
	output = Lambda(loss_function.loss_main, output_shape=(1,), name='joint_loss')(
		[input_y, cnn_model_output, input_feature])

	mibl_model = Model([input_img, input_feature, input_y], output)

	return mibl_model
                   

def minibatch_mibl_gen(x, y, batch_size, fmodel, num_instances, num_clusters=100, fraction_class=1.0):
	"""Generator for multiple instance learning framework

    Args:
      x, y: image data and labels (ndarray)
      batch_size: number of images per batch (int)
      fmodel: pretrained Keras autoencoder which outputs encoder per image (Keras Model)
      num_instances: number of images per bag (int)
      num_clusters: number of clusters to extract from latent space (int)
      fraction_class: percentage of bag which is reflective of bag label (float between 0 and 1)

    Returns:
      Batch containing image, label and estimated label
    """

    # Get encodings from model
	x_latent = x.reshape((len(x), np.prod(x.shape[1:])))
	z_mean, z_log = fmodel.predict(x_latent, batch_size=100)

	# Perform KMeans once at start
	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(z_mean)
	class_mean = kmeans.cluster_centers_
	labels = kmeans.labels_

	while True:

		y_dense = np.argmax(y, axis=1)
		indices = np.arange(len(y))
		np.random.shuffle(indices)

		num_correct_instances = int(math.ceil(fraction_class * float(num_instances)))
		num_random_instances = num_instances - num_correct_instances

		if num_correct_instances == 0:
			raise ("No positive labels in bag! We need to have a minimum of 1.")

		# --------------- Setup MNIST-BAG --------------- 
		# Split dataset so that a certain proportion of labels are in a bag
		if fraction_class < 1:
			sorted_classes = np.argsort(y_dense)
			if num_random_instances > 0:
				indices_toshuffle = np.array([])
				for kdx in range(num_correct_instances, len(y_dense) - num_instances, num_instances):
					indices_toshuffle = np.concatenate((indices_toshuffle, np.arange(kdx, kdx + num_random_instances)),
																	axis=0)

				indices_toshuffle = indices_toshuffle.astype('int32')
				np.random.shuffle(indices_toshuffle)

				i = 0
				for jdx in range(num_correct_instances, len(y_dense) - num_instances, num_instances):
					sorted_classes[jdx: jdx + num_random_instances] = sorted_classes[
							indices_toshuffle[i: min(len(y_dense), i + num_random_instances)]]
					i = i + num_random_instances

		# Shuffle bags
		sorted_classes = sorted_classes.reshape(int(len(y_dense) / num_instances), num_instances)
		np.random.shuffle(sorted_classes)
		indices = sorted_classes.flatten()

		# Generate new bag-level labels
		new_y = np.zeros((len(y), 1))
		for idx in range(0, len(y), num_instances):
				# Retrieve label for first image in bag
				max_occ = y_dense[indices[idx]]
				np.random.shuffle(indices[idx:idx + num_instances])	# shuffle images

				# Set "real" weak label
				new_y[indices[idx: idx + num_instances]] = max_occ

		# Perform majority vote to get estimated class label per cluster
		new_labels = np.zeros((num_clusters), dtype='int32')
		for a in range(0, num_clusters):
			lst = new_y[np.where(labels == a)[0]].squeeze().astype('int32')
			counts = np.bincount(lst, minlength=10)
			new_labels[a] = np.argmax(counts)


		y_c = to_categorical(new_y.flatten(), 10)
		for start_idx in range(0, len(new_y), batch_size):

			# Gather estimated class for current batch
			dist_cat = to_categorical(new_labels[labels[indices[start_idx: start_idx + batch_size]]], 10)

			yield [x[indices[start_idx: start_idx + batch_size]], dist_cat,
					y_c[indices[start_idx: start_idx + batch_size]]], y_c[indices[start_idx: start_idx + batch_size]]


def train_autoencoder(vae_model, vae_name, num_epochs=500):
	""" Train simple (variational) autoencoder

    Args:
      vae_model: "vae", "conv_vae" or "ae" depending on what type of model is to be trained (string)
      vae_name: location and name of model once trained (string)
      num_epochs: number of epochs for training (int)

    Returns:
      Trained Keras Model
    """
	print(vae_model.summary())

	print("Training VAE model...")
	start_time = time.time()
	(x_train, y_train), _ = mnist.load_data()
	x_train = x_train[:, np.newaxis, ...] / 255.
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

	vae_model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs)
	print("completed in {:.3f}s".format(time.time() - start_time))
	
	print("Trained variational autoencoder.")
	vae_model.save('./latent_models/mnist_ae_64f_500e.h5')

	return vae_model


def run_test_example(x_test, y_test, model):
	"""Simple predict function

    Returns:
      Test accuracy rate
    """
	new_model = Model(inputs=[model.get_layer('input_data').input],
							outputs=[model.get_layer('p_out').output])

	predictions = new_model.predict(x_test, batch_size=64)

	test_accuracy = float(np.sum(np.argmax(y_test, axis=1) == np.argmax(predictions, axis=1))) / len(y_test)
	print("test accuracy: ", test_accuracy)

	return test_accuracy


def get_latent_model(encoder_type, batch_size=100, latent_dim=FV_LENGTH):
	"""Calls function in latent.py depending on encoder_type - see train_autoencoder(...)
    """
	
	if encoder_type == 'vae':
		vae_model = latent.get_vae_model(batch_size, latent_dim=latent_dim)
		vae_model.load_weights('./latent_models/mnist_vae_dense.h5')
		vae_model.outputs = [vae_model.get_layer('z_mean').output]
		vae_model._make_predict_function()
		vmodel = vae_model
	    
	elif encoder_type == 'conv_vae':
		deepvae_model = latent.get_convvae_model()
		deepvae_model.load_weights('./latent_models/mnist_vae_conv.h5')
		deepvae_model.outputs = [deepvae_model.get_layer('z_mean').output]
		deepvae_model._make_predict_function(batch_size, latent_dim=latent_dim)
		vmodel = deepvae_model
	    
	else:
		ae_model = latent.get_ae_model(batch_size, latent_dim=latent_dim)
		ae_model.load_weights('./latent_models/mnist_ae.h5')
		ae_model.outputs = [ae_model.get_layer('z').output]
		ae_model._make_predict_function()
		vmodel = ae_model
	
	return vmodel
	

if __name__ == "__main__":

	batch_size = 100
	loss_type = "cluster_class"
	alpha = 0.5
	cluster_type = 'conv_vae'
	num_k_clusters = 100
	num_epochs = 5

	search_num_instances = [5, 25, 50, 100, 200]
	search_fraction = [0.5]

	start_time = time.time()
	mibl_model = get_model(loss_type, alpha)

	# presave weights so that we can set it back to original model after training (see below)
	old_weights = mibl_model.get_weights()

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_test = x_test[:, np.newaxis, ...] / 255.
	x_train = x_train[:, np.newaxis, ...] / 255.
	y_train = to_categorical(y_train, 10)
	y_test = to_categorical(y_test, 10)
    
   # train latent representation and save into predefined folder
	deepvae_model = latent.get_convvae_model(latent_dim=FV_LENGTH)
	#train_autoencoder(deepvae_model, './latent_models/mnist_convvae_64f_500e.h5')		

	# throw away the decoder part of the network
	deepvae_model.load_weights('./latent_models/mnist_convvae_64f_500e.h5')
	deepvae_model.outputs = [deepvae_model.get_layer('z_mean').output, deepvae_model.get_layer('z_log').output]
	deepvae_model._make_predict_function()

	print("Training CNN...")
	for s in search_num_instances:
		for f in search_fraction:
			
			test_accuracy = []

			# repeat same experiment 10 times to meaure variability
			for r in range(10):
				ni = s
				fraction = f

				mibl_model.compile(loss={'joint_loss': lambda y_true, y_pred: y_pred}, optimizer="adam")

				# Train - MNIST-BAG prepared in generator
				mibl_model.fit_generator(
					minibatch_mibl_gen(x_train, y_train, batch_size, deepvae_model,
											ni, num_clusters=num_k_clusters, fraction_class=f),
					steps_per_epoch=len(y_train) / batch_size,
					epochs=num_epochs)

				acc = run_test_example(x_test, y_test, mibl_model)
				test_accuracy.append(acc)
				mibl_model.set_weights(old_weights)

			print(">> n = " + str(s) + ", a = " + str(alpha) + ", fraction = " + str(f))
			print("mean accuracy: {}, std accuracy: {}".format(np.mean(np.array(test_accuracy)), np.std(np.array(test_accuracy)))) 

# mibl_model.save("./mnist_model.h5")
