import time
import os
import numpy as np
import random
import math

import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

from keras.models import Model, load_model
from keras.layers import Input, Lambda, Dense, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, \
	BatchNormalization, Conv2DTranspose, Reshape, UpSampling2D, Flatten
from keras.models import load_model
from keras.utils import to_categorical

from keras import backend as K
from keras.objectives import binary_crossentropy

K.set_image_data_format('channels_first')
K.set_session(sess)

	
def get_vae_model(batch_size=100, original_dim=784, intermediate_dim = [500, 500, 2000], latent_dim=10):

	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
		return z_mean + K.exp(z_log_var) * epsilon
		
	x = Input(batch_shape=(batch_size, original_dim))
	h = Dense(intermediate_dim[0], activation='relu')(x)
	h = Dense(intermediate_dim[1], activation='relu')(h)
	h = Dense(intermediate_dim[2], activation='relu')(h)
	z_mean = Dense(latent_dim, name='z_mean')(h)
	z_log_var = Dense(latent_dim, name='z_log')(h)
	z = Lambda(sampling, output_shape=(latent_dim,), name='encoder')([z_mean, z_log_var])
	h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
	h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
	h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
	x_decoded_mean = Dense(original_dim, activation='sigmoid', name='reconstruction')(h_decoded)

	# Define model
	def vae_loss(x, x_decoded_mean):
		xent_loss = 28 * 28 * binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean))
		kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return xent_loss + kl_loss

	vae_model = Model(x, x_decoded_mean)
	vae_model.compile(optimizer='adam', loss=vae_loss)

	return vae_model
	
	
def get_ae_model(batch_size=100, original_dim=784, intermediate_dim = [500, 500, 2000], latent_dim=10):
	x = Input(batch_shape=(batch_size, original_dim))
	h = Dense(intermediate_dim[0], activation='relu')(x)
	h = Dense(intermediate_dim[1], activation='relu')(h)
	h = Dense(intermediate_dim[2], activation='relu')(h)
	z = Dense(latent_dim, name='z')(h)
	h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
	h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
	h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
	x_decoded_mean = Dense(original_dim, activation='sigmoid', name='reconstruction')(h_decoded)

	ae_model = Model(x, x_decoded_mean)
	ae_model.compile(optimizer='adam', loss='binary_crossentropy')

	return ae_model
	

def get_convvae_model(batch_size=100, latent_dim=10, original_dim=784):
	# Build the autoencoder model
	#input_img = Input(shape=(1, 28, 28))
	filter_kernel_size = (3, 3)

	input_img_original = Input(batch_shape=(batch_size, original_dim))
	input_img = Reshape((1, 28, 28))(input_img_original)
	x = Conv2D(32, filter_kernel_size, activation='relu', padding='same')(input_img)
	x = MaxPool2D((2, 2), padding='same')(x)
	x = Conv2D(32, filter_kernel_size, activation='relu', padding='same')(x)
	x = MaxPool2D((2, 2), padding='same')(x)
	x = Flatten()(x)
	encoder = Dense(512, activation='relu')(x)

	z_mean = Dense(latent_dim, name='z_mean')(encoder)
	z_log_var = Dense(latent_dim, name='z_log')(encoder)

	def vae_loss(x, x_decoded_mean):
		xent_loss = binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean))
		kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
		return xent_loss + kl_loss

	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
		return z_mean + K.exp(z_log_var) * epsilon

	z = Lambda(sampling, name="encoder")([z_mean, z_log_var])

	x = Dense(512, activation='relu')(z)
	x = Dense(32 * 7 * 7, activation='relu')(x)
	x = Reshape((32, 7, 7))(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2DTranspose(32, filter_kernel_size, activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
	decoded = Flatten()(decoded)

	# Define model
	def vae_loss(x, x_decoded_mean):
		xent_loss = 28 * 28 * binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean))
		kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return xent_loss + kl_loss

	vae_model = Model(input_img_original, decoded)
	vae_model.compile(optimizer='adam', loss=vae_loss)

	return vae_model
	

	