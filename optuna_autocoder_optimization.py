####################################################################################
#
# exmining_twins_and_supermag/modeling_v0.py
#
# Performing the modeling using the Solar Wind and Ground Magnetomoeter data.
# TWINS data passes through a pre-trained autoencoder that reduces the TWINS maps
# to a reuced dimensionality. This data is then concatenated onto the model after
# both branches of the CNN hae been flattened, and before the dense layers.
# Similar model to Coughlan (2023) but with a different target variable.
#
####################################################################################


# Importing the libraries
import datetime
import gc
import glob
import json
import os
import pickle
import subprocess
import time

import keras
import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shapely
import tensorflow as tf
import tqdm
from optuna_dashboard import run_server
from scipy.special import expit, inv_boxcox
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from tensorflow.keras.backend import clear_session, int_shape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout, Flatten,
                                     Input, MaxPooling2D, Reshape, concatenate)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.backend import get_session

from data_generator import Generator
from data_prep import DataPrep

os.environ["CDF_LIB"] = "~/CDF/lib"

region_path = '../identifying_regions/outputs/adjusted_regions.pkl'
region_number = '163'
solarwind_path = '../data/SW/omniData.feather'
supermag_dir_path = '../data/supermag/'
twins_times_path = 'outputs/regular_twins_map_dates.feather'
rsd_path = '../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'
random_seed = 42


# loading config and specific model config files. Using them as dictonaries
with open('twins_config.json', 'r') as con:
	CONFIG = json.load(con)


def getting_prepared_data():
	'''
	Calling the data prep class without the TWINS data for this version of the model.

	Returns:
		X_train (np.array): training inputs for the model
		X_val (np.array): validation inputs for the model
		X_test (np.array): testing inputs for the model
		y_train (np.array): training targets for the model
		y_val (np.array): validation targets for the model
		y_test (np.array): testing targets for the model

	'''

	prep = DataPrep(region_path, region_number, solarwind_path, supermag_dir_path, twins_times_path,
					rsd_path, random_seed)

	train, val, test  = prep.twins_only_data_prep(CONFIG)

	train = train[:(int(len(train)*0.1)),:,:]
	val = val[:(int(len(val)*0.1)),:,:]

	print(train.shape)
	print(val.shape)

	# reshaping the model input vectors for a single channel
	train = train.reshape((train.shape[0], train.shape[1], train.shape[2], 1))
	val = val.reshape((val.shape[0], val.shape[1], val.shape[2], 1))
	test = test.reshape((test.shape[0], test.shape[1], test.shape[2], 1))

	input_shape = (train.shape[1], train.shape[2], 1)

	# train = Generator(train, train, batch_size=16, shuffle=True)
	# val = Generator(val, val, batch_size=16, shuffle=True)

	return train, val, test, input_shape


def Autoencoder(input_shape, trial, early_stopping_patience=10):


	initial_filters = trial.suggest_categorical('initial_filters', [32, 64, 128, 256])
	latent_dim = trial.suggest_int('latent_dim', 16, 128)
	learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-2)
	layers = trial.suggest_int('layers', 2, 5)
	activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
	loss = trial.suggest_categorical('loss', ['mse', 'binary_crossentropy'])


	model_input = Input(shape=input_shape, name='encoder_input')
	filters = initial_filters

	for i in range(layers):

		if i == 0:
			e = Conv2D(filters=filters, kernel_size=3, activation=activation, strides=1, padding='same')(model_input)
		elif i == (layers-1):
			e = Conv2D(filters=filters, kernel_size=2, activation=activation, strides=2, padding='same')(e)
		else:
			e = Conv2D(filters=filters, kernel_size=3, activation=activation, strides=1, padding='same')(e)

		filters = (filters*2)

	shape = int_shape(e)

	e = Flatten()(e)

	bottleneck = Dense(latent_dim, name='bottleneck')(e)

	d = Dense(shape[1]*shape[2]*shape[3])(bottleneck)

	d = Reshape((shape[1], shape[2], shape[3]))(d)

	for i in range(layers):
		if i == 0:
			d = Conv2DTranspose(filters=filters, kernel_size=2, activation=activation, strides=2, padding='same')(d)
		else:
			d = Conv2DTranspose(filters=filters, kernel_size=3, activation=activation, strides=1, padding='same')(d)

		filters = int(filters/2)

	model_outputs = Conv2DTranspose(filters=1, kernel_size=1, activation='linear', padding='same', name='decoder_output')(d)

	full_autoencoder = Model(inputs=model_input, outputs=model_outputs)

	opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)		# learning rate that actually started producing good results
	full_autoencoder.compile(optimizer=opt, loss=loss)					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)		# early stop process prevents overfitting

	return full_autoencoder, early_stop


def objective(trial, train, val, test, input_shape):

	model, early_stop = Autoencoder(input_shape, trial)
	print(model.summary())
	model.fit(train, train, validation_data=(val, val),
				verbose=1, shuffle=True, epochs=500,
				callbacks=[early_stop], batch_size=16)			# doing the training! Yay!

	return model.evaluate(test, test, verbose=1)


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	train, val, test, input_shape = getting_prepared_data()

	storage = optuna.storages.InMemoryStorage()
	# reshaping the model input vectors for a single channel
	study = optuna.create_study(direction='minimize', study_name='autoencoder_optimization_trial')
	study.optimize(lambda trial: objective(trial, train, val, test, input_shape), n_trials=100)
	print(study.best_params)

	optuna.visualization.plot_optimization_history(study).write_image('plots/optimization_history.png')

	optuna.visualization.plot_param_importances(study).write_image('plots/param_importances.png')

	best_model, ___ = Autoencoder(input_shape, study.best_params)

	best_model.evaluate(test, test)

	best_model.save('models/best_autoencoder.h5')

	run_server(storage)





if __name__ == '__main__':
	main()
	print('It ran. God job!')
