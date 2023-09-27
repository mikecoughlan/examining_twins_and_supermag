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
import pandas as pd
import shapely
import tensorflow as tf
import tqdm
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

	# train = train[:(int(len(train)*0.001)),:,:]
	# val = val[:(int(len(val)*0.001)),:,:]

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


def Autoencoder(input_shape, train, val, early_stopping_patience=10):


	model_input = Input(shape=input_shape, name='encoder_input')

	e = Conv2D(filters=32, kernel_size=3, activation='relu', strides=1, padding='same')(model_input)
	e = Conv2D(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')(e)
	e = Conv2D(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(e)
	e = Conv2D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same')(e)
	e = Conv2D(filters=512, kernel_size=3, activation='relu', strides=1, padding='same')(e)

	shape = int_shape(e)

	e = Flatten()(e)

	bottleneck = Dense(64, name='bottleneck')(e)

	d = Dense(shape[1]*shape[2]*shape[3])(bottleneck)

	d = Reshape((shape[1], shape[2], shape[3]))(d)

	d = Conv2DTranspose(filters=512, kernel_size=3, activation='relu', strides=1, padding='same')(d)
	d = Conv2DTranspose(filters=256, kernel_size=3, activation='relu', strides=1, padding='same')(d)
	d = Conv2DTranspose(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(d)
	d = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')(d)
	d = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', strides=1, padding='same')(d)

	model_outputs = Conv2DTranspose(filters=1, kernel_size=1, activation='linear', padding='same', name='decoder_output')(d)

	full_autoencoder = Model(inputs=model_input, outputs=model_outputs)

	opt = tf.keras.optimizers.Adam(learning_rate=1e-6)		# learning rate that actually started producing good results
	full_autoencoder.compile(optimizer=opt, loss='mse')					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)		# early stop process prevents overfitting

	full_autoencoder = fit_autoencoder(full_autoencoder, train, val, early_stop)

	encoder = Model(inputs=model_input, outputs = bottleneck)

	return full_autoencoder, encoder


def fit_autoencoder(model, train, val, early_stop):
	'''

	'''

	if not os.path.exists('models/autoencoder_v3.h5'):

		# # reshaping the model input vectors for a single channel
		# train = train.reshape((train.shape[0], train.shape[1], train.shape[2], 1))
		# val = val.reshape((val.shape[0], val.shape[1], val.shape[2], 1))

		print(model.summary())

		model.fit(train, train, validation_data=(val, val),
					verbose=1, shuffle=True, epochs=500, callbacks=[early_stop], batch_size=16)			# doing the training! Yay!

		# saving the model
		model.save('models/autoencoder_v3.h5')

	else:
		# loading the model if it has already been trained.
		model = load_model('models/autoencoder_v3.h5')				# loading the models if already trained
		print(model.summary())

	return model


def making_predictions(model, test):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test_dict (dict): dictonary with the testing model inputs and the real data for comparison
		split (int): which split is being tested

	Returns:
		dict: test dict now containing columns in the dataframe with the model predictions for this split
	'''

	test = test.reshape((test.shape[0], test.shape[1], test.shape[2], 1))			# reshpaing for one channel input
	print('Test input Nans: '+str(np.isnan(test).sum()))

	predicted = model.predict(test, verbose=1)						# predicting on the testing input data
	# predicted = predicted.numpy()									# turning to a numpy array

	return predicted



def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	train, val, test, input_shape = getting_prepared_data()

	train = train[:(int(len(train)*0.5)),:,:]
	val = val[:(int(len(val)*0.5)),:,:]

	# creating the model
	print('Initalizing model...')
	autoencoder, encoder = Autoencoder(input_shape, train, val)

	# # fitting the model
	# print('Fitting model...')
	# MODEL = fit_autoencoder(MODEL, train, val, early_stop)

	# making predictions
	print('Making predictions...')
	predictions = making_predictions(autoencoder, test)

	# rmse = np.sqrt(mean_squared_error(test, predictions[:,:,:,0]))
	# print(f'RMSE: {rmse}')

	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[15, :, :, 0])
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[15, :, :])
	ax2.set_title('Actual')
	plt.show()


	# encoder = Model(inputs=MODEL.inputs, outputs=MODEL.bottleneck)
	encoder.save('models/encoder_v3.h5')

	# # saving the results
	# print('Saving results...')
	# results_df.to_feather('outputs/non_twins_results.feather')

	# # calculating some metrics
	# print('Calculating metrics...')
	# metrics = calculate_some_metrics(results_df)

	# # saving the metrics
	# print('Saving metrics...')
	# metrics.to_feather('outputs/non_twins_metrics.feather')



if __name__ == '__main__':
	main()
	print('It ran. God job!')
