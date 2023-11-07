####################################################################################
#
# exmining_twins_and_supermag/non_twins_modeling_v0.py
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
import math
import os
import pickle
import subprocess
import time

import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from scipy.special import expit, inv_boxcox
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from spacepy import pycdf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import get_session

import utils
# from data_prep import DataPrep

# from datetime import strftime



os.environ["CDF_LIB"] = "~/CDF/lib"

data_directory = '../../../../data/'
supermag_dir = '../data/supermag/feather_files/'
regions_dict = 'mike_working_dir/identifying_regions_data/identifying_regions_data/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = 'mike_working_dir/identifying_regions_data/identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'
working_dir = data_directory+'mike_working_dir/twins_data_modeling/'

random_seed = 42


# loading config and specific model config files. Using them as dictonaries
# with open('twins_config.json', 'r') as con:
# 	CONFIG = json.load(con)

# with open('model_config.json', 'r') as mcon:
# 	MODEL_CONFIG = json.load(mcon)

CONFIG = {'region_numbers': [194, 270, 287, 207, 62, 241, 366, 387, 223, 19, 163],
			'load_twins':False,
			'mag_features':[],
			'solarwind_features':[],
			'delay':False,
			'rolling':False,
			'to_drop':[],
			'omni_or_ace':'omni',
			'time_history':30,
			'random_seed':42}

MODEL_CONFIG = {'filters':128,
				'initial_learning_rate':1e-6,
				'epochs':500,
				'loss':'mse',
				'early_stop_patience':25}


region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]

TARGET = 'rsd'
VERSION = 0


def loading_data(target_var, region, percentile=0.99):

	# loading all the datasets and dictonaries

	regions, stats = utils.loading_dicts()
	solarwind = utils.loading_solarwind(omni=True, limit_to_twins=True)

	with open('outputs/feature_engineering/solarwind_corr_dict.pkl', 'rb') as f:
		solarwind_corr_dict = pickle.load(f)
	with open('outputs/feature_engineering/mag_corr_dict.pkl', 'rb') as f:
		supermag_corr_dict = pickle.load(f)

	# converting the solarwind data to log10
	solarwind['logT'] = np.log10(solarwind['T'])
	solarwind.drop(columns=['T'], inplace=True)

	# reduce the regions dict to be only the ones that have keys in the region_numbers list
	regions = regions[f'region_{region}']
	stats = stats[f'region_{region}']

	# getting dbdt and rsd data for the region
	supermag_df = utils.combining_stations_into_regions(regions['station'], stats, features=['dbht', 'MAGNITUDE', 'theta', 'N', 'E'], mean=True, std=True, maximum=True, median=True)

	# getting the mean latitude for the region and attaching it to the regions dictionary
	mean_lat = utils.getting_mean_lat(regions['station'])

	threshold = supermag_df[target_var].quantile(percentile)

	supermag_df.drop(columns=supermag_corr_dict[f'region_{region}']['twins_corr'], inplace=True)
	solarwind.drop(columns=solarwind_corr_dict[f'region_{region}']['twins_corr_features'], inplace=True)

	merged_df = pd.merge(supermag_df, solarwind, left_index=True, right_index=True, how='inner')


	return merged_df, mean_lat, threshold



def getting_prepared_data(target_var, region, get_features=False):
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

	merged_df, mean_lat, threshold = loading_data(target_var=target_var, region=region, percentile=0.99)

	merged_df = utils.classification_column(merged_df, param=f'rolling_{target_var}', thresh=threshold, forecast=0, window=0)

	# target = merged_df['classification']
	target = merged_df[f'rolling_{target_var}']

	# removing the target var from the dataframe

	vars_to_drop = [target_var]

	if 'MLT' in merged_df.columns:
		vars_to_drop.append('MLT')
	if 'theta_max' in merged_df.columns:
		vars_to_drop.append('theta_max')
	if 'classification' in merged_df.columns:
		vars_to_drop.append('classification')

	merged_df.drop(columns=vars_to_drop, inplace=True)
	# merged_df.dropna(subset=[f'rolling_{target_var}'], inplace=True)

	print('Columns in Merged Dataframe: '+str(merged_df.columns))

	print(f'Target value positive percentage: {target.sum()/len(target)}')
	# merged_df.drop(columns=[f'rolling_{target_var}', 'classification'], inplace=True)

	if os.path.exists(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_version_{VERSION}.pkl'):
		with open(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_version_{VERSION}.pkl', 'rb') as f:
			storms_extracted_dict = pickle.load(f)
		storms = storms_extracted_dict['storms']
		target = storms_extracted_dict['target']

	else:
	# getting the data corresponding to the twins maps
		storms, target = utils.storm_extract(df=merged_df, lead=30, recovery=9, twins=True, target=True, target_var=f'rolling_{target_var}', concat=False)
		storms_extracted_dict = {'storms':storms, 'target':target}
		with open(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_version_{VERSION}.pkl', 'wb') as f:
			pickle.dump(storms_extracted_dict, f)

	print('Columns in Dataframe: '+str(storms[0].columns))
	features = storms[0].columns

	# splitting the data on a month to month basis to reduce data leakage
	month_df = pd.date_range(start=pd.to_datetime('2009-07-01'), end=pd.to_datetime('2017-12-01'), freq='MS')

	train_months, test_months = train_test_split(month_df, test_size=0.2, shuffle=True, random_state=CONFIG['random_seed'])
	train_months, val_months = train_test_split(train_months, test_size=0.125, shuffle=True, random_state=CONFIG['random_seed'])

	train_dates_df, val_dates_df, test_dates_df = pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]})
	x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []

	# using the months to split the data
	for month in train_months:
		train_dates_df = pd.concat([train_dates_df, pd.DataFrame({'dates':pd.date_range(start=month, end=month+pd.DateOffset(months=1), freq='min')})], axis=0)

	for month in val_months:
		val_dates_df = pd.concat([val_dates_df, pd.DataFrame({'dates':pd.date_range(start=month, end=month+pd.DateOffset(months=1), freq='min')})], axis=0)

	for month in test_months:
		test_dates_df = pd.concat([test_dates_df, pd.DataFrame({'dates':pd.date_range(start=month, end=month+pd.DateOffset(months=1), freq='min')})], axis=0)

	train_dates_df.set_index('dates', inplace=True)
	val_dates_df.set_index('dates', inplace=True)
	test_dates_df.set_index('dates', inplace=True)

	train_dates_df.index = pd.to_datetime(train_dates_df.index)
	val_dates_df.index = pd.to_datetime(val_dates_df.index)
	test_dates_df.index = pd.to_datetime(test_dates_df.index)

	date_dict = {'train':pd.DataFrame(), 'val':pd.DataFrame(), 'test':pd.DataFrame()}

	# getting the data corresponding to the dates
	for storm, y in zip(storms, target):

		copied_storm = storm.copy()
		copied_storm = copied_storm.reset_index(inplace=False, drop=False).rename(columns={'index':'Date_UTC'})

		if storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in train_dates_df.index:
			x_train.append(storm)
			y_train.append(y)
			date_dict['train'] = pd.concat([date_dict['train'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in val_dates_df.index:
			x_val.append(storm)
			y_val.append(y)
			date_dict['val'] = pd.concat([date_dict['val'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in test_dates_df.index:
			x_test.append(storm)
			y_test.append(y)
			date_dict['test'] = pd.concat([date_dict['test'], copied_storm['Date_UTC'][-10:]], axis=0)

	date_dict['train'].reset_index(drop=True, inplace=True)
	date_dict['val'].reset_index(drop=True, inplace=True)
	date_dict['test'].reset_index(drop=True, inplace=True)

	date_dict['train'].rename(columns={date_dict['train'].columns[0]:'Date_UTC'}, inplace=True)
	date_dict['val'].rename(columns={date_dict['val'].columns[0]:'Date_UTC'}, inplace=True)
	date_dict['test'].rename(columns={date_dict['test'].columns[0]:'Date_UTC'}, inplace=True)

	to_scale_with = pd.concat(x_train, axis=0)
	scaler = StandardScaler()
	scaler.fit(to_scale_with)
	x_train = [scaler.transform(x) for x in x_train]
	x_val = [scaler.transform(x) for x in x_val]
	x_test = [scaler.transform(x) for x in x_test]

	# splitting the sequences for input to the CNN
	x_train, y_train, train_dates_to_drop = utils.split_sequences(x_train, y_train, n_steps=CONFIG['time_history'], dates=date_dict['train'], model_type='regression')
	x_val, y_val, val_dates_to_drop = utils.split_sequences(x_val, y_val, n_steps=CONFIG['time_history'], dates=date_dict['val'], model_type='regression')
	x_test, y_test, test_dates_to_drop  = utils.split_sequences(x_test, y_test, n_steps=CONFIG['time_history'], dates=date_dict['test'], model_type='regression')

	# dropping the dates that correspond to arrays that would have had nan values
	date_dict['train'].drop(train_dates_to_drop, axis=0, inplace=True)
	date_dict['val'].drop(val_dates_to_drop, axis=0, inplace=True)
	date_dict['test'].drop(test_dates_to_drop, axis=0, inplace=True)

	date_dict['train'].reset_index(drop=True, inplace=True)
	date_dict['val'].reset_index(drop=True, inplace=True)
	date_dict['test'].reset_index(drop=True, inplace=True)

	if not get_features:
		return x_train, x_val, x_test, y_train, y_val, y_test, date_dict
	else:
		return x_train, x_val, x_test, y_train, y_val, y_test, date_dict, features


def CRPS(y_true, y_pred):
	'''
	Defining the CRPS loss function for model training.

	Args:
		y_true (np.array): true values
		y_pred (np.array): predicted values

	Returns:
		float: CRPS value
	'''
	mean, std = tf.unstack(y_pred, axis=-1)
	y_true = tf.unstack(y_true, axis=-1)

	# making the arrays the right dimensions
	mean = tf.expand_dims(mean, -1)
	std = tf.expand_dims(std, -1)
	y_true = tf.expand_dims(y_true, -1)

	# calculating the error

	crps = tf.math.reduce_mean(calculate_crps(epsilon_error(y_true, mean), std))

	return crps


def epsilon_error(y, u):

	epsilon = tf.math.abs(y-u)

	return epsilon


def calculate_crps(epsilon, sig):

	crps = sig * ((epsilon/sig) * tf.math.erf((epsilon/(np.sqrt(2)*sig))) + tf.math.sqrt(2/np.pi) * tf.math.exp(-epsilon**2/(2*sig**2)) - 1/tf.math.sqrt(np.pi))

	return crps


def full_model(input_shape, loss='mse', early_stop_patience=20, initial_filters=32, learning_rate=1e-05):
	'''
	Concatenating the CNN models together with the MLT input

	Args:
		loss (str, optional): loss function to be uesd for training. Defaults to 'categorical_crossentropy'.
		early_stop_patience (int, optional): number of epochs the model will continue training once there
												is no longer val loss improvements. Defaults to 3.

	Returns:
		object: model configuration ready for training
		object: early stopping conditions
	'''

	# CNN model
	inputs = Input(shape=(input_shape[1], input_shape[2], 1))
	conv1 = Conv2D(initial_filters, 5, padding='same', activation='relu')(inputs)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(2)(conv1)
	conv2 = Conv2D(initial_filters*2, 3, padding='same', activation='relu')(pool1)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(2)(conv2)
	conv3 = Conv2D(initial_filters*4, 2, padding='same', activation='relu')(pool2)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(2)(conv3)
	conv4 = Conv2D(initial_filters*4, 2, padding='same', activation='relu')(pool3)
	conv4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(2)(conv4)
	flat = Flatten()(pool4)

	# MLT input
	mlt_input = Input(shape=(2,))
	mlt_dense = Dense(32, activation='relu')(mlt_input)

	# combining the two
	combined = concatenate([flat, mlt_dense])
	dense1 = Dense(initial_filters*4, activation='relu')(combined)
	dense1 = BatchNormalization()(dense1)
	drop1 = Dropout(0.2)(dense1)
	dense2 = Dense(initial_filters*2, activation='relu')(drop1)
	dense2 = BatchNormalization()(dense2)
	drop2 = Dropout(0.2)(dense2)
	dense3 = Dense(initial_filters, activation='relu')(drop2)
	dense3 = BatchNormalization()(dense3)
	drop3 = Dropout(0.2)(dense3)
	output = Dense(1, activation='linear')(drop3)

	model = Model(inputs=[inputs, mlt_input], outputs=output)

	opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=loss)					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting

	return model, early_stop



def fit_full_model(model, xtrain, xval, ytrain, yval, train_mlt, val_mlt, early_stop, CV, delay, first_time=True):
	'''
	Performs the actual fitting of the model.

	Args:
		model (keras model): model as defined in the create_model function.

		xtrain (3D np.array): training data inputs
		xval (3D np.array): validation inputs
		ytrain (2D np.array): training target vectors
		yval (2D np.array): validation target vectors
		early_stop (keras early stopping dict): predefined early stopping function
		split (int): split being trained. Used for saving model.
		station (str): station being trained.
		first_time (bool, optional): if True model will be trainined, False model will be loaded. Defaults to True.

	Returns:
		model: fit model ready for making predictions.
	'''

	if first_time:

		# reshaping the model input vectors for a single channel
		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))

		print(f'XTrain: {np.isnan(Xtrain).sum()}')
		print(f'XVal: {np.isnan(Xval).sum()}')
		print(f'yTrain: {np.isnan(ytrain).sum()}')
		print(f'yVal: {np.isnan(yval).sum()}')
		print(f'train_mlt: {np.isnan(train_mlt).sum()}')
		print(f'val_mlt: {np.isnan(val_mlt).sum()}')

		model.fit(x=[Xtrain, train_mlt], y=ytrain,
					validation_data=([Xval, val_mlt], yval),
					verbose=1, shuffle=True, epochs=500,
					callbacks=[early_stop], batch_size=64)			# doing the training! Yay!

		# saving the model
		model.save(f'models/delay_{delay}/CV_{CV}.h5')

	if not first_time:

		# loading the model if it has already been trained.
		model = load_model(f'models/delay_{delay}/CV_{CV}.h5')				# loading the models if already trained

	return model


def making_predictions(model, Xtest, test_mlt, CV, boxcox_mean):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test_dict (dict): dictonary with the testing model inputs and the real data for comparison
		split (int): which split is being tested

	Returns:
		dict: test dict now containing columns in the dataframe with the model predictions for this split
	'''


	Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1))			# reshpaing for one channel input

	predicted = model.predict([Xtest, test_mlt], verbose=1)						# predicting on the testing input data
	predicted = tf.gather(predicted, 0, axis=1)					# grabbing the positive node
	predicted = predicted.numpy()									# turning to a numpy array
	# predicted = predicted + boxcox_mean
	# predicted = inv_boxcox(predicted, 0)

	return predicted