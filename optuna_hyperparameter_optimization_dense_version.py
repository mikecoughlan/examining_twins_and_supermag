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
# import plotly
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

import utils
from data_generator import Generator
from data_prep import DataPrep

CONFIG = {'time_history':30, 'random_seed':7}

EVALUATION_DICT = {}

os.environ["CDF_LIB"] = "~/CDF/lib"

working_dir = '../../../../data/mike_working_dir/'
region_path = working_dir+'identifying_regions_data/adjusted_regions.pkl'
region_number = 163
solarwind_path = '../data/SW/omniData.feather'
supermag_dir_path = '../data/supermag/'
twins_times_path = 'outputs/regular_twins_map_dates.feather'
rsd_path = working_dir+'identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'
random_seed = 7

VERSION = 'optimizing_dense'


def loading_data(target_var, region):

	# loading all the datasets and dictonaries

	regions, stats = utils.loading_dicts()
	solarwind = utils.loading_solarwind(omni=True, limit_to_twins=True)

	# converting the solarwind data to log10
	solarwind['logT'] = np.log10(solarwind['T'])
	solarwind.drop(columns=['T'], inplace=True)

	# reduce the regions dict to be only the ones that have keys in the region_numbers list
	regions = regions[f'region_{region}']
	stats = stats[f'region_{region}']

	# getting dbdt and rsd data for the region
	supermag_df = utils.combining_stations_into_regions(regions['station'], stats, features=['dbht', 'MAGNITUDE', \
		'theta', 'N', 'E', 'sin_theta', 'cos_theta'], mean=True, std=True, maximum=True, median=True)

	# getting the mean latitude for the region and attaching it to the regions dictionary
	mean_lat = utils.getting_mean_lat(regions['station'])

	merged_df = pd.merge(supermag_df, solarwind, left_index=True, right_index=True, how='inner')

	print('Loading TWINS maps....')
	maps = utils.loading_twins_maps()

	return merged_df, mean_lat, maps


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

	merged_df, mean_lat, maps = loading_data(target_var=target_var, region=region)

	# target = merged_df['classification']
	target = merged_df[f'rolling_{target_var}']

	# reducing the dataframe to only the features that will be used in the model plus the target variable
	vars_to_keep = [f'rolling_{target_var}', 'dbht_median', 'MAGNITUDE_median', 'MAGNITUDE_std', 'sin_theta_std', 'cos_theta_std', 'cosMLT', 'sinMLT',
					'B_Total', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'proton_density', 'logT']
	merged_df = merged_df[vars_to_keep]

	print('Columns in Merged Dataframe: '+str(merged_df.columns))

	print(f'Target value positive percentage: {target.sum()/len(target)}')
	# merged_df.drop(columns=[f'rolling_{target_var}', 'classification'], inplace=True)

	temp_version = 'final_1'

	if os.path.exists(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl'):
		with open(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl', 'rb') as f:
			storms_extracted_dict = pickle.load(f)
		storms = storms_extracted_dict['storms']
		target = storms_extracted_dict['target']

	else:
		# getting the data corresponding to the twins maps
		storms, target = utils.storm_extract(df=merged_df, lead=30, recovery=9, twins=True, target=True, target_var=f'rolling_{target_var}', concat=False, map_keys=maps.keys())
		storms_extracted_dict = {'storms':storms, 'target':target}
		with open(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl', 'wb') as f:
			pickle.dump(storms_extracted_dict, f)

	print('Columns in Dataframe: '+str(storms[0].columns))
	features = storms[0].columns

	# splitting the data on a month to month basis to reduce data leakage
	month_df = pd.date_range(start=pd.to_datetime('2009-07-01'), end=pd.to_datetime('2017-12-01'), freq='MS')
	month_df = month_df.drop([pd.to_datetime('2012-03-01'), pd.to_datetime('2017-09-01')])

	train_months, test_months = train_test_split(month_df, test_size=0.1, shuffle=True, random_state=CONFIG['random_seed'])
	train_months, val_months = train_test_split(train_months, test_size=0.125, shuffle=True, random_state=CONFIG['random_seed'])

	test_months = test_months.tolist()
	# adding the two dateimte values of interest to the test months df
	test_months.append(pd.to_datetime('2012-03-01'))
	test_months.append(pd.to_datetime('2017-09-01'))
	test_months = pd.to_datetime(test_months)

	train_dates_df, val_dates_df, test_dates_df = pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]})
	x_train, x_val, x_test, y_train, y_val, y_test, twins_train, twins_val, twins_test = [], [], [], [], [], [], [], [], []

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

	print(f'Size of the training storms: {len(storms)}')
	print(f'Size of the training target: {len(target)}')
	print(f'Size of the twins maps: {len(maps)}')

	# getting the data corresponding to the dates
	for storm, y, twins_map in zip(storms, target, maps):

		copied_storm = storm.copy()
		copied_storm = copied_storm.reset_index(inplace=False, drop=False).rename(columns={'index':'Date_UTC'})

		if storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in train_dates_df.index:
			x_train.append(storm)
			y_train.append(y)
			twins_train.append(maps[twins_map]['map'])
			date_dict['train'] = pd.concat([date_dict['train'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in val_dates_df.index:
			x_val.append(storm)
			y_val.append(y)
			twins_val.append(maps[twins_map]['map'])
			date_dict['val'] = pd.concat([date_dict['val'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in test_dates_df.index:
			x_test.append(storm)
			y_test.append(y)
			twins_test.append(maps[twins_map]['map'])
			date_dict['test'] = pd.concat([date_dict['test'], copied_storm['Date_UTC'][-10:]], axis=0)

	date_dict['train'].reset_index(drop=True, inplace=True)
	date_dict['val'].reset_index(drop=True, inplace=True)
	date_dict['test'].reset_index(drop=True, inplace=True)

	date_dict['train'].rename(columns={date_dict['train'].columns[0]:'Date_UTC'}, inplace=True)
	date_dict['val'].rename(columns={date_dict['val'].columns[0]:'Date_UTC'}, inplace=True)
	date_dict['test'].rename(columns={date_dict['test'].columns[0]:'Date_UTC'}, inplace=True)

	# scaling the solar wind and mag data
	to_scale_with = pd.concat(x_train, axis=0)
	scaler = StandardScaler()
	scaler.fit(to_scale_with)
	x_train = [scaler.transform(x) for x in x_train]
	x_val = [scaler.transform(x) for x in x_val]
	x_test = [scaler.transform(x) for x in x_test]

	# scaling the twins maps
	twins_scaling_array = np.vstack(twins_train)
	twins_scaler = MinMaxScaler()
	twins_scaler.fit(twins_scaling_array)
	twins_train = np.array([twins_scaler.transform(x) for x in twins_train])
	twins_val = np.array([twins_scaler.transform(x) for x in twins_val])
	twins_test = np.array([twins_scaler.transform(x) for x in twins_test])

	# saving the scalers
	with open(f'models/{TARGET}/twins_region_{region}_version_{VERSION}_scaler.pkl', 'wb') as f:
		pickle.dump({'mag_and_solarwind':scaler, 'twins':twins_scaler}, f)

	# splitting the sequences for input to the CNN
	x_train, y_train, train_dates_to_drop, twins_train = utils.split_sequences(x_train, y_train, n_steps=CONFIG['time_history'], dates=date_dict['train'], model_type='regression', maps=twins_train)
	x_val, y_val, val_dates_to_drop, twins_val = utils.split_sequences(x_val, y_val, n_steps=CONFIG['time_history'], dates=date_dict['val'], model_type='regression', maps=twins_val)
	x_test, y_test, test_dates_to_drop, twins_test = utils.split_sequences(x_test, y_test, n_steps=CONFIG['time_history'], dates=date_dict['test'], model_type='regression', maps=twins_test)

	# dropping the dates that correspond to arrays that would have had nan values
	date_dict['train'].drop(train_dates_to_drop, axis=0, inplace=True)
	date_dict['val'].drop(val_dates_to_drop, axis=0, inplace=True)
	date_dict['test'].drop(test_dates_to_drop, axis=0, inplace=True)

	date_dict['train'].reset_index(drop=True, inplace=True)
	date_dict['val'].reset_index(drop=True, inplace=True)
	date_dict['test'].reset_index(drop=True, inplace=True)

	# reshaping the data to be 4D for the CNN
	x_train = x_train.reshape((x_train.shape[0], xtrain.shape[1], x_train.shape[2], 1))
	x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

	# reshaping the twins maps to be 4D for the CNN
	twins_train = twins_train.reshape((twins_train.shape[0], twins_train.shape[1], twins_train.shape[2], 1))
	twins_val = twins_val.reshape((twins_val.shape[0], twins_val.shape[1], twins_val.shape[2], 1))
	twins_test = twins_test.reshape((twins_test.shape[0], twins_test.shape[1], twins_test.shape[2], 1))


	if not get_features:
		return x_train, x_val, x_test, y_train, y_val, y_test, twins_train, twins_val, twins_test, date_dict
	else:
		return x_train, x_val, x_test, y_train, y_val, y_test, twins_train, twins_val, twins_test, date_dict, features


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


def Convolutional_Neural_Network(swmag_input_shape, twins_input_shape, encoder, trial, early_stop_patience=25):
	'''
	Initializing our model

	Args:
		n_features (int): number of input features into the model
		loss (str, optional): loss function to be uesd for training. Defaults to 'categorical_crossentropy'.
		early_stop_patience (int, optional): number of epochs the model will continue training once there
												is no longer val loss improvements. Defaults to 10.

	Returns:
		object: model configuration ready for training
		object: early stopping conditions
	'''

	# defining the hyperparameters to be tuned

	initial_filters = trial.suggest_categorical('initial_filters', [32, 64, 128])
	learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-2)
	window_size = trial.suggest_int('window_size', 1, 3)
	stride_length = trial.suggest_int('stride_length', 1, 3)
	cnn_layers = trial.suggest_int('cnn_layers', 1, 4)
	dense_layers = trial.suggest_int('dense_layers', 2, 4)
	initial_dense_nodes = trial.suggest_categorical('initial_dense_nodes', [64, 128, 256, 512])
	dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.6)


	inputs = Input(shape=input_shape)
	c = Conv2D(initial_filters, window_size, padding='same', activation='relu')(inputs)			# adding the CNN layer
	for i in range(cnn_layers):
		c = Conv2D(initial_filters*(2*(i+1)), window_size, padding='same', activation='relu')(c)			# adding the CNN layer
		if i % 2 == 0:
			c = MaxPooling2D()(c)
	f = Flatten()(c)							
	d = Dense(initial_dense_nodes, activation='relu')(f)		# Adding dense layers with dropout in between

	# twins input
	twins_input = Input(shape=twins_input_shape)
	encoder = encoder(twins_input)
	encoder = Flatten()(encoder)

	concat = concatenate([d, encoder])
	d = Dropout(dropout_rate)(concat)
	for j in range(dense_layers):
		d = Dense(int(initial_dense_nodes/(2*(j+1))), activation='relu')(d)
		d = Dropout(dropout_rate)(d)

	output = Dense(2, activation='linear')(d)

	model = Model(inputs=[inputs, twins_input], outputs=output)

	opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=CRPS)					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting

	return model, early_stop


def objective(trial, xtrain, twins_train, ytrain, xval, twins_val, yval, xtest, twins_test, ytest, swmag_input_shape, twins_input_shape):

	encoder = load-model('models/best_autoencoder.h5')
	model, early_stop = Convolutional_Neural_Network(swmag_input_shape, twins_input_shape, encoder, trial)
	print(model.summary())
	clear_session()

	try:
		model.fit(x=[xtrain,twins_train], y=ytrain, validation_data=([xval, twins_val], yval),
					verbose=1, shuffle=True, epochs=200, callbacks=[early_stop], batch_size=8)			# doing the training! Yay!
	except:
		try:
			gen = Generator(features=[Xtrain, twins_train], results=ytrain, batch_size=2)
			val_gen = Generator(features=[Xval, twins_val], results=yval, batch_size=2)

			model.fit(x=gen, validation_data=(val_gen),
					verbose=1, shuffle=True, epochs=200, callbacks=[early_stop], batch_size=2)
		except:
			print('Resource Exhausted Error')
			return None
	
	evaluation = model.evaluate([xtest, twins_test], ytest, verbose=1)
	EVALUATION_DICT[trial.number] = {'evaluation':evaluation, 'params':trial.params}
	with open('outputs/optimizing dense_version.pkl', 'ab') as f:
		pickle.dump(EVALUATION_DICT, f)
	
	return evaluation


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	xtrain, xval, xtest, ytrain, yval, ytest, twins_train, twins_val, twins_test, dates_dict = getting_prepared_data(target_var='rsd', region=region)


	# getting the input shapes for the model
	swmag_input_shape = xtrain[0].shape
	twins_input_shape = twins_train[0].shape

	storage = optuna.storages.InMemoryStorage()
	# reshaping the model input vectors for a single channel
	study = optuna.create_study(direction='minimize', study_name='autoencoder_optimization_trial')
	study.optimize(lambda trial: objective(trial, xtrain, twins_train, ytrain, xval, twins_val, yval, 
											xtest, twins_test, ytest, swmag_input_shape, twins_input_shape), 
											n_trials=50, callbacks=[lambda study, trial: gc.collect()])
	print(study.best_params)

	run_server(storage)

	optuna.visualization.plot_param_importances(study).write_image('plots/dense_param_importances.png')

	best_model, ___ = Autoencoder(input_shape, study.best_params)

	best_model.evaluate(test, test)

	best_model.save('models/best_dense_autoencoder.h5')

	optuna.visualization.plot_optimization_history(study).write_image('plots/optimization_history_dense_version.png')



if __name__ == '__main__':
	main()
	print('It ran. God job!')
