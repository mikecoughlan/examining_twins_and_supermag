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

VERSION = 'optimizing_autoencoder'

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

	if os.path.exists(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{VERSION}.pkl'):
		with open(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{VERSION}.pkl', 'rb') as f:
			storms_extracted_dict = pickle.load(f)
		storms = storms_extracted_dict['storms']
		target = storms_extracted_dict['target']

	else:
		# getting the data corresponding to the twins maps
		storms, target = utils.storm_extract(df=merged_df, lead=30, recovery=9, twins=True, target=True, target_var=f'rolling_{target_var}', concat=False, map_keys=maps.keys())
		storms_extracted_dict = {'storms':storms, 'target':target}
		with open(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{VERSION}.pkl', 'wb') as f:
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

	# scaling the twins maps
	twins_scaling_array = np.vstack(twins_train)
	twins_scaler = MinMaxScaler()
	twins_scaler.fit(twins_scaling_array)
	twins_train = np.array([twins_scaler.transform(x) for x in twins_train])
	twins_val = np.array([twins_scaler.transform(x) for x in twins_val])
	twins_test = np.array([twins_scaler.transform(x) for x in twins_test])

	input_shape = (twins_train.shape[1], twins_train.shape[2], 1)

	return twins_train, twins_val, twins_test, input_shape


def Autoencoder(input_shape, trial, early_stopping_patience=10):


	initial_filters = trial.suggest_categorical('initial_filters', [32, 64, 128])
	latent_dim = trial.suggest_categorical('latent_dim', [60, 120])
	learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-2)
	layers = trial.suggest_int('layers', 2, 4)
	activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
	loss = trial.suggest_categorical('loss', ['mse', 'binary_crossentropy'])


	model_input = Input(shape=input_shape, name='encoder_input')
	filters = initial_filters

	for i in range(layers):

		if i == 0:
			e = Conv2D(filters=filters, kernel_size=3, activation=activation, strides=1, padding='same')(model_input)
			filters = (filters*2)
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
	clear_session()
	try:
		model.fit(train, train, validation_data=(val, val),
				verbose=1, shuffle=True, epochs=200,
				callbacks=[early_stop], batch_size=16)			# doing the training! Yay!
	except:
		print('Resource Exhausted Error')
		return None
	evaluation = model.evaluate(test, test, verbose=1)
	EVALUATION_DICT[trial.number] = {'evaluation':evaluation, 'params':trial.params}
	with open('outputs/optimizing autoencoder_evaluation_dict.pkl', 'ab') as f:
		pickle.dump(EVALUATION_DICT, f)
	return evaluation


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	train, val, test, input_shape = getting_prepared_data(target_var='rsd', region=region_number)

	storage = optuna.storages.InMemoryStorage()
	# reshaping the model input vectors for a single channel
	study = optuna.create_study(direction='minimize', study_name='autoencoder_optimization_trial')
	study.optimize(lambda trial: objective(trial, train, val, test, input_shape), n_trials=50, callbacks=[lambda study, trial: gc.collect()])
	print(study.best_params)

	run_server(storage)

	optuna.visualization.plot_param_importances(study).write_image('plots/param_importances.png')

	best_model, ___ = Autoencoder(input_shape, study.best_params)

	best_model.evaluate(test, test)

	best_model.save('models/best_autoencoder.h5')

	optuna.visualization.plot_optimization_history(study).write_image('plots/optimization_history.png')



if __name__ == '__main__':
	main()
	print('It ran. God job!')
