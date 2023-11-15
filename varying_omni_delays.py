####################################################################################
#
# exmining_twins_and_supermag/varying_omni_delays.py
#
#
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

DELAYS = [10, 15, 20]

CONFIG = {'region_numbers': [223],
			'load_twins':False,
			'mag_features':[],
			'solarwind_features':[],
			'delay':False,
			'rolling':False,
			'to_drop':[],
			'omni_or_ace':'omni',
			'time_history':30,
			'random_seed':42}

with open('outputs/best_params_207_version_optimizer.pkl', 'rb') as f:
	MODEL_CONFIG = pickle.load(f)


region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]

TARGET = 'rsd'
VERSION = 'optimized'


def loading_data(target_var, region, delay, percentile=0.99):

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

	solarwind = solarwind.shift(-delay)

	# reduce the regions dict to be only the ones that have keys in the region_numbers list
	regions = regions[f'region_{region}']
	stats = stats[f'region_{region}']

	# getting dbdt and rsd data for the region
	supermag_df = utils.combining_stations_into_regions(regions['station'], stats, features=['dbht', 'MAGNITUDE', 'theta', 'sin_theta', 'cos_theta', 'N', 'E'], mean=True, std=True, maximum=True, median=True)

	# getting the mean latitude for the region and attaching it to the regions dictionary
	mean_lat = utils.getting_mean_lat(regions['station'])

	threshold = supermag_df[target_var].quantile(percentile)

	# supermag_df.drop(columns=supermag_corr_dict[f'region_{region}']['twins_corr'], inplace=True)
	# solarwind.drop(columns=solarwind_corr_dict[f'region_{region}']['twins_corr_features'], inplace=True)

	merged_df = pd.merge(supermag_df, solarwind, left_index=True, right_index=True, how='inner')


	return merged_df, mean_lat, threshold



def getting_prepared_data(target_var, region, delay, get_features=False):
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

	merged_df, mean_lat, threshold = loading_data(target_var=target_var, region=region, delay=delay, percentile=0.99)

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
	if 'dbht_std' in merged_df.columns:
		vars_to_drop.append('dbht_std')
	if 'dbht_max' in merged_df.columns:
		vars_to_drop.append('dbht_max')
	if 'dbht_median' in vars_to_drop:
		vars_to_drop.pop('dbht_median')
	if 'cos_theta_median' in merged_df.columns:
		vars_to_drop.append('cos_theta_median')
	if 'cos_theta_max' in merged_df.columns:
		vars_to_drop.append('cos_theta_max')
	if 'cos_theta_mean' in merged_df.columns:
		vars_to_drop.append('cos_theta_mean')
	if 'sin_theta_median' in merged_df.columns:
		vars_to_drop.append('sin_theta_median')
	if 'sin_theta_max' in merged_df.columns:
		vars_to_drop.append('sin_theta_max')
	if 'sin_theta_mean' in merged_df.columns:
		vars_to_drop.append('sin_theta_mean')
	if 'theta_mean' in merged_df.columns:
		vars_to_drop.append('theta_mean')
	if 'theta_median' in merged_df.columns:
		vars_to_drop.append('theta_median')
	if 'theta_max' in merged_df.columns:
		vars_to_drop.append('theta_max')
	if 'theta_std' in merged_df.columns:
		vars_to_drop.append('theta_std')
	if 'dbht_mean' in merged_df.columns:
		vars_to_drop.append('dbht_mean')
	if 'MAGNITUDE_mean' in merged_df.columns:
		vars_to_drop.append('MAGNITUDE_mean')
	if 'MAGNITUDE_median' in merged_df.columns:
		vars_to_drop.append('MAGNITUDE_median')
	if 'MAGNITUDE_max' in merged_df.columns:
		vars_to_drop.append('MAGNITUDE_max')

	vars_to_keep = [f'rolling_{target_var}', 'dbht_median', 'MAGNITUDE_median', 'MAGNITUDE_std', 'sin_theta_std', 'cos_theta_std', 'N_std', 'N_max', 'E_std', 'E_median', 'cosMLT', 'sinMLT',
					'B_Total', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz', 'logT']

	merged_df = merged_df[vars_to_keep]
	# merged_df.drop(columns=vars_to_drop, inplace=True)
	# merged_df.dropna(subset=[f'rolling_{target_var}'], inplace=True)

	print('Columns in Merged Dataframe: '+str(merged_df.columns))

	print(merged_df.corr())
	merged_df.corr().to_csv(f'outputs/{TARGET}/corr.csv')

	print(f'Target value positive percentage: {target.sum()/len(target)}')
	# merged_df.drop(columns=[f'rolling_{target_var}', 'classification'], inplace=True)

	if os.path.exists(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_delay_{delay}.pkl'):
		with open(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_delay_{delay}.pkl', 'rb') as f:
			storms_extracted_dict = pickle.load(f)
		storms = storms_extracted_dict['storms']
		target = storms_extracted_dict['target']

	else:
	# getting the data corresponding to the twins maps
		storms, target = utils.storm_extract(df=merged_df, lead=30, recovery=9, twins=True, target=True, target_var=f'rolling_{target_var}', concat=False)
		storms_extracted_dict = {'storms':storms, 'target':target}
		with open(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_delay_{delay}.pkl', 'wb') as f:
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

	print(f'shape of x_train: {len(x_train)}')
	print(f'shape of x_val: {len(x_val)}')
	print(f'shape of x_test: {len(x_test)}')

	# splitting the sequences for input to the CNN
	x_train, y_train, train_dates_to_drop, ___ = utils.split_sequences(x_train, y_train, n_steps=CONFIG['time_history'], dates=date_dict['train'], model_type='regression')
	x_val, y_val, val_dates_to_drop, ___ = utils.split_sequences(x_val, y_val, n_steps=CONFIG['time_history'], dates=date_dict['val'], model_type='regression')
	x_test, y_test, test_dates_to_drop, ___  = utils.split_sequences(x_test, y_test, n_steps=CONFIG['time_history'], dates=date_dict['test'], model_type='regression')

	# dropping the dates that correspond to arrays that would have had nan values
	date_dict['train'].drop(train_dates_to_drop, axis=0, inplace=True)
	date_dict['val'].drop(val_dates_to_drop, axis=0, inplace=True)
	date_dict['test'].drop(test_dates_to_drop, axis=0, inplace=True)

	date_dict['train'].reset_index(drop=True, inplace=True)
	date_dict['val'].reset_index(drop=True, inplace=True)
	date_dict['test'].reset_index(drop=True, inplace=True)

	print(f'Total training dates: {len(date_dict["train"])}')

	print(f'shape of x_train: {x_train.shape}')
	print(f'shape of x_val: {x_val.shape}')
	print(f'shape of x_test: {x_test.shape}')

	print(f'Nans in training data: {np.isnan(x_train).sum()}')
	print(f'Nans in validation data: {np.isnan(x_val).sum()}')
	print(f'Nans in testing data: {np.isnan(x_test).sum()}')

	print(f'Nans in training target: {np.isnan(y_train).sum()}')
	print(f'Nans in validation target: {np.isnan(y_val).sum()}')
	print(f'Nans in testing target: {np.isnan(y_test).sum()}')

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


def create_CNN_model(input_shape, model_dict, early_stop_patience=10):
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


	model = Sequential()						# initalizing the model

	model.add(Conv2D(model_dict['initial_filters'], model_dict['window_size'], padding='same', activation=model_dict['activation'], input_shape=input_shape))			# adding the CNN layer
	for i in range(model_dict['cnn_layers']):
		model.add(Conv2D(model_dict['initial_filters']*model_dict['cnn_step_up'], model_dict['window_size'], padding='same', activation=model_dict['activation']))			# adding the CNN layer
		if i % 2 == 0:
			model.add(MaxPooling2D())
		model_dict['cnn_step_up'] *= 2

	model.add(Flatten())							# changes dimensions of model. Not sure exactly how this works yet but improves results
	model.add(Dense(model_dict['initial_dense_nodes'], activation=model_dict['activation']))		# Adding dense layers with dropout in between
	model.add(Dropout(model_dict['dropout_rate']))
	for j in range(model_dict['dense_layers']):
		model.add(Dense(int(model_dict['initial_dense_nodes']/model_dict['dense_node_decrease_step']), activation=model_dict['activation']))
		model.add(Dropout(model_dict['dropout_rate']))
		model_dict['dense_node_decrease_step'] *= 2

	model.add(Dense(2, activation='linear'))
	opt = tf.keras.optimizers.Adam(learning_rate=model_dict['learning_rate'])		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=CRPS)					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting


	return model, early_stop


def fit_CNN(model, xtrain, xval, ytrain, yval, early_stop, delay):
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

	if not os.path.exists(f'models/{TARGET}/non_twins_delay_{delay}_v{VERSION}.h5'):

		# reshaping the model input vectors for a single channel
		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))

		model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
					verbose=1, shuffle=True, epochs=500, callbacks=[early_stop])			# doing the training! Yay!

		# saving the model
		model.save(f'models/{TARGET}/non_twins_delay_{delay}_v{VERSION}.h5')

	else:
		# loading the model if it has already been trained.
		model = load_model(f'models/{TARGET}/non_twins_delay_{delay}_v{VERSION}.h5', compile=False)				# loading the models if already trained
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']), loss=CRPS)


	return model


def making_predictions(model, Xtest, ytest, test_dates):
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
	print('Test input Nans: '+str(np.isnan(Xtest).sum()))

	nans = pd.Series(np.isnan(Xtest.sum(axis=1).sum(axis=1)).reshape(len(np.isnan(Xtest.sum(axis=1).sum(axis=1))),))

	predicted = model.predict(Xtest, verbose=1)						# predicting on the testing input data

	predicted_mean = tf.gather(predicted, [[0]], axis=1)					# grabbing the positive node
	predicted_std = tf.gather(predicted, [[1]], axis=1)					# grabbing the positive node

	predicted_mean = predicted_mean.numpy()									# turning to a numpy array
	predicted_std = predicted_std.numpy()									# turning to a numpy array

	predicted_mean = pd.Series(predicted_mean.reshape(len(predicted_mean),))		# and then into a pd.series
	predicted_std = pd.Series(predicted_std.reshape(len(predicted_std),))		# and then into a pd.series

	ytest = pd.Series(ytest.reshape(len(ytest),))			# turning the ytest into a pd.series

	# results_df = pd.DataFrame()						# and storing the results
	# results_df['predicted'] = predicted
	# results_df['actual'] = ytest
	# dates = pd.Series(test_dates.reshape(len(test_dates),))
	dates = pd.Series(test_dates['Date_UTC'])
	results_df = pd.DataFrame({'predicted_mean':predicted_mean, 'predicted_std':predicted_std, 'actual':ytest, 'dates':test_dates['Date_UTC']})

	return results_df


def calculate_some_metrics(results_df):

	# calculating the RMSE
	rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['predicted']))
	print('RMSE: '+str(rmse))

	# calculating the MAE
	mae = mean_absolute_error(results_df['actual'], results_df['predicted'])
	print('MAE: '+str(mae))

	# calculating the MAPE
	mape = np.mean(np.abs((results_df['actual'] - results_df['predicted']) / results_df['actual'])) * 100
	print('MAPE: '+str(mape))

	# calculating the R^2
	r2 = r2_score(results_df['actual'], results_df['predicted'])
	print('R^2: '+str(r2))

	metrics = {'rmse':rmse,
							'mae':mae,
							'mape':mape,
							'r2':r2}

	return metrics


def main(region, delay):
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''
	if not os.path.exists(f'outputs/{TARGET}'):
		os.makedirs(f'outputs/{TARGET}')
	if not os.path.exists(f'models/{TARGET}'):
		os.makedirs(f'models/{TARGET}')

	# loading all data and indicies
	print('Loading data...')
	xtrain, xval, xtest, ytrain, yval, ytest, dates_dict = getting_prepared_data(target_var=TARGET, region=region, delay=delay)

	print('xtrain shape: '+str(xtrain.shape))
	print('xval shape: '+str(xval.shape))
	print('xtest shape: '+str(xtest.shape))
	print('ytrain shape: '+str(ytrain.shape))
	print('yval shape: '+str(yval.shape))
	print('ytest shape: '+str(ytest.shape))

	print('Nans in training data: '+str(np.isnan(xtrain).sum()))
	print('Nans in validation data: '+str(np.isnan(xval).sum()))
	print('Nans in testing data: '+str(np.isnan(xtest).sum()))
	print('Nans in training target: '+str(np.isnan(ytrain).sum()))
	print('Nans in validation target: '+str(np.isnan(yval).sum()))
	print('Nans in testing target: '+str(np.isnan(ytest).sum()))

	# with open(f'outputs/dates_dict_version_{VERSION}.pkl', 'wb') as f:
	# 	pickle.dump(dates_dict, f)

	# creating the model
	print('Initalizing model...')
	MODEL, early_stop = create_CNN_model(input_shape=(xtrain.shape[1], xtrain.shape[2], 1), model_dict=MODEL_CONFIG,
											early_stop_patience=25)

	# fitting the model
	print('Fitting model...')
	MODEL = fit_CNN(MODEL, xtrain, xval, ytrain, yval, early_stop, delay=delay)

	# making predictions
	print('Making predictions...')
	results_df = making_predictions(MODEL, xtest, ytest, dates_dict['test'])
	results_df.to_feather(f'outputs/{TARGET}/omni_delay_results/non_twins_modeling_delay_{delay}_version_{VERSION}.feather')



if __name__ == '__main__':
	for region in CONFIG['region_numbers']:
		for delay in DELAYS:
			main(region, delay)
	print('It ran. God job!')
