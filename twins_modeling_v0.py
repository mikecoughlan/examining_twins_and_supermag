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
import argparse
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
from data_generator import Generator

# from data_prep import DataPrep

# from datetime import strftime



os.environ["CDF_LIB"] = "~/CDF/lib"

data_directory = '../../../../data/'
supermag_dir = '../data/supermag/feather_files/'
regions_dict = 'mike_working_dir/identifying_regions_data/identifying_regions_data/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = 'mike_working_dir/identifying_regions_data/identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'
working_dir = data_directory+'mike_working_dir/twins_data_modeling/'

random_seed = 7


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
			'random_seed':7,
			'initial_filters':128,
			'learning_rate':1e-7,
			'epochs':500,
			'loss':'mse',
			'early_stop_patience':25}


region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111, 401]

TARGET = 'rsd'
VERSION = 'minmax'


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
	# to_scale_with = pd.concat(x_train, axis=0)
	# scaler = StandardScaler()
	# scaler.fit(to_scale_with)
	# x_train = [scaler.transform(x) for x in x_train]
	# x_val = [scaler.transform(x) for x in x_val]
	# x_test = [scaler.transform(x) for x in x_test]

	# scaling the twins maps
	twins_scaling_array = np.vstack(twins_train)
	twins_scaling_array = twins_scaling_array.flatten()

	# dropping all negative values
	twins_scaling_array = twins_scaling_array[twins_scaling_array>0]

	# getting the mean and std of the scaling twins maps
	scaling_mean = twins_scaling_array.mean()
	scaling_std = twins_scaling_array.std()

	# scaling the twins maps
	def scaling(x, mean, std):
		return (x-mean)/std

	print(f"Scaling mean and std: {scaling_mean}, {scaling_std}")
	print(twins_train[0])

	twins_train = [scaling(x, scaling_mean, scaling_std) for x in twins_train]
	twins_val = [scaling(x, scaling_mean, scaling_std) for x in twins_val]
	twins_test = [scaling(x, scaling_mean, scaling_std) for x in twins_test]

	print(twins_train[0])

	raise

	# saving the scalers
	with open(f'models/{TARGET}/twins_region_{region}_version_{VERSION}_scaler.pkl', 'wb') as f:
		pickle.dump({'mag_and_solarwind':scaler, 'twins_mean':scaling_mean, 'twins_std':scaling_std}, f)


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


def full_model(encoder, sw_and_mag_input_shape, twins_input_shape, early_stop_patience=25):
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
	inputs = Input(shape=sw_and_mag_input_shape)
	conv1 = Conv2D(CONFIG['initial_filters'], 2, padding='same', activation='relu')(inputs)
	pool1 = MaxPooling2D(2)(conv1)
	conv2 = Conv2D(CONFIG['initial_filters']*2, 2, padding='same', activation='relu')(pool1)

	flat = Flatten()(conv2)

	# combined = concatenate([flat, encoder])
	dense1 = Dense((CONFIG['initial_filters']*2), activation='relu')(flat)

	# twins input
	twins_input = Input(shape=twins_input_shape)
	encoder = encoder(twins_input)
	encoder = Flatten()(encoder)

	combined = concatenate([dense1, encoder])
	drop1 = Dropout(0.2)(combined)
	dense2 = Dense(CONFIG['initial_filters'], activation='relu')(drop1)
	drop2 = Dropout(0.2)(dense2)
	dense3 = Dense(CONFIG['initial_filters']/2, activation='relu')(drop2)
	drop3 = Dropout(0.2)(dense3)
	output = Dense(2, activation='linear')(drop3)

	model = Model(inputs=[inputs, twins_input], outputs=output)

	opt = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=CRPS)
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting

	print("after model initialization....")
	print(model.summary())

	return model, early_stop



def fit_full_model(model, xtrain, xval, ytrain, yval, twins_train, twins_val, early_stop, region):
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

	if not os.path.exists(f'models/{TARGET}/twins_region_{region}_v{VERSION}.h5'):

		# reshaping the model input vectors for a single channel
		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
		# twins_train = twins_train.reshape((twins_train.shape[0], twins_train.shape[1], twins_train.shape[2], 1))

		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))
		# twins_val = twins_val.reshape((twins_val.shape[0], twins_val.shape[1], twins_val.shape[2], 1))

		print('Mean of target variable: '+str(ytrain.mean()))
		print('Std of target variable: '+str(ytrain.std()))
		print('Max of target variable: '+str(ytrain.max()))
		print('Min of target variable: '+str(ytrain.min()))

		print(model.summary())

				# doing the training! Yay!
		try:
			model.fit(x=[xtrain,twins_train], y=ytrain, validation_data=([xval, twins_val], yval),
						verbose=1, shuffle=True, epochs=500, callbacks=[early_stop], batch_size=8)			# doing the training! Yay!
		except:
			gen = Generator(features=[Xtrain, twins_train], results=ytrain, batch_size=2)
			val_gen = Generator(features=[Xval, twins_val], results=yval, batch_size=2)

			model.fit(x=gen, validation_data=(val_gen),
						verbose=1, shuffle=True, epochs=500, callbacks=[early_stop], batch_size=2)

		# saving the model
		model.save(f'models/{TARGET}/twins_region_{region}_v{VERSION}.h5')

	else:

		# loading the model if it has already been trained.
		model = load_model(f'models/{TARGET}/twins_region_{region}_v{VERSION}.h5', compile=False)				# loading the models if already trained
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss=CRPS)

	return model


def making_predictions(model, Xtest, twins_test, ytest, test_dates):
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
	# twins_test = twins_test.reshape((twins_test.shape[0], twins_test.shape[1], twins_test.shape[2], 1))

	predicted = model.predict([Xtest, twins_test], verbose=1)						# predicting on the testing input data

	predicted_mean = tf.gather(predicted, [[0]], axis=1)					# grabbing the positive node
	predicted_std = tf.gather(predicted, [[1]], axis=1)					# grabbing the positive node

	predicted_mean = predicted_mean.numpy()									# turning to a numpy array
	predicted_std = predicted_std.numpy()									# turning to a numpy array

	predicted_mean = pd.Series(predicted_mean.reshape(len(predicted_mean),))		# and then into a pd.series
	predicted_std = pd.Series(predicted_std.reshape(len(predicted_std),))		# and then into a pd.series

	ytest = pd.Series(ytest.reshape(len(ytest),))			# turning the ytest into a pd.series

	dates = pd.Series(test_dates['Date_UTC'])
	results_df = pd.DataFrame({'predicted_mean':predicted_mean, 'predicted_std':predicted_std, 'actual':ytest, 'dates':test_dates['Date_UTC']})

	return results_df


def main(region):
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''
	if not os.path.exists(f'outputs/{TARGET}'):
		os.makedirs(f'outputs/{TARGET}')
	if not os.path.exists(f'models/{TARGET}'):
		os.makedirs(f'models/{TARGET}')

	# loading all data and indicies
	print('Loading data...')
	xtrain, xval, xtest, ytrain, yval, ytest, twins_train, twins_val, twins_test, dates_dict = getting_prepared_data(target_var=TARGET, region=region)

	print('xtrain shape: '+str(xtrain.shape))
	print('xval shape: '+str(xval.shape))
	print('xtest shape: '+str(xtest.shape))
	print('ytrain shape: '+str(ytrain.shape))
	print('yval shape: '+str(yval.shape))
	print('ytest shape: '+str(ytest.shape))
	print('twins_train shape: '+str(twins_train.shape))
	print('twins_val shape: '+str(twins_val.shape))
	print('twins_test shape: '+str(twins_test.shape))

	with open(f'outputs/dates_dict_version_{VERSION}.pkl', 'wb') as f:
		pickle.dump(dates_dict, f)

	encoder = load_model('models/encoder_final_version_2.h5')
	encoder.trainable = False
	print(encoder.summary())

	# creating the model
	print('Initalizing model...')
	MODEL, early_stop = full_model(encoder=encoder,
									sw_and_mag_input_shape=(xtrain.shape[1], xtrain.shape[2], 1),
									twins_input_shape=(twins_train.shape[1], twins_train.shape[2], 1),
									early_stop_patience=CONFIG['early_stop_patience'])

	# fitting the model
	print('Fitting model...')
	MODEL = fit_full_model(MODEL, xtrain, xval, ytrain, yval, twins_train, twins_val, early_stop, region)

	# making predictions
	print('Making predictions...')
	print(MODEL.summary())
	raise
	results_df = making_predictions(model=MODEL, Xtest=xtest, twins_test=twins_test, ytest=ytest, test_dates=dates_dict['test'])

	results_df.to_feather(f'outputs/{TARGET}/twins_modeling_region_{region}_version_{VERSION}.feather')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--region',
						action='store',
						choices=region_numbers,
						type=int,
						help='Region number to be trained.')

	args=parser.parse_args()

	if os.path.exists(f'models/{TARGET}/twins_region_{args.region}_v{VERSION}.h5'):
		if os.path.exists(f'outputs/{TARGET}/twins_modeling_region_{args.region}_version_{VERSION}.feather'):
			print(f'Runing region {args.region}...')
			main(args.region)
			print('It ran. God job!')
		else:
			print(f'Already have results for this region {args.region}. Skipping...')
	else:
		print(f'Already ran region {args.region}. Skipping...')