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

# import keras
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

import utils
from data_generator import Generator
from data_prep import DataPrep

# stops this program from hogging the GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

TARGET = 'rsd'
REGION=163
VERSION = 'final'

CONFIG = {'time_history':30, 'random_seed':7}


os.environ["CDF_LIB"] = "~/CDF/lib"

working_dir = '../../../../data/mike_working_dir/'
region_path = working_dir+'identifying_regions_data/adjusted_regions.pkl'
region_number = '163'
solarwind_path = '../data/SW/omniData.feather'
supermag_dir_path = '../data/supermag/'
twins_times_path = 'outputs/regular_twins_map_dates.feather'
rsd_path = working_dir+'identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'
random_seed = 7


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
	twins_train = [twins_scaler.transform(x) for x in twins_train]
	twins_val = [twins_scaler.transform(x) for x in twins_val]
	twins_test = [twins_scaler.transform(x) for x in twins_test]

	if not get_features:
		return np.array(twins_train), np.array(twins_val), np.array(twins_test), date_dict
	else:
		return np.array(twins_train), np.array(twins_val), np.array(twins_test), date_dict, features



def Autoencoder(input_shape, train, val, early_stopping_patience=25):


	model_input = Input(shape=input_shape, name='encoder_input')

	e = Conv2D(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')(model_input)
	e = Conv2D(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(e)
	e = Conv2D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same')(e)

	shape = int_shape(e)

	e = Flatten()(e)

	bottleneck = Dense(60, name='bottleneck')(e)

	d = Dense(shape[1]*shape[2]*shape[3])(bottleneck)

	d = Reshape((shape[1], shape[2], shape[3]))(d)

	d = Conv2DTranspose(filters=256, kernel_size=3, activation='relu', strides=1, padding='same')(d)
	d = Conv2DTranspose(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(d)
	d = Conv2DTranspose(filters=64, kernel_size=2, activation='relu', strides=1, padding='same')(d)

	model_outputs = Conv2DTranspose(filters=1, kernel_size=1, activation='linear', padding='same', name='decoder_output')(d)

	full_autoencoder = Model(inputs=model_input, outputs=model_outputs)

	opt = tf.keras.optimizers.Adam(learning_rate=1e-6)		# learning rate that actually started producing good results
	full_autoencoder.compile(optimizer=opt, loss='mse')					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)		# early stop process prevents overfitting

	full_autoencoder = fit_autoencoder(full_autoencoder, train, val, early_stop)

	encoder = Model(inputs=model_input, outputs = bottleneck)

	return full_autoencoder, encoder, early_stop


def fit_autoencoder(model, train, val, early_stop):

	if not os.path.exists('models/autoencoder_v_final_minmax.h5'):

		# # reshaping the model input vectors for a single channel
		# train = train.reshape((train.shape[0], train.shape[1], train.shape[2], 1))
		# val = val.reshape((val.shape[0], val.shape[1], val.shape[2], 1))

		print(model.summary())

		model.fit(train, train, validation_data=(val, val),
					verbose=1, shuffle=True, epochs=500, callbacks=[early_stop], batch_size=32)			# doing the training! Yay!

		# saving the model
		model.save('models/autoencoder_v_final_minmax.h5')

		# saving history
		history_df = pd.DataFrame(model.history.history)
		history_df.to_feather('outputs/autoencoder_v_final_minmax_history.feather')

	else:
		# loading the model if it has already been trained.
		model = load_model('models/autoencoder_v_final_minmax.h5')				# loading the models if already trained
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
	train, val, test, ___ = getting_prepared_data(target_var=TARGET, region=REGION)

	input_shape = (train.shape[1], train.shape[2], 1)

	# creating the model
	print('Initalizing model...')
	autoencoder, encoder, early_stop = Autoencoder(input_shape, train, val)

	# # fitting the model
	# print('Fitting model...')
	# MODEL = fit_autoencoder(autoencoder, train, val, early_stop)

	# encoder = Model(inputs=MODEL.inputs, outputs=MODEL.bottleneck)
	print(encoder.summary())
	encoder.save('models/encoder_final_minmax.h5')
	# encoder = Model(inputs=MODEL.inputs, outputs=MODEL.get_layer('bottleneck').output)
	# print(encoder.summary())
	# encoder.save('models/encoder_final_version_2-1.h5')

	# making predictions
	print('Making predictions...')
	predictions = making_predictions(autoencoder, test)

	# calculating the RMSE
	metrics_test = test.reshape((test.shape[0], test.shape[1]*test.shape[2]))
	metrics_predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]*predictions.shape[2]))
	rmse = np.sqrt(mean_squared_error(metrics_test, metrics_predictions))
	print(f'RMSE: {rmse}')

	vmin = min([predictions[0, :, :, 0].min(), test[0, :, :].min()])
	vmax = max([predictions[0, :, :, 0].max(), test[0, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[0, :, :, 0], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[0, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	plt.show()

	vmin = min([predictions[324, :, :, 0].min(), test[324, :, :].min()])
	vmax = max([predictions[324, :, :, 0].max(), test[324, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[324, :, :, 0], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[324, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	plt.show()

	vmin = min([predictions[256, :, :, 0].min(), test[256, :, :].min()])
	vmax = max([predictions[256, :, :, 0].max(), test[256, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[256, :, :, 0], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[256, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	plt.show()

	vmin = min([predictions[1000, :, :, 0].min(), test[1000, :, :].min()])
	vmax = max([predictions[1000, :, :, 0].max(), test[1000, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[1000, :, :, 0], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[1000, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	plt.show()

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
