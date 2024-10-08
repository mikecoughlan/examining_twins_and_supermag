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
from spacepy import pycdf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import to_categorical

import utils
from data_prep import DataPrep

os.environ["CDF_LIB"] = "~/CDF/lib"

twins_dir = '../data/twins/'
supermag_dir = '../data/supermag/'
regions_dict = '../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = '../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'
twins_times_path = 'outputs/regular_twins_map_dates.feather'
random_seed = 42


# loading config and specific model config files. Using them as dictonaries
with open('twins_config.json', 'r') as con:
	CONFIG = json.load(con)

with open('model_config.json', 'r') as mcon:
	MODEL_CONFIG = json.load(mcon)

# defining global vars used for saving files
MLT_SPAN = 2
MLT_BIN_TARGET = 4
VERSION = '3-1'



region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]


def get_all_data(percentile, mlt_span):


	# loading all the datasets and dictonaries
	if os.path.exists('outputs/twins_maps_with_footpoints.pkl'):
		with open('outputs/twins_maps_with_footpoints.pkl', 'rb') as f:
			twins = pickle.load(f)
	else:
		twins = utils.loading_twins_maps()

	regions, stats = utils.loading_dicts()
	solarwind = utils.loading_solarwind()

	# reduce the regions dict to be only the ones that have keys in the region_numbers list
	regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in region_numbers}

	percentile_dataframe = pd.DataFrame()

	# Getting regions data for each region
	for region in regions.keys():

		# getting dbdt and rsd data for the region
		temp_df = utils.combining_regional_dfs(regions[region]['station'], stats[region])

		# getting the mean latitude for the region and attaching it to the regions dictionary
		mean_lat = utils.getting_mean_lat(regions[region]['station'])
		regions[region]['mean_lat'] = mean_lat

		# cutting the temp df down to the TWINS era
		temp_df = temp_df[pd.to_datetime('2009-07-20'):pd.to_datetime('2017-12-31')]

		# segmenting the rsd data for calculating percentiles
		percentile_dataframe = pd.concat([percentile_dataframe, temp_df[['rsd', 'MLT']]], axis=0, ignore_index=True)

		# attaching the regional data to the regions dictionary with only the keys that are in the twins dictionary
		regions[region]['combined_dfs'] = temp_df[temp_df.index.isin(twins.keys())]

	# calculating the percentiles for each region
	mlt_perc = utils.calculate_percentiles(percentile_dataframe, mlt_span, percentile)

	# Attaching the algorithm maps to the twins dictionary
	algorithm_maps = utils.loading_algorithm_maps()

	data_dict = {'twins_maps':twins, 'solarwind':solarwind, 'regions':regions,
					'algorithm_maps':algorithm_maps, 'percentiles':mlt_perc}

	return data_dict



def getting_prepared_data(mlt_span, mlt_bin_target, percentile=0.99, start_date=pd.to_datetime('2009-07-20'), end_date=pd.to_datetime('2017-12-31')):

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
	start_date = pd.to_datetime('2009-07-20')
	end_date = pd.to_datetime('2017-12-31')

	data_dict = get_all_data(percentile=percentile, mlt_span=mlt_span)

	# splitting up the regions based on MLT value into 1 degree bins
	mlt_bins = np.arange(0, 24, mlt_span)
	mlt_dict = {}
	for mlt in mlt_bins:
		# TEMPORARY DURING THE DEBUGGING PROCESS!!!!!!
		if mlt != mlt_bin_target:
			continue
		mlt_df = pd.DataFrame(index=pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='min'))
		for region in data_dict['regions'].values():

			# segmenting one MLT wedge
			temp_df = region['combined_dfs'][region['combined_dfs']['MLT'].between(mlt, mlt+mlt_span)]

			mlt_df = pd.concat([mlt_df, temp_df['rsd']], axis=1, ignore_index=False)

		mlt_df.columns = [f'region_{reg}' for reg in region_numbers]
		max_regions = mlt_df.idxmax(axis=1)
		mlt_df['max'] = mlt_df.max(axis=1)
		mlt_df['region_max'] = max_regions

		mlt_df.dropna(inplace=True, subset=['max'])

		mlt_df = utils.classification_column(df=mlt_df, param='max', thresh=data_dict['percentiles'][f'{mlt}'], forecast=0, window=0)

		mlt_df['mean_lat'] = mlt_df['region_max'].apply(lambda x: data_dict['regions'][x]['mean_lat'])

		# getting only the mid and high lat data
		mlt_df = mlt_df[mlt_df['mean_lat'] >= 55]

		mlt_dict[f'{mlt}'] = mlt_df

	# segmenting the bin that's going to be trained on
	target_mlt_bin = mlt_dict[f'{mlt_bin_target}']

	# making sure the dates for the twins maps and the targets match up
	twins_dates = [key for key in data_dict['twins_maps'].keys()]
	target_mlt_bin = target_mlt_bin[target_mlt_bin.index.isin(twins_dates)]

	# creating the target vectors
	y = to_categorical(target_mlt_bin['classification'].to_numpy(), num_classes=2)

	# creating the input vectors
	X = []
	for date in target_mlt_bin.index:
		X.append(data_dict['twins_maps'][date.strftime('%Y-%m-%d %H:%M:%S')]['map'])
	X = np.array(X)

	# getting dates so they can be split with the input and target data
	dates = target_mlt_bin.index

	# splitting the data into training, validation, and testing sets
	x_train, x_test, x_val, y_train, y_test, y_val, dates_dict = utils.splitting_and_scaling(X, y, dates, test_size=0.2, val_size=0.125, random_seed=random_seed)

	return x_train, x_val, x_test, y_train, y_val, y_test, dates_dict


def create_CNN_model(input_shape, loss='binary_crossentropy', early_stop_patience=10):
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

	model.add(Conv2D(MODEL_CONFIG['filters'], 3, padding='same',
								activation='relu', input_shape=input_shape))			# adding the CNN layer
	model.add(MaxPooling2D())
	model.add(Conv2D(MODEL_CONFIG['filters']*2, 3, padding='same', activation='relu'))			# adding the CNN layer
	model.add(MaxPooling2D())
	model.add(Conv2D(MODEL_CONFIG['filters']*2, 3, padding='same', activation='relu'))			# adding the CNN layer
	model.add(MaxPooling2D())
	model.add(Conv2D(MODEL_CONFIG['filters']*2, 2, padding='same', activation='relu'))			# adding the CNN layer
	model.add(MaxPooling2D())
	model.add(Flatten())							# changes dimensions of model. Not sure exactly how this works yet but improves results
	model.add(Dense(MODEL_CONFIG['filters']*2, activation='relu'))		# Adding dense layers with dropout in between
	model.add(Dropout(0.2))
	model.add(Dense(MODEL_CONFIG['filters'], activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	loss = BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.5, gamma=2.0)		# loss function that is good for imbalanced data
	opt = tf.optimizers.Adam(learning_rate=MODEL_CONFIG['initial_learning_rate'])		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=loss)					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting


	return model, early_stop


def fit_CNN(model, xtrain, xval, ytrain, yval, early_stop, mlt_bin, mlt_span):
	'''
	Performs the actual fitting of the model.

	Args:
		model ( model): model as defined in the create_model function.
		xtrain (3D np.array): training data inputs
		xval (3D np.array): validation inputs
		ytrain (2D np.array): training target vectors
		yval (2D np.array): validation target vectors
		early_stop ( early stopping dict): predefined early stopping function
		split (int): split being trained. Used for saving model.
		station (str): station being trained.
		first_time (bool, optional): if True model will be trainined, False model will be loaded. Defaults to True.

	Returns:
		model: fit model ready for making predictions.
	'''

	if not os.path.exists(f'models/mlt_bin_{mlt_bin}_span_{mlt_span}_version_{VERSION}.h5'):

		# reshaping the model input vectors for a single channel
		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))

		model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
					verbose=1, shuffle=True, epochs=MODEL_CONFIG['epochs'], callbacks=[early_stop])			# doing the training! Yay!

		# saving the model
		model.save(f'models/mlt_bin_{mlt_bin}_span_{mlt_span}_version_{VERSION}.h5')

	else:
		# loading the model if it has already been trained.
		model = load_model(f'models/mlt_bin_{mlt_bin}_span_{mlt_span}_version_{VERSION}.h5')				# loading the models if already trained

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
	predicted = tf.gather(predicted, [[1]], axis=1)					# grabbing the positive node
	predicted = predicted.numpy()									# turning to a numpy array
	predicted = pd.Series(predicted.reshape(len(predicted),), index=test_dates)		# and then into a pd.series
	ytest = pd.Series(ytest[:,1].reshape(len(ytest),), index=test_dates)			# turning the ytest into a pd.series

	results_df = pd.DataFrame(index=test_dates)						# and storing the results
	results_df['predicted'] = predicted
	results_df['actual'] = ytest

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

	metrics = pd.DataFrame({'rmse':rmse,
							'mae':mae,
							'mape':mape,
							'r2':r2},
							index=[0])

	return metrics


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	xtrain, xval, xtest, ytrain, yval, ytest, dates_dict = getting_prepared_data(mlt_span=MLT_SPAN, mlt_bin_target=MLT_BIN_TARGET, percentile=0.99,\
																				start_date=pd.to_datetime('2009-07-20'), end_date=pd.to_datetime('2017-12-31'))

	# creating the model
	print('Initalizing model...')
	MODEL, early_stop = create_CNN_model(input_shape=(xtrain.shape[1], xtrain.shape[2], 1), loss=MODEL_CONFIG['loss'],
											early_stop_patience=MODEL_CONFIG['early_stop_patience'])

	# fitting the model
	print('Fitting model...')
	MODEL = fit_CNN(MODEL, xtrain, xval, ytrain, yval, early_stop, mlt_bin=MLT_BIN_TARGET, mlt_span=MLT_SPAN)

	# making predictions
	print('Making predictions...')
	results_df = making_predictions(MODEL, xtest, ytest, dates_dict['test'])
	results_df = results_df.reset_index(drop=False).rename(columns={'index':'Date_UTC'})

	# saving the results
	print('Saving results...')
	results_df.to_feather(f'outputs/mlt_bin_{MLT_BIN_TARGET}_span_{MLT_SPAN}_version_{VERSION}.feather')

	# calculating some metrics
	print('Calculating metrics...')
	metrics = calculate_some_metrics(results_df)

	# # saving the metrics
	# print('Saving metrics...')
	# metrics.to_feather('outputs/non_twins_metrics.feather')



if __name__ == '__main__':
	main()
	print('It ran. God job!')
