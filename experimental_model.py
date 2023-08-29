import datetime
import gc
import glob
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
import tensorflow as tf
import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python.keras.backend import get_session

os.environ["CDF_LIB"] = "~/CDF/lib"

# stops this program from hogging the GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

twins_dir = '../data/twins/'
supermag_dir = '../data/supermag/'
regions_dict = '../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = '../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'


region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]

times = pd.read_feather('outputs/regular_twins_map_dates.feather')

def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted


def loading_dicts():

	with open(regions_dict, 'rb') as f:
		regions = pickle.load(f)

	regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in region_numbers}

	with open(regions_stat_dict, 'rb') as g:
		stats = pickle.load(g)

	stats = {f'region_{reg}': stats[f'region_{reg}'] for reg in region_numbers}

	return regions, stats


def loading_twins_maps():

	twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

	maps = {}
	maps_array = []

	for file in twins_files:
		twins_map = pycdf.CDF(file)
		for i, date in enumerate(twins_map['Epoch']):
			if len(np.unique(twins_map['Ion_Temperature'][i][50:140,40:100])) == 1:
				continue
			check = pd.to_datetime(date.strftime(format='%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
			if check in times.values:
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')] = twins_map['Ion_Temperature'][i][50:140,40:100]
				maps_array.append(twins_map['Ion_Temperature'][i][50:140,40:100])

	maps_array = np.array(maps_array)

	return maps


def combining_regional_dfs(stations, rsd, elements, delay):

	start_time = pd.to_datetime('2009-07-20')
	end_time = pd.to_datetime('2017-12-31')
	twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

	combined_stations = pd.DataFrame(index=twins_time_period)

	for station in stations:
		stat = loading_supermag(station, start_time, end_time)
		stat = stat[['MLT', 'dbht']]
		for col in stat.columns:
			stat[f'{col}_delay_{delay}'] = stat[col].shift(-delay)
		combined_stations = pd.concat([combined_stations, stat[f'{col}_delay_{delay}']], axis=1, ignore_index=False)

	mean_dbht = combined_stations.mean(axis=1)
	max_dbht = combined_stations.max(axis=1)

	combined_stations['reg_mean'] = mean_dbht
	combined_stations['reg_max'] = max_dbht
	combined_stations['rsd'] = rsd['max_rsd']
	combined_stations['MLT'] = rsd['MLT']

	segmented_df = pd.DataFrame()
	# for element in elements:
	# try:
		# segmented_df = pd.concat([segmented_df, combined_stations[element]], axis=0)
	segmented_df = combined_stations[combined_stations.index.isin(elements)]
	# except KeyError:
	# 	continue

	return segmented_df


def loading_supermag(station, start_time, end_time):

	df = pd.read_feather(supermag_dir+station+'.feather')

	# limiting the analysis to the nightside
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
	df = df[start_time:end_time]

	return df


def preparing_data_for_modeling(combined_stations, maps, target_var, cv):

	mlt = combined_stations['MLT'].values
	target = combined_stations[target_var].values
	print(f'Length of mlt: {len(mlt)}')
	print(f'Length of maps: {len(maps)}')
	print(f'Length of target: {len(target)}')
	print(f"Nan's in target: {np.isnan(target).sum()} ")

	map_array = []
	# maps_array = np.array(list(maps.values()))
	for i, (tmap, mag_time) in enumerate(zip(maps, mlt)):

		# add a column to the tmap array with the mlt value for that map
		mag_time = np.full(maps[tmap].shape[0], mag_time)
		new_map = np.hstack((maps[tmap], mag_time.reshape(-1, 1)))
		# mag_time = np.array([mag_time for i in range(maps[tmap].shape[1])])
		# new_map = np.vstack(maps[tmap], mag_time)
		new_map[np.isnan(new_map)] = -1
		map_array.append(new_map)
	map_array = np.array(map_array)

	map_array = map_array[~np.isnan(target)]
	target = np.delete(target, list(np.where(np.isnan(target))))
	# map_array = np.delete(map_array, list(np.where(np.isnan(target))), axis=0)

	print(f'Length of map_array: {len(map_array)}')
	print(f'Length of target: {len(target)}')
	print(f"Nan's in target: {np.isnan(target).sum()} ")

	train_x, test_x, train_y, test_y = train_test_split(map_array, target, test_size=0.2, random_state=cv, shuffle=True)
	train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=cv, shuffle=True)

	processed_dict = {'train_x':train_x, 'val_x':val_x, 'test_x':test_x,
						'train_y':train_y, 'val_y':val_y, 'test_y':test_y}

	stats_df = checking_stats(processed_dict)

	scaling_max = np.max(train_x)
	scaling_min = np.min(train_x)

	train_x = (train_x - scaling_min) / (scaling_max - scaling_min)
	val_x = (val_x - scaling_min) / (scaling_max - scaling_min)
	test_x = (test_x - scaling_min) / (scaling_max - scaling_min)

	processed_dict = {'train_x':train_x, 'train_y':train_y,
						'val_x':val_x, 'val_y':val_y,
						'test_x':test_x, 'test_y':test_y}


	return processed_dict, stats_df

def checking_stats(processed_dict):

	means, maxes, stds, perc = [], [], [], []
	for key in processed_dict.keys():
		means.append(np.nanmean(processed_dict[key]))
		maxes.append(np.nanmax(processed_dict[key]))
		stds.append(np.nanstd(processed_dict[key]))
		perc.append(np.nanpercentile(processed_dict[key], 99.9))

	stats_df = pd.DataFrame({'mean':means, 'max':maxes,
								'std':stds, '99th':perc},
								index=processed_dict.keys())

	return stats_df


def CNN(input_shape, loss='mse', early_stop_patience=20, initial_filters=32, learning_rate=1e-05):
	'''
	Initializing our model

	Args:
		loss (str, optional): loss function to be uesd for training. Defaults to 'categorical_crossentropy'.
		early_stop_patience (int, optional): number of epochs the model will continue training once there
												is no longer val loss improvements. Defaults to 3.

	Returns:
		object: model configuration ready for training
		object: early stopping conditions
	'''


	model = Sequential()						# initalizing the model

	model.add(Conv2D(initial_filters, 3, padding='same',
								activation='relu', input_shape=(input_shape[1], input_shape[2], 1)))			# adding the CNN layer
	model.add(MaxPooling2D(2))
	model.add(Dropout(0.2))
	model.add(Conv2D(initial_filters*2, 3, padding='same', activation='relu'))
	model.add(MaxPooling2D(2))
	model.add(Dropout(0.2))
	model.add(Conv2D(initial_filters*4, 3, padding='same', activation='relu'))
	model.add(MaxPooling2D(2))
	model.add(Dropout(0.2))
	model.add(Conv2D(initial_filters*4, 3, padding='same', activation='relu'))
	model.add(MaxPooling2D(2))
	model.add(Flatten())							# changes dimensions of model. Not sure exactly how this works yet but improves results
	model.add(Dense(initial_filters*4, activation='relu'))		# Adding dense layers with dropout in between
	model.add(Dropout(0.2))
	model.add(Dense(initial_filters*2, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='linear'))
	le_scheduler = tf.keras.experimental.CosineDecay(learning_rate, 1000, alpha=0.001) # learning rate scheduler
	opt = tf.keras.optimizers.Adam(learning_rate=le_scheduler, clipvalue=3.0)		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=loss, metrics = ['accuracy'])					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting

	return model, early_stop



def fit_CNN(model, xtrain, xval, ytrain, yval, early_stop, CV, delay, first_time=True):
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

		model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
					verbose=1, shuffle=True, epochs=200, callbacks=[early_stop],
					batch_size=128)			# doing the training! Yay!

		# saving the model
		model.save(f'models/delay_{delay}/CV_{CV}.h5')

	if not first_time:

		# loading the model if it has already been trained.
		model = load_model(f'models/delay_{delay}/CV_{CV}.h5')				# loading the models if already trained

	return model


def making_predictions(model, Xtest, CV):
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

	predicted = model.predict(Xtest, verbose=1)						# predicting on the testing input data
	predicted = tf.gather(predicted, 0, axis=1)					# grabbing the positive node
	predicted = predicted.numpy()									# turning to a numpy array

	return predicted


def calculating_errors(CV_dict):

	for CV in CV_dict.keys():
		print(CV_dict[CV]['predicted'].shape)
		print(CV_dict[CV]['test_y'].shape)
		CV_dict[CV]['rmse'] = np.sqrt(mean_squared_error(CV_dict[CV]['predicted'], CV_dict[CV]['test_y']))

	CV_dict['mean_rmse'] = np.mean([CV_dict[CV]['rmse'] for CV in CV_dict.keys()])

	return CV_dict


def plotting_results(CV_dict):

	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	for CV in CV_dict.keys():
		# print(CV_dict[CV]['predicted'].shape)
		# print(CV_dict[CV]['test_y'].shape)
		ax.scatter(CV_dict[CV]['predicted'], CV_dict[CV]['test_y'], marker='o', label=f'CV_{CV}')

	ax.plot([0, 100], [0, 100], 'k--')
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Observed')
	ax.set_title(f'CV RMSE: {CV_dict["mean_rmse"]}')
	ax.legend()

	plt.savefig('plots/CV_predicted_vs_observed.png')


def main():

	delay_stats_dict = {}

	delays = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60]

	for delay in delays:

		region = 'region_163'

		regions, stats = loading_dicts()

		maps = loading_twins_maps()

		map_keys = list(maps.keys())

		if not os.path.exists(f'models/delay_{delay}'):
			os.makedirs(f'models/delay_{delay}')

		combined_stations = combining_regional_dfs(regions[region]['station'], stats[region]['max_rsd'], map_keys, delay=delay)

		CrossValidations = 5
		CV_dict = {}

		for cv in range(CrossValidations):

			# if 'MODEL' in locals():
			# 	reset_keras(MODEL)

			keras.backend.clear_session()

			processed_dict, stats_df = preparing_data_for_modeling(combined_stations, maps, 'reg_max', cv)

			MODEL, early_stop = CNN(input_shape=processed_dict['train_x'].shape, loss='mse', early_stop_patience=20, initial_filters=64, learning_rate=1e-05)

			if os.path.exists(f'models/delay_{delay}/CV_{cv}.h5'):
				first_time=False
			else:
				first_time=True
			MODEL = fit_CNN(MODEL, processed_dict['train_x'], processed_dict['val_x'], processed_dict['train_y'],
							processed_dict['val_y'], early_stop, cv, delay=delay, first_time=first_time)

			predicted = making_predictions(MODEL, processed_dict['test_x'], cv)

			CV_dict[f'CV_{cv}'] = {'predicted':predicted, 'test_y':processed_dict['test_y']}


		CV_dict = calculating_errors(CV_dict)

		delay_stats_dict[f'delay_{delay}'] = CV_dict
	with open('outputs/delay_stats_dict.pkl', 'wb') as f:
		pickle.dump(delay_stats_dict, f)

	# plotting_results(CV_dict)

if __name__ == '__main__':
	main()