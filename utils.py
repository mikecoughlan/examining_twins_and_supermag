import gc
import glob
import math
import os
import pickle
from datetime import datetime
from functools import partial
from multiprocessing import Manager, Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from dateutil import parser
# from geopack import geopack, t89
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from tqdm import tqdm

os.environ["CDF_LIB"] = "~/CDF/lib"

twins_dir = '../data/twins/'
supermag_dir = '../data/supermag/'
regions_dict = '../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = '../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'

region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]


def loading_dicts():
	'''
	Loads the regional dictionaries and stats dictionaries

	Returns:
		regions (dict): dictionary containing the regional dictionaries
		stats (dict): dictionary containing the regional stats dictionaries including rsd and mlt data
	'''

	print('Loading regional dictionaries....')

	with open(regions_dict, 'rb') as f:
		regions = pickle.load(f)

	regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in region_numbers}

	with open(regions_stat_dict, 'rb') as g:
		stats = pickle.load(g)

	stats = {f'region_{reg}': stats[f'region_{reg}'] for reg in region_numbers}

	return regions, stats


def loading_twins_maps():
	'''
	Loads the twins maps

	Returns:
		maps (dict): dictionary containing the twins maps
	'''


	print('Loading twins maps....')
	times = pd.read_feather('outputs/regular_twins_map_dates.feather')
	twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

	maps = {}

	for file in twins_files:
		twins_map = pycdf.CDF(file)
		for i, date in enumerate(twins_map['Epoch']):
			if len(np.unique(twins_map['Ion_Temperature'][i][50:140,40:100])) == 1:
				continue
			check = pd.to_datetime(date.strftime(format='%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
			if check in times.values:
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')] = {}
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')]['map'] = twins_map['Ion_Temperature'][i][35:125,40:140]

	return maps


def loading_algorithm_maps():

	with open('outputs/twins_algo_dict.pkl', 'rb') as f:
		maps = pickle.load(f)

	times = pd.read_feather('outputs/regular_twins_map_dates.feather')

	new_maps = {}
	for date, entry in maps.items():
		if date in times.values:
			date = date.strftime(format('%Y-%m-%d %H:%M:%S'))
			new_maps[date] = {}
			new_maps[date]['map'] = entry[35:125,40:140]

	return new_maps


def loading_solarwind():
	'''
	Loads the solar wind data

	Returns:
		df (pd.dataframe): dataframe containing the solar wind data
	'''

	print('Loading solar wind data....')
	df = pd.read_feather('../data/SW/ace_data.feather')
	df.set_index('ACEepoch', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')

	return df


def loading_supermag(station):
	'''
	Loads the supermag data

	Args:
		station (string): station of interest

	Returns:
		df (pd.dataframe): dataframe containing the supermag data with a datetime index
	'''

	print(f'Loading station {station}....')
	df = pd.read_feather(supermag_dir+station+'.feather')

	# limiting the analysis to the nightside
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')

	return df


def getting_mean_lat(stations):

	# getting the mean latitude of the stations
	latitudes = []
	for station in stations:
		stat = loading_supermag(station)
		latitudes.append(stat['MLAT'].mean())

	mean_lat = np.mean(latitudes)

	return mean_lat


def combining_regional_dfs(stations, rsd, map_keys=None):
	'''
	Combines the regional data into one dataframe

	Args:
		stations (list): list of stations in the region
		rsd (pd.dataframe): dataframe containing the rsd and mlt data for the region
		map_keys (list): list of keys for the twins maps

	Returns:
		segmented_df (pd.dataframe): dataframe containing the regional dbdt, rsd, and mlt data for a given date
	'''

	print('Combining regional data....')
	start_time = pd.to_datetime('2009-07-20')
	end_time = pd.to_datetime('2017-12-31')
	twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

	combined_stations = pd.DataFrame(index=twins_time_period)

	for station in stations:
		stat = loading_supermag(station)
		stat = stat[start_time:end_time]
		stat = stat[['dbht']]

		stat[f'{station}_dbdt'] = stat['dbht']
		combined_stations = pd.concat([combined_stations, stat[f'{station}_dbdt']], axis=1, ignore_index=False)

	mean_dbht = combined_stations.mean(axis=1)
	max_dbht = combined_stations.max(axis=1)

	indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)

	combined_stations['reg_mean'] = mean_dbht
	combined_stations['reg_max'] = max_dbht
	combined_stations['rolling_max'] = max_dbht.rolling(indexer, min_periods=1).max()
	combined_stations['rsd'] = rsd['max_rsd']['max_rsd']
	combined_stations['rolling_rsd'] = rsd['max_rsd']['max_rsd'].rolling(indexer, min_periods=1).max()
	# combined_stations['rsd'] = rsd['max_rsd']['max_rsd']
	combined_stations['MLT'] = rsd['max_rsd']['MLT']

	if map_keys is not None:
		segmented_df = combined_stations[combined_stations.index.isin(map_keys)]
		return segmented_df

	else:
		return combined_stations


def calculate_percentiles(df, param, mlt_span, percentile):

	# splitting up the regions based on MLT value into 1 degree bins
	mlt_bins = np.arange(0, 24, mlt_span)
	mlt_perc = {}
	for mlt in mlt_bins:
		mlt_df = df[df['MLT'].between(mlt, mlt+mlt_span)]
		mlt_df.dropna(inplace=True, subset=[param])
		mlt_perc[f'{mlt}'] = mlt_df[param].quantile(percentile)

	return mlt_perc


def splitting_and_scaling(input_array, target_array, dates=None, scaling_method='standard', test_size=0.2, val_size=0.25, random_seed=42):
		'''
		Splits the data into training, validation, and testing sets and scales the data.

		Args:
			scaling_method (string): scaling method to use for the solar wind and supermag data.
									Options are 'standard' and 'minmax'. Defaults to 'standard'.
			test_size (float): size of the testing set. Defaults to 0.2.
			val_size (float): size of the validation set. Defaults to 0.25. This equates to a 60-20-20 split for train-val-test
			random_seed (int): random seed for reproducibility. Defaults to 42.

		Returns:
			np.array: training input array
			np.array: testing input array
			np.array: validation input array
			np.array: training target array
			np.array: testing target array
			np.array: validation target array
		'''

		if dates is not None:
			x_train, x_test, y_train, y_test, dates_train, dates_test = train_test_split(input_array, target_array, dates, test_size=test_size, random_state=random_seed)
			x_train, x_val, y_train, y_val, dates_train, dates_val = train_test_split(x_train, y_train, dates_train, test_size=val_size, random_state=random_seed)

			dates_dict = {'train':dates_train, 'test':dates_test, 'val':dates_val}
		else:
			x_train, x_test, y_train, y_test = train_test_split(input_array, target_array, test_size=test_size, random_state=random_seed)
			x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_seed)


		# defining the TWINS scaler
		if scaling_method == 'standard':
			scaler = StandardScaler()
		elif scaling_method == 'minmax':
			scaler = MinMaxScaler()
		else:
			raise ValueError('Must specify a valid scaling method for TWINS. Options are "standard" and "minmax".')

		# scaling the TWINS data
		x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
		x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
		x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)

		return x_train, x_test, x_val, y_train, y_test, y_val, dates_dict



def classification_column(df, param, thresh, forecast, window):
		'''
		Creating a new column which labels whether there will be a crossing of threshold
			by the param selected in the forecast window.

		Args:
			df (pd.dataframe): dataframe containing the param values.
			param (str): the paramaeter that is being examined for threshold crossings (dBHt for this study).
			thresh (float or list of floats): threshold or list of thresholds to define parameter crossing.
			forecast (int): how far out ahead we begin looking in minutes for threshold crossings.
								If forecast=30, will begin looking 30 minutes ahead.
			window (int): time frame in which we look for a threshold crossing starting at t=forecast.
								If forecast=30, window=30, we look for threshold crossings from t+30 to t+60

		Returns:
			pd.dataframe: df containing a bool column called crossing and a persistance colmun
		'''


		df[f'shifted_{param}'] = df[param].shift(-forecast)					# creates a new column that is the shifted parameter. Because time moves foreward with increasing

		if window > 0:																				# index, the shift time is the negative of the forecast instead of positive.
			indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)			# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
			df['window_max'] = df[f'shifted_{param}'].rolling(indexer, min_periods=1).max()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
		# df['pers_max'] = df[param].rolling(0, min_periods=1).max()						# looks backwards to find the max param value in the time history limit
		else:
			df['window_max'] = df[f'shifted_{param}']
		# df.reset_index(drop=False, inplace=True)											# resets the index

		'''This section creates a binary column for each of the thresholds. Binary will be one if the parameter
			goes above the given threshold, and zero if it does not.'''

		conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions
		# pers_conditions = [(df['pers_max'] < thresh), (df['pers_max'] >= thresh)]			# defining the conditions for a persistance model

		binary = [0, 1] 																	# 0 if not cross 1 if cross

		df['classification'] = np.select(conditions, binary)						# new column created using the conditions and the binary
		# df['persistance'] = np.select(pers_conditions, binary)				# creating the persistance column

		# df.drop(['pers_max', 'window_max', f'shifted_{param}'], axis=1, inplace=True)			# removes the working columns for memory purposes
		df.drop(['window_max', f'shifted_{param}'], axis=1, inplace=True)			# removes the working columns for memory purposes

		return df