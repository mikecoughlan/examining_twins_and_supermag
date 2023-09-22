####################################################################################
#
# examining_twins_and_supermag/data_prep.py
#
# The file contains teh class for preparing the data for the modeling. The class
# will be used in the modeling_v0.py and modeling_v1.py files. This process uses
# solar wind data from the ACE satellite that has been linearly interpolated up to
# 15 minutes as in Coughlan et al.(2023).
#
####################################################################################


# Importing the libraries

import glob
import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf

os.environ["CDF_LIB"] = "~/CDF/lib"

pd.options.mode.chained_assignment = None  # default='warn'

# stops this program from hogging all of the gpu
physical_devices = tf.config.list_physical_devices("GPU")
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized.
	pass



class DataPrep:

	def __init__(self, region_path, region_number, solarwind_path, supermag_dir_path, twins_times_path,
					rsd_path, random_seed):
		'''
		Initialization function for the class.

		Args:
			region_path (string): path to the region data
			region_number (string): the region number of interest
			solar_wind_path (string): path to the solar wind data
			supermag_dir_path (string): path to the supermag data
			twins_times_path (string): path to the twins timestamps
			rsd_path (string): path to the rsd data
			random_seed (int): random seed for reproducibility
		'''

		self.region_path = region_path
		self.region_number = region_number
		self.solarwind_path = solarwind_path
		self.supermag_dir_path = supermag_dir_path
		self.twins_times_path = twins_times_path
		self.rsd_path = rsd_path
		self.random_seed = random_seed


	def loading_global_data(self, load_twins=False, twins_dir=None, twins_col_limits=[50,140], twins_row_limits=[40, 100]):
		'''
		Loads the global data for the modeling.

		Args:
			load_twins (bool): whether or not to load the TWINS data
			twins_dir (string): path to the TWINS data. Required if load_twins == True
			twins_col_limits (list): column limits for the TWINS data with the first featent the min and the second the max
			twins_row_limits (list): row limits for the TWINS data with the first featent the min and the second the max
		'''

		# loading the region data
		with open(self.region_path, 'rb') as f:
			self.region_data = pickle.load(f)

		# taking only the region of interest
		self.region_data = self.region_data[f'region_{self.region_number}']

		# defining the stations int eh region of interest
		self.stations = self.region_data['station']

		# loading the solar wind data and setting datetime index
		self.solarwind_data = pd.read_feather(self.solarwind_path)
		self.solarwind_data.set_index('Date_UTC', inplace=True, drop=True)
		self.solarwind_data.index = pd.to_datetime(self.solarwind_data.index)

		# loading the rsd data for the region of interest
		with open(self.rsd_path, 'rb') as f:
			self.rsd = pickle.load(f)
		self.rsd = self.rsd[self.region_number]['max_rsd']

		# loading the twins times
		self.twins_times = pd.read_csv(self.twins_times_path)

		# Loading the TWINS maps if load_twins == True:
		if load_twins:
			if twins_dir == None:
				raise ValueError('TWINS directory not specified. TWINS directory must be specified if load_twins == True')
			twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

			self.maps = {}

			for file in twins_files:
				twins_map = pycdf.CDF(file)
				for i, date in enumerate(twins_map['Epoch']):
					if len(np.unique(twins_map['Ion_Temperature'][i][twins_col_limits[0]:twins_col_limits[1],twins_row_limits[0]:twins_row_limits[1]])) == 1:
						continue
					check = pd.to_datetime(date.strftime(format='%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
					if check in self.twins_times.values:
						self.maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')] = \
							twins_map['Ion_Temperature'][i][twins_col_limits[0]:twins_col_limits[1],twins_row_limits[0]:twins_row_limits[1]]



	def combining_regional_dfs(self, features, delay=False, delay_time=0, delay_columns='all',
								rolling=False, rolling_window=None, rolling_vars=None, data_manipulations=None,
								to_drop=None):
		'''
		Combines the regional dataframes into one dataframe for modeling.

		Args:
			features (list): list of features to include in the dataframe
			delay (bool): whether or not to delay certain columns of the dataframe
			delay_time (int): number of minutes to delay the columns
			delay_columns (list): list of columns to delay. Only used if delay == True. If 'all' then all columns are delayed.
			rolling (bool): whether or not to apply a rolling window to the dataframe. Applies maximum value to the window.
							Only used if rolling == True. Uses forward rolling indexer. Will need to be re-written to inclue
							backward rolling indexer and use of the mean instead of the max value in the window.
			rolling_window (int): number of minutes to use for the rolling window. Only used if rolling == True.
			rolling_vars (list): list of columns to apply the rolling window to. Only used if rolling == True.
			data_manipulations (string or list of strings): list of data manipulations to apply to the dataframe such as
								"mean", "std", "max" etc.
			to_drop (list): list of columns to drop from the dataframe.

		Returns:
			pd.DataFrame: dataframe containing the regional data
		'''

		if 'dbht' not in features:
			features.append('dbht')

		start_time = pd.to_datetime('2009-07-20')
		end_time = pd.to_datetime('2017-12-31')
		twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

		self.regional_dataframe = pd.DataFrame(index=twins_time_period)

		formatting_dict = {}
		for feature in features:
			formatting_dict[feature] = pd.DataFrame(index=twins_time_period)

		for station in self.stations:
			stat = pd.read_feather(self.supermag_dir_path+station+'.feather')
			stat.set_index('Date_UTC', inplace=True, drop=True)
			stat.index = pd.to_datetime(stat.index, format='%Y-%m-%d %H:%M:$S')
			stat = stat[start_time:end_time]
			stat = stat[features]

			if delay:
				if delay_columns == 'all':
					for col in stat.columns:
						formatting_dict[feat][f'{station}_{feat}'] = stat[feat].shift(-delay_time)
				else:
					for feat in features:
						if feat in delay_columns:
							formatting_dict[feat][f'{station}_{feat}'] = stat[feat].shift(-delay_time)
						else:
							formatting_dict[feat][f'{station}_{feat}'] = stat[feat]

		if data_manipulations != None:
			for feature in features:
				temp_df = formatting_dict[feature].aggregate(data_manipulations, axis=1)
				temp_df.columns = [f'{feature}_{col}' for col in temp_df.columns]
				self.regional_dataframe = pd.concat([self.regional_dataframe, temp_df], axis=1, ignore_index=False)

		self.regional_dataframe['max_rsd'] = self.rsd['max_rsd']
		self.regional_dataframe['MLT'] = self.rsd['MLT']

		# sin and cos MLT are used to avoid the 23 -> 0 hard drop which the model may have trouble with
		self.regional_dataframe['sinMLT'] = np.sin(self.regional_dataframe.MLT * 2 * np.pi * 15 / 360)
		self.regional_dataframe['cosMLT'] = np.cos(self.regional_dataframe.MLT * 2 * np.pi * 15 / 360)

		if rolling:
			indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=rolling_window)
			for var in rolling_vars:
				self.regional_dataframe[f'shifted_{var}'] = self.regional_dataframe[var].rolling(indexer, min_periods=1).max()

		if to_drop != None:
			self.regional_dataframe.drop(to_drop, axis=1, inplace=True)

		return self.regional_dataframe


	def time_shifting_solarwind(self, omni_or_ace, features, rolling=False, rolling_window=None, rolling_vars=None, rolling_value='mean'):

		'''
		Shifts the solar wind data by a specified amount of time. Forward shift time will
		depend on whether ACE data is used or OMNI data is used because of the propogation time
		to the bow shock done by OMNI. May default to OMNI here to eliminate the problem with
		figuring out propogation time. Need to limit forecasting time due to TWINS maps proximity
		to the Earth.

		Args:
			omni_or_ace (string): whether to use OMNI or ACE data
			rolling (bool): whether or not to apply a rolling window to the dataframe. Applies rolling_value value to the window.
			rolling_window (int): number of minutes to use for the rolling window. Only used if rolling == True.
			rolling_vars (list): list of columns to apply the rolling window to. Only used if rolling == True.
			rolling_value (string): transformation to apply to the rolling window. Only used if rolling == True. Defaults to mean.

		Returns:
			pd.DataFrame: dataframe containing the shifted solar wind data
		'''

		# need to fully think out these time shifts.
		if omni_or_ace == 'omni':
			time_shift = 10
		elif omni_or_ace == 'ace':
			time_shift = 40
		else:
			raise ValueError('Must specify whether to use OMNI or ACE data')

		self.solarwind_data = self.solarwind_data[features]
		self.solarwind_data = self.solarwind_data.shift(-time_shift)

		if rolling:
			if rolling_vars == 'all':
				rolling_vars = self.solarwind_data.columns
			for var in rolling_vars:
				if rolling_value == 'mean':
					print("You're using a rolling mean....")
					self.solarwind_data[f'{var}_rolling_mean'] = self.solarwind_data[var].rolling(window=rolling_window, min_periods=1).mean()
				elif rolling_value == 'max':
					print("You're using a rolling max....")
					self.solarwind_data[f'{var}_rolling_max'] = self.solarwind_data[var].rolling(window=rolling_window, min_periods=1).max()
				elif rolling_value == 'std':
					print("You're using a rolling std....")
					self.solarwind_data[f'{var}_rolling_std'] = self.solarwind_data[var].rolling(window=rolling_window, min_periods=1).std()


		return self.solarwind_data


	def split_sequences(self, df, target_var=None, n_steps=30):
		'''
			Takes input from the input array and creates the input and target arrays that can go into the models.

			Args:
				sequences (np.array): input features. Shape = (length of data, number of input features)
				results_y: series data of the targets for each threshold. Shape = (length of data, 1)
				n_steps (int): the time history that will define the 2nd demension of the resulting array.
				include_target (bool): true if there will be a target output. False for the testing data.

			Returns:
				np.array (n, time history, n_features): array for model input
				np.array (n, 1): target array
			'''
		df.reset_index(inplace=True, drop=False)			# resetting the index of the dataframe

		# Getting the index values of the df based on maching the
		# Date_UTC column and the value fo the twins_dates series. '''
		indices = df.index[df['Date_UTC'].isin(self.twins_times['dates'])].tolist()
		sequences = df.copy()
		target = sequences[target_var]
		sequences.drop(['Date_UTC', target_var], axis=1, inplace=True)

		X, y1, bad_dates = list(), list(), list()							# creating lists for storing results
		for i in indices:			# going to the end of the dataframes
			beginning_ix = i - n_steps						# find the end of this pattern
			if beginning_ix < 0:					# check if we are beyond the dataset
				raise ValueError('Time history goes below the beginning of the dataset')
			seq_x = sequences[beginning_ix:i, :]				# grabs the appropriate chunk of the data
			if target != None:
				if np.isnan(seq_x).any():				# doesn't add arrays with nan values to the training set
					print(f'nan values in the input array for {df["Date_UTC"][i]}')
					bad_dates.append(df['Date_UTC'][i])
					continue
			if target != None:
				seq_y1 = target[i]				# gets the appropriate target
				y1.append(seq_y1)
			X.append(seq_x)

		if target != None:
			return np.array(X), np.array(y1), bad_dates
		else:
			return np.array(X)


	def combining_supermag_and_solarwind_data(self, target_var, time_history=30):

		'''
		Combines the regional dataframe with the solar wind data. Data shoudl have already been shifted such that the
		appropriate lead and delay times have been factored in given that all times are centered on the TWINS timestamps.
		For instance, if the TWINS timestamp is at time t, and we are predicting supermag conditions at time t+11, then
		the supermag data should have been shifted by -11 minutes so the datetimes line up. Similarly, if the OMNI data
		is being used, then the OMNI data should have been shifted by +10 minutes if we are thinking it takes 10 minutes
		for the solar wind at the Bow Shock to impact the tail region.

		Args:
			target_var (string): target variable for the modeling.
			time_history (int): number of minutes of solar wind data to include in the dataframe

		Returns:
			pd.DataFrame: dataframe containing the regional data and solar wind data
		'''

		# combining the solarwind and supermag data.
		supermag_and_solarwind_data = pd.concat([self.regional_dataframe, self.solarwind_data], axis=1, ignore_index=False)

		print(supermag_and_solarwind_data.columns)

		self.X, self.y, self.bad_dates = self.split_sequences(supermag_and_solarwind_data, target_var=target_var, n_steps=time_history)

		return self.X, self.y, self.bad_dates



	def splitting_and_scaling(self, solarwind_and_supermag_scaling_method='standard',
								test_size=0.2, val_size=0.25, include_twins=False,
								twins_scaling_method='standard'):
		'''
		Splits the data into training, validation, and testing sets and scales the data.

		Args:
			solarwind_and_supermag_scaling_method (string): scaling method to use for the solar wind and
															supermag data. Options are 'standard' and 'minmax'.
															Defaults to 'standard'.
			test_size (float): size of the testing set. Defaults to 0.2.
			val_size (float): size of the validation set. Defaults to 0.25. This equates to a 60-20-20 split for train-val-test
			include_twins (bool): whether or not split and scale the TWINS data. Defaults to False.
			twins_scaling_method (string): scaling method to use for the TWINS data. Options are 'standard' and 'minmax'.

		Returns:
			np.array: training input array
			np.array: testing input array
			np.array: validation input array
			np.array: training target array
			np.array: testing target array
			np.array: validation target array
		'''

		if include_twins:

			# need to eliminate the maps that were skipped in the split sequences functions
			map_keys = list(self.maps.keys())
			twins_arrays = [self.maps[i] for i in map_keys if i not in self.bad_dates]
			twins_arrays = np.array(twins_arrays)

			# splitting the TWINS data into training, testing, and validation sets
			twins_x_train, twins_x_test = train_test_split(twins_arrays, test_size=test_size, random_state=self.random_seed)
			twins_x_train, twins_x_val = train_test_split(twins_x_train, test_size=val_size, random_state=self.random_seed)

			# defining the TWINS scaler
			if twins_scaling_method == 'standard':
				self.twins_scaler = StandardScaler()
			elif twins_scaling_method == 'minmax':
				self.twins_scaler = MinMaxScaler()
			else:
				raise ValueError('Must specify a valid scaling method for TWINS. Options are "standard" and "minmax".')

			# scaling the TWINS data
			self.twins_x_train = self.twins_scaler.fit_transform(twins_x_train)
			self.twins_x_test = self.twins_scaler.transform(twins_x_test)
			self.twins_x_val = self.twins_scaler.transform(twins_x_val)

		# splitting the solar wind and supermag data into training, testing, and validation sets
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
		self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=random_state)

		# defining the solar wind and supermag scaler
		if solarwind_and_supermag_scaling_method == 'standard':
			self.scaler = StandardScaler()
		elif solarwind_and_supermag_scaling_method == 'minmax':
			self.scaler = MinMaxScaler()
		else:
			raise ValueError('Must specify a valid scaling method for solarwind and supermag. Options are "standard" and "minmax".')

		# scaling the solar wind and supermag data
		self.X_train = self.scaler.fit_transform(self.X_train)
		self.X_test = self.scaler.transform(self.X_test)
		self.X_val = self.scaler.transform(self.X_val)

		if include_twins:
			return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, self.twins_x_train, self.twins_x_test, self.twins_x_val
		else:
			return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val


	def twins_only_data_prep(self, config=None):

		if config == None:
			raise ValueError('Must specify a config file or variable dictionary.')

		# loading gloabl data
		self.loading_global_data(load_twins=config['load_twins'], twins_dir=config['twins_dir'],
									twins_col_limits=config['twins_col_limits'], twins_row_limits=config['twins_row_limits'])

		# splitting and scaling the data
		self.splitting_and_scaling(solarwind_and_supermag_scaling_method=config['solarwind_and_supermag_scaling_method'],
									test_size=config['test_size'], val_size=config['val_size'],
									include_twins=config['include_twins'],
									twins_scaling_method=config['twins_scaling_method'])


		return self.twins_x_train, self.twins_x_val, self.twins_x_test


	def do_full_data_prep(self, config=None):

		if config == None:
			raise ValueError('Must specify a config file or variable dictionary.')

		# loading gloabl data
		self.loading_global_data(load_twins=config['load_twins'], twins_dir=config['twins_dir'],
									twins_col_limits=config['twins_col_limits'], twins_row_limits=config['twins_row_limits'])

		# combining regional dataframes
		self.regional_dataframes = self.combining_regional_dfs(features=config['mag_features'], delay=config['delay'], delay_time=config['delay_time'],
									delay_columns=config['delay_columns'], rolling=config['rolling'],
									rolling_window=config['rolling_window'], rolling_vars=config['rolling_vars'],
									data_manipulations=config['data_manipulations'], to_drop=config['to_drop'])

		# shifting the solar wind data
		self.solarwind_data = self.time_shifting_solarwind(omni_or_ace=config['omni_or_ace'], features=config['sw_features'],
										rolling=config['sw_rolling'], rolling_window=config['sw_rolling_window'],
										rolling_vars=config['sw_rolling_vars'], rolling_value=config['rolling_value'])

		# combining the supermag and solar wind data
		self.X, self.y, self.bad_dates = self.combining_supermag_and_solarwind_data(target_var=config['target_var'], time_history=config['time_history'])

		# splitting and scaling the data
		self.splitting_and_scaling(solarwind_and_supermag_scaling_method=config['solarwind_and_supermag_scaling_method'],
									test_size=config['test_size'], val_size=config['val_size'],
									include_twins=config['include_twins'],
									twins_scaling_method=config['twins_scaling_method'])


		if config['include_twins']:
			return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, self.twins_x_train, self.twins_x_test, self.twins_x_val

		else:
			return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
