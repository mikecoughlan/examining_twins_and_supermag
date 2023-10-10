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


def combining_regional_dfs(stations, rsd, map_keys):
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
	combined_stations['rsd'] = rsd['max_rsd']['max_rsd'].rolling(indexer, min_periods=1).max()
	combined_stations['MLT'] = rsd['max_rsd']['MLT']

	segmented_df = combined_stations[combined_stations.index.isin(map_keys)]

	return segmented_df
