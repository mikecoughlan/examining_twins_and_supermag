import datetime
import gc
import glob
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
import tqdm
from spacepy import pycdf

os.environ["CDF_LIB"] = "~/CDF/lib"


twins_dir = '../data/twins/'
supermag_dir = '../data/supermag/'
regions_dict = '../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'

region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]

with open(regions_dict, 'rb') as f:
	regions = pickle.load(f)

regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in region_numbers}

times = pd.read_feather('outputs/regular_twins_map_dates.feather')

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

	return maps, maps_array


def combining_mag_station_data(stations, delay, elements):

	combined_dict = {}
	elements = [pd.to_datetime(element) for element in elements]
	for station in tqdm.tqdm(stations):
		df = pd.read_feather(f'../data/supermag/{station}.feather')
		df = assigning_color(df)
		df.set_index('Date_UTC', inplace=True)
		df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
		df = df[['MLAT', 'MLT', 'dbht', 'color']]
		for col in df.columns:
			df[f'{col}_delay_{delay}'] = df[col].shift(-delay)
		df.drop(['MLAT', 'MLT', 'dbht', 'color'], inplace=True, axis=1)
		segmented_df = pd.DataFrame()
		for element in elements:
			try:
				segmented_df = pd.concat([segmented_df, df.loc[element]], axis=0)
			except KeyError:
				continue

		combined_dict[station] = segmented_df

	return combined_dict


map_keys = list(maps.keys())
map_keys = [pd.to_datetime(key, format='%Y-%m-%d %H:%M:%S') for key in map_keys]


def combining_regional_dfs(stations, elements, delay):

	start_time = pd.to_datetime('2009-07-20')
	end_time = pd.to_datetime('2017-12-31')
	twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

	combined_stations = pd.DataFrame(index=twins_time_period)

	mlats = []

	for station in stations:
		stat = loading_supermag(station, start_time, end_time)
		combined_stations = pd.concat([combined_stations, stat['dbht']], axis=1, ignore_index=False)

	mean_dbht = combined_stations.mean(axis=1)
	max_dbht = combined_stations.max(axis=1)

	combined_stations['reg_mean'] = mean_dbht
	combined_stations['reg_max'] = max_dbht

	return combined_stations