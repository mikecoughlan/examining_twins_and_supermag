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

	new_maps = {}
	for date, entry in maps.items():
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

	combined_stations['reg_mean'] = mean_dbht
	combined_stations['reg_max'] = max_dbht
	combined_stations['rsd'] = rsd['max_rsd']['max_rsd']
	combined_stations['MLT'] = rsd['max_rsd']['MLT']

	segmented_df = combined_stations[combined_stations.index.isin(map_keys)]

	return segmented_df


def get_data():


	# loading all the datasets and dictonaries
	if os.path.exists('outputs/twins_maps_with_footpoints.pkl'):
		with open('outputs/twins_maps_with_footpoints.pkl', 'rb') as f:
			twins = pickle.load(f)
	else:
		twins = loading_twins_maps()

	regions, stats = loading_dicts()
	solarwind = loading_solarwind()

	# reduce the regions dict to be only the ones that have keys in the region_numbers list
	regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in region_numbers}

	# Getting regions data for each region
	for region in regions.keys():

		# getting dbdt and rsd data for the region
		regions[region]['combined_dfs'] = combining_regional_dfs(regions[region]['station'], stats[region], twins.keys())

	# Attaching the algorithm maps to the twins dictionary
	algorithm_maps = loading_algorithm_maps()

	data_dict = {'twins_maps':twins, 'solarwind':solarwind, 'regions':regions, 'algorithm_maps':algorithm_maps}

	return data_dict


def plotting_intervls(data_dict, start_date, end_date, twins_or_algo):

	''' Function that plots either the twins maps or teh algoritm outputs for a given date range
			alongside a polar plot of the divided up rsd regional data that will be used to examine
			case studies of blobs and how they affect the ground dbdt.'''

	if not os.path.exists(f'plots/{start_date}_{end_date}_{twins_or_algo}'):
		os.makedirs(f'plots/{start_date}_{end_date}_{twins_or_algo}')

	# getting the twins maps and algorithm maps
	if twins_or_algo == 'twins':
		maps = data_dict['twins_maps']
		# getting the maps within the date range
		maps = {date:maps[date] for date in maps.keys() if datetime.strptime(date, '%Y-%m-%d %H:%M:%S') >= start_date and datetime.strptime(date, '%Y-%m-%d %H:%M:%S') <= end_date}
	elif twins_or_algo == 'algo':
		maps = data_dict['algorithm_maps']
		# getting the maps within the date range
		maps = {date:maps[date] for date in maps.keys() if datetime.strptime(date, '%Y-%m-%d %H:%M:%S') >= start_date and datetime.strptime(date, '%Y-%m-%d %H:%M:%S') <= end_date}
	else:
		raise ValueError('twins_or_algo must be either twins or algo')

	# getting the regional data
	regions = data_dict['regions']
	segmented_regions = {}
	for region in regions.keys():
		segmented_regions[region] = regions[region]['combined_dfs'][start_date:end_date]

	# splitting up the regions based on MLT value into 1 degree bins
	mlt_bins = np.arange(0, 24, 1)
	mlt_dict = {}
	for mlt in mlt_bins:
		mlt_df = pd.DataFrame(index=pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='min'))
		for region in regions.keys():
			temp_df = segmented_regions[region][segmented_regions[region]['MLT'].between(mlt, mlt+1)]
			mlt_df = pd.concat([mlt_df, temp_df['rsd']], axis=1, ignore_index=False)
		mlt_df['max'] = mlt_df.max(axis=1)
		mlt_df.dropna(inplace=True, subset=['max'])
		mlt_dict[f'{mlt}'] = mlt_df

	# setting the min and max values for the maps and the polar plot
	finding_max_rsd = [mlt_dict[key]['max'].max() for key in mlt_dict.keys()]

	map_min = 0
	map_max = 20
	polar_min = 0
	polar_max = max(finding_max_rsd)
	# plotting the maps and the regional data
	for key in maps.keys():

		# plotting maps
		fig = plt.figure(figsize=(20,10))
		plt.title(f'{key}')
		ax0=plt.subplot(121)
		plt.imshow(maps[key]['map'], cmap='jet', origin='lower', vmin=map_min, vmax=map_max)
		plt.colorbar()

		# plotting gridded max rsd
		ax1=plt.subplot(122, projection='polar')
		ax1.set_theta_zero_location("W")
		ax1.set_theta_direction(-1)
		r=1
		cmap = plt.get_cmap('jet')
		normalize = Normalize(vmin=polar_min, vmax=polar_max)
		for i, df in enumerate(mlt_dict.values()):
			theta = np.linspace(2 * np.pi * i / 24, 2 * np.pi * (i + 1) / 24, 100)
			try:
				colors = [cmap(normalize(df.loc[key]['max']))]
			except KeyError:
				continue
			# Fill the arc between specified theta values
			theta_start = 2 * np.pi * i / 24  # Adjust as needed
			theta_end = 2 * np.pi * (i + 1) / 24  # Adjust as needed

			ax1.fill_between(theta, 0, r, where=(theta >= theta_start) & (theta <= theta_end),
                    alpha=0.5, label=f'Section {i+1}', color=colors)
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
		sm.set_array([])
		cbar = plt.colorbar(sm, ax=ax1)
		cbar.set_label('Max RSD')
		plt.tight_layout()
		plt.savefig(f'plots/{start_date}_{end_date}_{twins_or_algo}/{key}.png')




def main():

	data_dict = get_data()

	start_date = pd.to_datetime('2012-03-08')
	end_date = pd.to_datetime('2012-03-10')

	plotting_intervls(data_dict, start_date, end_date, 'algo')



if __name__ == '__main__':
	main()