##########################################################################
#
# File for looking at the correlations between the TWINS maps and ground
# mag data as a function of delay time.
#
##########################################################################



import gc
import os

import scipy
from scipy.io import netcdf_file

os.environ["CDF_LIB"] = "~/CDF/lib"
import datetime
import glob
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacepy
from spacepy import pycdf
from tqdm import tqdm

twins_dir = '../data/twins/'
supermag_dir = '../data/supermag/'
regions_dict = '../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = '../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'


region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]


def loading_twins_times():

	times = pd.read_feather('outputs/regular_twins_map_dates.feather')

	return times


def loading_twins_maps():

	times = loading_twins_times()
	times = pd.to_datetime(times['dates'], format='%Y-%m-%d %H:%M:%S')

	twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

	maps = {}

	for file in twins_files:
		twins_map = pycdf.CDF(file)
		for i, date in enumerate(twins_map['Epoch']):
			check = pd.to_datetime(date.strftime(format='%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
			if check in times.values:
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')] = twins_map['Ion_Temperature'][i][:,60:]

	stats_df = pd.DataFrame(index=maps.keys())
	stats_df.index = pd.to_datetime(stats_df.index)
	means, maxs, perc, std = [], [], [], []
	for arr in maps.values():
		arr[arr ==-1] = np.nan
		means.append(np.nanmean(arr))
		std.append(np.nanstd(arr))
		perc.append(np.nanpercentile(arr, 99))
		maxs.append(np.nanmax(arr))

	stats_df['mean'] = means
	stats_df['max'] = maxs
	stats_df['std'] = std
	stats_df['perc'] = perc

	maps['stats_df'] = stats_df

	return maps


def loading_regions_dicts():

	with open(regions_dict, 'rb') as f:
		regions = pickle.load(f)

	segmented_regions = {f'region_{reg}':regions[f'region_{reg}'] for reg in region_numbers}

	return segmented_regions


def loading_supermag(station, start_time, end_time):

	df = pd.read_feather(supermag_dir+station+'.feather')

	# limiting the analysis to the nightside
	df = df[(df['MLT']>17) | (df['MLT']<7)]
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
	df = df[start_time:end_time]

	return df


def combining_regional_dfs(stations, latitude=False):

	start_time = pd.to_datetime('2010-01-01')
	end_time = pd.to_datetime('2017-12-31')
	twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

	combined_stations = pd.DataFrame(index=twins_time_period)

	mlats = []

	for station in stations:
		stat = loading_supermag(station, start_time, end_time)
		mlats.append(stat['MLAT'].mean())
		combined_stations = pd.concat([combined_stations, stat['dbht']], axis=1, ignore_index=False)

	region_mlat = round((np.max(mlats) + np.min(mlats)) / 2, 2)

	mean_dbht = combined_stations.mean(axis=1)
	max_dbht = combined_stations.max(axis=1)

	combined_stations['reg_mean'] = mean_dbht
	combined_stations['reg_max'] = max_dbht

	if latitude:
		combined_stations = combined_stations[['reg_mean', 'reg_max']]

	return combined_stations, region_mlat


def calculating_correlations(regions, maps, delays):

	corrs_dict = {}

	for region in regions:
		combined_df, region_mlat = combining_regional_dfs(regions[region]['station'])


		mag_stats = ['max', 'mean']
		map_stats = ['max', 'mean', 'std', 'perc']

		corrs_dict[region] = {}
		corrs_dict[region]['mlat'] = region_mlat

		for mag_stat in mag_stats:
			for map_stat in map_stats:
				corrs_dict[region][f'{mag_stat}-{map_stat}'] = []

		for delay in delays:
			'''creates a new column that is the shifted parameter. Because
			time moves foreward with increasing index, the shift time is the
			negative of the forecast instead of positive.'''

			combined_df[f'shifted_mean_{delay}'] = combined_df['reg_mean'].shift(-delay)
			combined_df[f'shifted_max_{delay}'] = combined_df['reg_max'].shift(-delay)

			# modify these lines to look at the max in a rolling window
			indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=60)			# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
			combined_df[f'shifted_mean_{delay}_max'] = combined_df[f'shifted_mean_{delay}'].rolling(indexer, min_periods=1).max()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
			combined_df[f'shifted_max_{delay}_max'] = combined_df[f'shifted_max_{delay}'].rolling(indexer, min_periods=1).max()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
			# df.reset_index(drop=True, inplace=True)

			common_index = maps['stats_df'].index.intersection(combined_df.index)

			for mag_stat in mag_stats:
				for map_stat in map_stats:
					corr = maps['stats_df'].loc[common_index, map_stat].corr(combined_df.loc[common_index, f'shifted_{mag_stat}_{delay}_max'])
					corrs_dict[region][f'{mag_stat}-{map_stat}'].append(corr)

	return corrs_dict


def calculating_latitude_correlations(regions, maps, delays):

	corrs_dict = {}
	low_lat_max, mid_lat_max, high_lat_max = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
	low_lat_mean, mid_lat_mean, high_lat_mean = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

	for region in regions:
		combined_df, region_mlat = combining_regional_dfs(regions[region]['station'], latitude=True)

		if region_mlat <= 45:
			low_lat_max[region] = combined_df['reg_max']
			low_lat_mean[region] = combined_df['reg_mean']
		elif (region_mlat > 45) & ((region_mlat <= 70)):
			mid_lat_max[region] = combined_df['reg_max']
			mid_lat_mean[region] = combined_df['reg_mean']
		else:
			high_lat_max[region] = combined_df['reg_max']
			high_lat_mean[region] = combined_df['reg_mean']

	low_lat_max['mean'] = low_lat_max.mean(axis=1, skipna=True)
	mid_lat_max['mean'] = mid_lat_max.mean(axis=1, skipna=True)
	high_lat_max['mean'] = high_lat_max.mean(axis=1, skipna=True)

	low_lat_mean['mean'] = low_lat_mean.mean(axis=1, skipna=True)
	mid_lat_mean['mean'] = mid_lat_mean.mean(axis=1, skipna=True)
	high_lat_mean['mean'] = high_lat_mean.mean(axis=1, skipna=True)

	mag_stats = ['mean']
	map_stats = ['max', 'mean', 'std', 'perc']

	df_labels = ['low_max', 'low_mean', 'mid_max', 'mid_mean', 'high_max', 'high_mean']
	dfs = [low_lat_max, low_lat_mean, mid_lat_max, mid_lat_mean, high_lat_max, high_lat_mean]

	for label, df in zip(df_labels, dfs):
		corrs_dict[label] = {}

		for mag_stat in mag_stats:
			for map_stat in map_stats:
				corrs_dict[label][f'{mag_stat}-{map_stat}'] = []

		for delay in delays:
			'''creates a new column that is the shifted parameter. Because
			time moves foreward with increasing index, the shift time is the
			negative of the forecast instead of positive.'''

			df[f'shifted_mean_{delay}'] = df['mean'].shift(-delay)

			# modify these lines to look at the max in a rolling window
			indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=1)			# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
			df[f'shifted_mean_{delay}_max'] = df[f'shifted_mean_{delay}'].rolling(indexer, min_periods=1).max()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
			# df.reset_index(drop=True, inplace=True)

			common_index = maps['stats_df'].index.intersection(df.index)

			for mag_stat in mag_stats:
				for map_stat in map_stats:
					corr = maps['stats_df'].loc[common_index, map_stat].corr(df.loc[common_index, f'shifted_{mag_stat}_{delay}_max'])
					corrs_dict[label][f'{mag_stat}-{map_stat}'].append(corr)

		corrs_dict[label]['delay_df'] = df

	return corrs_dict


def plotting_corrs(corrs_dict, delays, mag_stat, latitudes=False):

	x = [i for i in range(len(delays))]

	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,15))
	colormap = plt.get_cmap('magma')
	norm_param = mcolors.Normalize(vmin=30, vmax=85)

	ax1 = plt.subplot(2,2,1)
	for region in corrs_dict.keys():
		if latitudes:
			line, = plt.plot(corrs_dict[region][f'{mag_stat}-mean'], label=region)
		else:
			line, = plt.plot(corrs_dict[region][f'{mag_stat}-mean'], label=region, color=colormap(norm_param(corrs_dict[region]['mlat'])))

	plt.title(f'{mag_stat}-mean')
	plt.margins(x=0, y=0)
	plt.xticks(x, labels=delays)
	plt.legend()

	ax2 = plt.subplot(2,2,2)
	for region in corrs_dict.keys():
		if latitudes:
			line, = plt.plot(corrs_dict[region][f'{mag_stat}-max'], label=region)
		else:
			line, = plt.plot(corrs_dict[region][f'{mag_stat}-max'], label=region, color=colormap(norm_param(corrs_dict[region]['mlat'])))
	plt.title(f'{mag_stat}-max')
	plt.margins(x=0, y=0)
	plt.xticks(x, labels=delays)

	ax3 = plt.subplot(2,2,3)
	for region in corrs_dict.keys():
		if latitudes:
			line, = plt.plot(corrs_dict[region][f'{mag_stat}-std'], label=region)
		else:
			line, = plt.plot(corrs_dict[region][f'{mag_stat}-std'], label=region, color=colormap(norm_param(corrs_dict[region]['mlat'])))
	plt.title(f'{mag_stat}-std')
	plt.margins(x=0, y=0)
	plt.xticks(x, labels=delays)

	ax4 = plt.subplot(2,2,4)
	for region in corrs_dict.keys():
		if latitudes:
			line, = plt.plot(corrs_dict[region][f'{mag_stat}-perc'], label=region)
		else:
			line, = plt.plot(corrs_dict[region][f'{mag_stat}-perc'], label=region, color=colormap(norm_param(corrs_dict[region]['mlat'])))
	plt.title(f'{mag_stat}-perc')
	plt.margins(x=0, y=0)
	plt.xticks(x, labels=delays)

	if not latitudes:
		sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm_param)
		# sm.set_array([])  # Set dummy array to create colorbar
		cbar = plt.colorbar(sm, ax=axes, location='bottom', pad=0.1)

	if latitudes:
		plt.savefig(f'plots/{mag_stat}_latitude_delay_correlations.png')
	# plt.tight_layout()
	else:
		plt.savefig(f'plots/{mag_stat}_delay_correlations.png')


def direct_plotting(regions, maps, delays):

	# for region in regions:
	combined_df, ____ = combining_regional_dfs(regions['region_163']['station'])


	mag_stats = ['max', 'mean']
	map_stats = ['max', 'mean', 'std', 'perc']

	for delay in delays:
		'''creates a new column that is the shifted parameter. Because
		time moves foreward with increasing index, the shift time is the
		negative of the forecast instead of positive.'''

		combined_df[f'shifted_mean_{delay}'] = combined_df['reg_mean'].shift(-delay)
		combined_df[f'shifted_max_{delay}'] = combined_df['reg_max'].shift(-delay)

		# modify these lines to look at the max in a rolling window
		indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=10)			# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
		combined_df[f'shifted_mean_{delay}_max'] = combined_df[f'shifted_mean_{delay}'].rolling(indexer, min_periods=1).mean()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
		combined_df[f'shifted_max_{delay}_max'] = combined_df[f'shifted_max_{delay}'].rolling(indexer, min_periods=1).mean()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
		# df.reset_index(drop=True, inplace=True)

		# for mag_stat in mag_stats:
		# 		for map_stat in map_stats:
		# 			corr = maps['stats_df'].loc[common_index, map_stat].corr(combined_df.loc[common_index, f'shifted_{mag_stat}_{delay}_max'])
		# 			corrs_dict[region][f'{mag_stat}-{map_stat}'].append(corr)

	mashing = pd.concat([combined_df, maps['stats_df']], axis=1, ignore_index=False)
	mashing.dropna(inplace=True)
	print(mashing)

	delay = 60
	start = 0
	end = 100
	fig = plt.figure(figsize=(10,10))
	# fig.autofmt_xdate(rotation=45)

	ax1 = plt.subplot(221)
	ax12 = ax1.twinx()
	ax1.plot(mashing['mean'][start:end])
	ax12.plot(mashing[f'shifted_max_{delay}_max'][start:end], color='orange')
	ax12.plot(mashing[f'shifted_mean_{delay}_max'][start:end], color='red')
	# ax1.set_xticklabels(maps['stats_df']['mean'][15500:15700].index, rotation=45)
	plt.title('Mean')

	ax2 = plt.subplot(222)
	ax22 = ax2.twinx()
	ax2.plot(mashing['max'][start:end])
	ax22.plot(mashing[f'shifted_max_{delay}_max'][start:end], color='orange')
	ax22.plot(mashing[f'shifted_mean_{delay}_max'][start:end], color='red')
	# ax2.set_xticklabels(maps['stats_df']['max'][15500:15700].index, rotation=45)
	plt.title('Max')

	ax3 = plt.subplot(223)
	ax32 = ax3.twinx()
	ax3.plot(mashing['std'][start:end])
	ax32.plot(mashing[f'shifted_max_{delay}_max'][start:end], color='orange')
	ax32.plot(mashing[f'shifted_mean_{delay}_max'][start:end], color='red')
	ax3.set_xticklabels(ax32.get_xticklabels(), rotation=45, ha='right')
	plt.title('std')

	ax4 = plt.subplot(224)
	ax42 = ax4.twinx()
	ax4.plot(mashing['perc'][start:end])
	ax42.plot(mashing[f'shifted_max_{delay}_max'][start:end], color='orange')
	ax42.plot(mashing[f'shifted_mean_{delay}_max'][start:end], color='red')
	# ax4.set_xticklabels(maps['stats_df']['perc'][15500:15700].index, rotation=45)
	plt.title('Perc')

	plt.savefig('plots/twins_stats.png')


def direct_latitude_plotting(coors_dict, maps, delays):

	delay = 60
	start = 0
	end = 100
	fig = plt.figure(figsize=(10,10))
	# fig.autofmt_xdate(rotation=45)

	ax1 = plt.subplot(221)
	ax12 = ax1.twinx()
	ax1.plot(maps['stats_df']['mean'][start:end], color='black')
	for coor in coors_dict.keys():
		ax12.plot(coors_dict[coor]['delay_df'][f'shifted_mean_{delay}_max'][start:end], label=coor)
	# ax1.set_xticklabels(maps['stats_df']['mean'][15500:15700].index, rotation=45)
	plt.title('Mean')
	plt.legend()

	ax2 = plt.subplot(222)
	ax22 = ax2.twinx()
	ax2.plot(maps['stats_df']['max'][start:end], color='black')
	for coor in coors_dict.keys():
		ax22.plot(coors_dict[coor]['delay_df'][f'shifted_mean_{delay}_max'][start:end], label=coor)
	# ax2.set_xticklabels(maps['stats_df']['max'][15500:15700].index, rotation=45)
	plt.title('Max')

	ax3 = plt.subplot(223)
	ax32 = ax3.twinx()
	ax3.plot(maps['stats_df']['std'][start:end], color='black')
	for coor in coors_dict.keys():
		ax32.plot(coors_dict[coor]['delay_df'][f'shifted_mean_{delay}_max'][start:end], label=coor)
	ax3.set_xticklabels(ax32.get_xticklabels(), rotation=45, ha='right')
	plt.title('std')

	ax4 = plt.subplot(224)
	ax42 = ax4.twinx()
	ax4.plot(maps['stats_df']['perc'][start:end], color='black')
	for coor in coors_dict.keys():
		ax42.plot(coors_dict[coor]['delay_df'][f'shifted_mean_{delay}_max'][start:end], label=coor)
	# ax4.set_xticklabels(maps['stats_df']['perc'][15500:15700].index, rotation=45)
	plt.title('Perc')

	plt.savefig('plots/twins_stats_latitude.png')

def main():

	delays = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 45, 60]
	maps = loading_twins_maps()
	regions = loading_regions_dicts()
	latitude_coors = calculating_latitude_correlations(regions, maps, delays)
	plotting_corrs(latitude_coors, delays, 'mean', latitudes=True)
	direct_latitude_plotting(latitude_coors, maps, delays)
	# corrs_dict = calculating_correlations(regions, maps, delays)
	# plotting_corrs(corrs_dict, delays, 'mean')
	# plotting_corrs(corrs_dict, delays, 'max')


	print('SOMETHING IS WRONG HERE!!!!')




if __name__ =='__main__':
	main()
