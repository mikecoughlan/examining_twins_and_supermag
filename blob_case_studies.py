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

import utils

os.environ["CDF_LIB"] = "~/CDF/lib"

twins_dir = '../data/twins/'
supermag_dir = '../data/supermag/'
regions_dict = '../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = '../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'

region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]


def get_all_data():


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

	# Getting regions data for each region
	for region in regions.keys():

		# getting dbdt and rsd data for the region
		regions[region]['combined_dfs'] = utils.combining_regional_dfs(regions[region]['station'], stats[region], twins.keys())

	# Attaching the algorithm maps to the twins dictionary
	algorithm_maps = utils.loading_algorithm_maps()

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
		print(f'Length of mlt_df for {mlt} MLT: {len(mlt_df)}. Mean max_RSD: {mlt_df["max"].mean()}. 99th Percentile: {mlt_df["max"].quantile(0.99)}')


	total_df = [mlt_df['max'].to_numpy() for mlt_df in mlt_dict.values()]

	fig = plt.figure(figsize=(20,10))
	plt.title(f'All MLTs')
	plt.boxplot(total_df, vert=True, labels=[f'{mlt} MLT' for mlt in mlt_bins], whis=[5,95])
	plt.show()

	# setting the min and max values for the maps and the polar plot
	finding_max_rsd = [mlt_dict[key]['max'].max() for key in mlt_dict.keys()]

	map_min = 0
	map_max = 20
	polar_min = 0
	polar_max = 50
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
		ax1.set_theta_direction(1)
		ax1.set
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
                    alpha=1, label=f'Section {i+1}', color=colors)
			ax1.set_xticklabels(['0', '3', '6', '9', '12', '15', '18', '21'])
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
		sm.set_array([])
		cbar = plt.colorbar(sm, ax=ax1)
		cbar.set_label('Max RSD')
		plt.tight_layout()
		plt.savefig(f'plots/{start_date}_{end_date}_{twins_or_algo}/{key}.png')




def main():

	data_dict = get_all_data()

	# start_date = pd.to_datetime('2012-03-08')
	# end_date = pd.to_datetime('2012-03-10')
	start_date = pd.to_datetime('2009-07-20')
	end_date = pd.to_datetime('2017-12-31')

	plotting_intervls(data_dict, start_date, end_date, 'twins')



if __name__ == '__main__':
	main()