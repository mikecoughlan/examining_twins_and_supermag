import gc
import os
import pickle

import scipy
from scipy.io import netcdf_file

os.environ["CDF_LIB"] = "~/CDF/lib"
import datetime
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacepy
from spacepy import pycdf
from tqdm import tqdm

from calculating_mlt_statistics import Calculating_MLT_Statistics

twins_dir = '../data/twins/'
supermag_dir = '../data/supermag/'
regions_dict = '../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'
stats_dict = '../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'


twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]

# Define the degree grid
mlt_min = 0
mlt_max = 24
mlt_step = (1/6)


def loading_twins_times():

	times = pd.read_feather('outputs/regular_twins_map_dates.feather')

	return times


def loading_regions_dicts():

	with open(regions_dict, 'rb') as f:
		regions = pickle.load(f)

	with open(stats_dict, 'rb') as s:
		stats = pickle.load(s)

	segmented_regions = {f'region_{reg}':regions[f'region_{reg}'] for reg in region_numbers}
	segmented_stats = {f'region_{reg}':stats[f'region_{reg}'] for reg in region_numbers}

	return segmented_regions, segmented_stats


def loading_supermag_data(station):

	df = pd.read_feather(supermag_dir+station+'.feather')
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')

	return df


def segmenting_dataframe_into_twins_map_times(df, twins_times):

	twins_times['dates'] = pd.to_datetime(twins_times['dates'], format='%Y-%m-%d %H:%M:$S')

	segmented_df = pd.DataFrame()

	for i in range(len(twins_times)):
		start_time = twins_times['dates'].iloc[i]
		end_time = twins_times['dates'].iloc[i] + pd.Timedelta(minutes=10)

		temp = df[start_time:end_time]

		segmented_df = pd.concat([segmented_df, temp], axis=0, ignore_index=False)

	return segmented_df


def main():

	twins_times = loading_twins_times()
	regions, stats = loading_regions_dicts()

	segmented = {}

	for region in tqdm(regions.keys()):
		stations = regions[region]['station']
		segmented[region] = {}

		for stat in stations:
			station = loading_supermag_data(stat)
			station = segmenting_dataframe_into_twins_map_times(station, twins_times)
			segmented[region][stat] = {}
			segmented[region][stat]['dataframe'] = station

			calculating = Calculating_MLT_Statistics(df=station, mlt_min=mlt_min, mlt_max=mlt_max, mlt_step=mlt_step, param='dbht')
			statistics = calculating.process_directory()

			segmented[region][stat]['statistics'] = statistics

		segmented_rsd = segmenting_dataframe_into_twins_map_times(stats[region]['max_rsd'], twins_times)

		segmented[region]['rsd'] = segmented_rsd
		calculating_rsd = Calculating_MLT_Statistics(df=segmented_rsd, mlt_min=mlt_min, mlt_max=mlt_max, mlt_step=mlt_step, param='max_rsd')
		rsd_stats = calculating_rsd.process_directory()

		segmented[region]['rsd_stats'] = rsd_stats

	with open('outputs/twins_time_segmented_data_and_stats.pkl', 'wb') as f:
		pickle.dump(segmented, f)





if __name__ == '__main__':
	main()

