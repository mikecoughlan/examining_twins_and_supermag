import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklern.tree import DecisionTreeClassifier

import utils

REGIONS = []
FEATURES = []


def combining_stations_into_regions(stations, rsd, features, mean=False, std=False, maximum=False, median=False):

	start_time = pd.to_datetime('2009-07-20')
	end_time = pd.to_datetime('2017-12-31')
	twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

	regional_df = pd.DataFrame(index=twins_time_period)

	# creating a dataframe for each feature with the twins time period as the index and storing them in a dict
	feature_dfs = {}
	for feature in features:
		feature_dfs[feature] = pd.DataFrame(index=twins_time_period)

	for stat in stations:
		df = utils.loading_supermag(stat)
		df = df[start_time:end_time]
		for feature in features:
			feature_dfs[feature][f'{stat}_{feature}'] = df[feature]

	for feature in features:
		if mean:
			regional_df[f'{feature}_mean'] = feature_dfs[feature].mean(axis=1)
		if std:
			regional_df[f'{feature}_std'] = feature_dfs[feature].std(axis=1)
		if maximum:
			regional_df[f'{feature}_max'] = feature_dfs[feature].max(axis=1)
		if median:
			regional_df[f'{feature}_median'] = feature_dfs[feature].median(axis=1)

	indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)

	regional_df['rsd'] = rsd['max_rsd']['max_rsd']
	regional_df['rolling_rsd'] = rsd['max_rsd']['max_rsd'].rolling(indexer, min_periods=1).max()
	regional_df['MLT'] = rsd['max_rsd']['MLT']
	regional_df['cosMLT'] = np.cos(regional_df['MLT'] * 2 * np.pi * 15 / 360)
	regional_df['sinMLT'] = np.sin(regional_df['MLT'] * 2 * np.pi * 15 / 360)


	return regional_df


def loading_data(prediction_param):

	# loading all the datasets and dictonaries

	regions, stats = utils.loading_dicts()
	solarwind = utils.loading_solarwind(omni=True)

	# reduce the regions dict to be only the ones that have keys in the region_numbers list
	regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in REGIONS}

	# Getting regions data for each region
	for region in regions.keys():

		# getting dbdt and rsd data for the region
		temp_df = combining_stations_into_regions(regions[region]['station'], stats[region], features=FEATURES, mean=True, std=True, maximum=True, median=True)
		regions[region]['regional_df'] = temp_df

		# getting the mean latitude for the region and attaching it to the regions dictionary
		mean_lat = utils.getting_mean_lat(regions[region]['station'])
		regions[region]['mean_lat'] = mean_lat


	data_dict = {'solarwind':solarwind, 'regions':regions}

	return data_dict


def finding_correlations(df, target):

	correlations = df.corr()[target].sort_values(ascending=False)

	return correlations