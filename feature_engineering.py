import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils

# list of regions. Taking two from each cluster and two low lat non cluster regions
REGIONS = [194, 270, 287, 207, 62, 241, 366, 387, 223, 19, 163]

# supermag features to use
FEATURES = ['N', 'E', 'theta', 'MAGNITUDE', 'dbht']

# version number
VERSION = 0


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


def loading_data():

	# loading all the datasets and dictonaries

	regions, stats = utils.loading_dicts()
	solarwind = utils.loading_solarwind(omni=True)

	# converting the solarwind data to log10
	solarwind['logT'] = np.log10(solarwind['T'])
	solarwind.drop(columns=['T'], inplace=True)

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


def merging_solarwind_and_supermag(data_dict):

	# merging the solarwind and supermag dataframes
	for region in data_dict['regions'].keys():
		data_dict['regions'][region]['merged_df'] = pd.merge(data_dict['regions'][region]['regional_df'], \
																data_dict['solarwind'], left_index=True, \
																right_index=True, how='inner')

	return data_dict


def finding_correlations(df, target, region):
	'''
	Calculates and plots the correlations between the features and the target

	Args:
		df (pandas dataframe): df of all the input features and the target variable
		target (string): string of the column name of the target variable
		region (string): string of the region name
	'''

	# calculating the correlations
	correlations = df.corr()
	target_corrs = correlations[target].sort_values(ascending=False)

	# plotting the target correlations
	plt.figure(figsize=(10,10))
	plt.imshow(target_corrs.values.reshape(-1,1), cmap='hot', interpolation='nearest')
	plt.colorbar()
	plt.yticks(np.arange(len(target_corrs.index)), target_corrs.index)
	plt.xticks([])
	plt.title('Correlations between features and target')
	plt.savefig(f'plots/feature_engineering/{region}_target {target}_correlations_v{VERSION}.png')

	# plotting the correlations between the features
	plt.figure(figsize=(10,10))
	plt.imshow(correlations.values, cmap='bwr', interpolation='nearest')
	plt.colorbar()
	plt.yticks(np.arange(len(correlations.index)), correlations.index)
	plt.xticks(np.arange(len(correlations.columns)), correlations.columns, rotation=90)
	plt.title('Correlations between features')
	plt.savefig(f'plots/feature_engineering/{region}_features_correlations_v{VERSION}.png')

	return correlations


def main():

	data_dict = loading_data()
	data_dict = merging_solarwind_and_supermag(data_dict)
	correlation_dict = {}
	for region in data_dict['regions'].keys():
		corr = finding_correlations(data_dict['regions'][region]['merged_df'], target='rolling_rsd', region=region)
		correlation_dict[region] = corr
	
	print(correlation_dict['region_163']['rolling_rsd'].sort_values(ascending=False))

	with open(f'outputs/feature_engineering/correlation_dict_v{VERSION}.pkl', 'wb') as f:
		pickle.dump(correlation_dict, f)


if __name__ == '__main__':
	main()
