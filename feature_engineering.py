import gc
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_engine.selection import (RecursiveFeatureElimination,
                                      SmartCorrelatedSelection)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

import utils

# list of regions. Taking two from each cluster and two low lat non cluster regions
REGIONS = [194, 270, 287, 207, 62, 241, 366, 387, 223, 19, 163]
# REGIONS = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
# 						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
# 						62, 327, 293, 241, 107, 55, 111]

# supermag features to use
FEATURES = ['N', 'E', 'theta', 'MAGNITUDE', 'dbht']

# version number
VERSION = 1


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
			if feature == 'N' or feature == 'E':
				regional_df[f'{feature}_mean'] = feature_dfs[feature].abs().mean(axis=1)
			else:
				regional_df[f'{feature}_mean'] = feature_dfs[feature].mean(axis=1)
		if std:
			regional_df[f'{feature}_std'] = feature_dfs[feature].std(axis=1)
		if maximum:
			if feature == 'N' or feature == 'E':
				regional_df[f'{feature}_max'] = feature_dfs[feature].abs().max(axis=1)
			else:
				regional_df[f'{feature}_max'] = feature_dfs[feature].max(axis=1)
		if median:
			if feature == 'N' or feature == 'E':
				regional_df[f'{feature}_median'] = feature_dfs[feature].abs().median(axis=1)
			else:
				regional_df[f'{feature}_median'] = feature_dfs[feature].median(axis=1)

	indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=15)

	regional_df['rsd'] = rsd['max_rsd']['max_rsd']
	regional_df['rolling_rsd'] = rsd['max_rsd']['max_rsd'].rolling(indexer, min_periods=1).max()
	regional_df['MLT'] = rsd['max_rsd']['MLT']
	regional_df['cosMLT'] = np.cos(regional_df['MLT'] * 2 * np.pi * 15 / 360)
	regional_df['sinMLT'] = np.sin(regional_df['MLT'] * 2 * np.pi * 15 / 360)


	return regional_df


def loading_data():

	# loading all the datasets and dictonaries

	regions, stats = utils.loading_dicts()
	solarwind = utils.loading_solarwind(omni=True, limit_to_twins=True)

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
	for region in tqdm(data_dict['regions'].keys()):

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

	# normalizing the variables between 0 and 1 before calculating correlation
	print(f'Normalizing the variables in {region}')
	for col in df.columns:
		df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
	gc.collect()

	# calculating the correlations
	correlations = df.corr()
	del df
	gc.collect()
	target_corrs = correlations[target].sort_values(ascending=False)

	# plotting the target correlations
	plt.figure(figsize=(10,10))
	plt.imshow(target_corrs.values.reshape(-1,1), cmap='bwr', vmin=-1, vmax=1)
	plt.colorbar()
	plt.yticks(np.arange(len(target_corrs.index)), target_corrs.index)
	plt.xticks([])
	plt.title('Correlations between features and target')
	plt.savefig(f'plots/feature_engineering/{region}_target {target}_correlations_v{VERSION}.png')

	# plotting the correlations between the features
	plt.figure(figsize=(10,10))
	plt.imshow(correlations.values, cmap='bwr', vmin=-1, vmax=1)
	plt.colorbar()
	plt.yticks(np.arange(len(correlations.index)), correlations.index)
	plt.xticks(np.arange(len(correlations.columns)), correlations.columns, rotation=90)
	plt.title('Correlations between features')
	plt.savefig(f'plots/feature_engineering/{region}_features_correlations_v{VERSION}.png')

	return correlations


def plotting_correlations_as_funtion_of_latitude(correlations, regions, variables, file_tag):


	corr_dict, lats = {var:[] for var in variables}, []
	for region in regions:
		lats.append(regions[region]['mean_lat'])
		for var in variables:
			corr_dict[var].append(correlations[region]['rolling_rsd'][var])

	plt.figure(figsize=(10,10))
	for var in variables:
		plt.scatter(lats, corr_dict[var], label=var)
	plt.legend()
	plt.xlabel('Mean Latitude')
	plt.ylabel('Correlation')
	plt.title(f'Correlation between {file_tag} and rolling_rsd as a function of latitude')
	plt.savefig(f'plots/feature_engineering/{file_tag}_correlation_vs_latitude_v{VERSION}.png')


def plotting_correlation_sum_as_funtion_of_latitude(correlations, regions, target, sw_params, mag_params, file_tag):

	sw_corr_sum, mag_corr_sum, lats = [], [], []

	for region in regions:
		lats.append(regions[region]['mean_lat'])
		corrs_abs = correlations[region][target].abs()
		sw_corrs = corrs_abs[sw_params]
		mag_corrs = corrs_abs[mag_params]

		if target in sw_params:
			sw_corrs.drop(target, inplace=True)
		if target in mag_params:
			mag_corrs.drop(target, inplace=True)

		sw_corr_sum.append(sw_corrs.sum())
		mag_corr_sum.append(mag_corrs.sum())
		gc.collect()

	plt.figure(figsize=(10,10))
	ax0 = plt.subplot(111)
	ax0.scatter(lats, sw_corr_sum, label='SW params', color='blue')
	ax1 = ax0.twinx()
	ax1.scatter(lats, mag_corr_sum, label='MAG params', color='orange')
	plt.legend()
	plt.xlabel('Mean Latitude')
	plt.ylabel('Sum of Correlations')
	plt.title(f'Sum of Correlations between features and {target} as a function of latitude')
	plt.savefig(f'plots/feature_engineering/{file_tag}_sum_correlation_vs_latitude_v{VERSION}.png')


def feature_elimination(df, target, region, threshold=0.1):

	print('Eliminating correlated features...')

	df.dropna(inplace=True)

	perc = df['rsd'].quantile(0.99)
	df = utils.classification_column(df=df, param=target, thresh=perc, forecast=0, window=0)
	target = df['classification'].to_numpy()

	df.drop(columns=['rsd', 'rolling_rsd', 'MLT', 'classification'], inplace=True)

	print(f'Nans before corr: {df.isna().sum()}')

	smart_correlation = SmartCorrelatedSelection(threshold=0.7, method='pearson', selection_method='variance')
	smart_correlation.fit(df)
	corr_feature_names = smart_correlation.features_to_drop_
	df = smart_correlation.transform(df)

	print(f'Nans after corr: {df.isna().sum()}')
	print(f'Length before dropping rows: {len(df)}')

	print(f'Features eliminated by correlated features for region {region}: {corr_feature_names}')

	print('Eliminating features using recursive feature elimination...')
	recursor = RecursiveFeatureElimination(estimator=RandomForestClassifier(random_state=42), scoring='roc_auc', cv=3, threshold=0.15)

	recursor.fit(df, target)
	recursor_feature_names = recursor.features_to_drop_
	df = recursor.transform(df)

	print(f'Features eliminated by recursive feature elimination for region {region}: {recursor_feature_names}')

	return df



def main():

	if os.path.exists(f'../../../../data/mike_working_dir/feature_engineering/data_dict.pkl'):
		with open(f'../../../../data/mike_working_dir/feature_engineering/data_dict.pkl', 'rb') as f:
			data_dict = pickle.load(f)

	else:
		data_dict = loading_data()
		data_dict = merging_solarwind_and_supermag(data_dict)
		with open(f'../../../../data/mike_working_dir/feature_engineering/data_dict.pkl', 'wb') as f:
			pickle.dump(data_dict, f)



	# sw_parameters = [param for param in data_dict['solarwind'].columns if param != 'Date_UTC']
	# mag_parameters = [param for param in data_dict['regions']['region_163']['merged_df'].columns if param not in sw_parameters or param in ['Date_UTC', 'rsd']]

	# gc.collect()

	# if os.path.exists(f'outputs/feature_engineering/correlation_dict_v{VERSION}.pkl'):
	# 	with open(f'outputs/feature_engineering/correlation_dict_v{VERSION}.pkl', 'rb') as f:
	# 		correlation_dict = pickle.load(f)

	# else:
	# 	correlation_dict = {}
	# 	for region in data_dict['regions'].keys():
	# 		corr = finding_correlations(data_dict['regions'][region]['merged_df'], target='rolling_rsd', region=region)
	# 		correlation_dict[region] = corr
	# 		gc.collect()

	# plotting_correlation_sum_as_funtion_of_latitude(correlation_dict, data_dict['regions'], target='rolling_rsd',\
	# 												sw_params=sw_parameters, mag_params=mag_parameters,\
	# 												file_tag='SW_and_mag')

	# plotting_correlations_as_funtion_of_latitude(correlation_dict, data_dict['regions'], variables=['E_std', 'N_std', 'MAGNITUDE_std'], file_tag='mag_std')
	# gc.collect()
	# plotting_correlations_as_funtion_of_latitude(correlation_dict, data_dict['regions'], variables=['E_mean', 'N_mean', 'MAGNITUDE_mean'], file_tag='mag_means')

	# with open(f'outputs/feature_engineering/correlation_dict_v{VERSION}.pkl', 'wb') as f:
	# 	pickle.dump(correlation_dict, f)

	# gc.collect()

	selected_regions = [194, 270, 287, 207, 62, 241, 366, 387, 223, 19, 163]

	for region in selected_regions:
		print(f'Initial length of the dataframe: {len(data_dict["regions"][f"region_{region}"]["merged_df"])}')
		df = utils.storm_extract(data_dict['regions'][f'region_{region}']['merged_df'])
		print(f'Length after storm extraction: {len(df)}')
		df = feature_elimination(df, target='rolling_rsd', region=region)
		df.reset_index(inplace=True, drop=False)
		df.to_feather(f'outputs/feature_engineering/eliminated_region_{region}_v{VERSION}.feather')

		gc.collect()


if __name__ == '__main__':
	main()
