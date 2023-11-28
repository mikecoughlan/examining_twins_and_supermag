import datetime as dt
import gc
import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
from matplotlib import colors
from tensorflow.keras.models import Sequential, load_model
from tqdm import tqdm
import gc

import twins_modeling_v0 as modeling
import utils

CONFIG = {'region_numbers': [44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
								61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
								62, 327, 293, 241, 107, 55, 111, 83, 143, 223, 387],
			'load_twins':False,
			'mag_features':[],
			'solarwind_features':[],
			'delay':False,
			'rolling':False,
			'to_drop':[],
			'omni_or_ace':'omni',
			'time_history':30,
			'random_seed':7,
			'initial_filters':128,
			'learning_rate':1e-7,
			'epochs':500,
			'loss':'mse',
			'early_stop_patience':25}

TARGET = 'rsd'
VERSION = 'final'
REGIONS = [44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
			61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
			62, 327, 293, 241, 107, 55, 111, 83, 143, 223, 387]
# REGIONS = [83]


def loading_model(model_path):

	# Load model
	model = load_model(model_path, compile=False)
	model.compile(loss=modeling.CRPS, optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']))

	return model

def segmenting_testing_data(xtest, ytest, dates, storm_months=['2017-09-01', '2012-03-07'], storm_duration=[pd.DateOffset(months=1), pd.DateOffset(days=7)]):

	evaluation_dict = {month:{} for month in storm_months}

	if len(storm_months) != len(storm_duration):
		raise ValueError('The storm months and storm duration must be the same length.')


	# turning storm months into datetime objects
	storm_months = [pd.to_datetime(month) for month in storm_months]

	# Creating daterange for storm months at 1 min frequency
	storm_date_ranges = [pd.date_range(start=month, end=month+duration, freq='min') for month, duration in zip(storm_months, storm_duration)]

	# getting the indicies in the dates df that correspond to the storm months and using those indicies to extract
	# the corresponding arrays from xtest and ytest
	storm_indicies = []
	dates['Date_UTC'] = pd.to_datetime(dates['Date_UTC'])
	for key, date_range in zip(evaluation_dict.keys(), storm_date_ranges):

		temp_df = dates['Date_UTC'].isin(date_range)
		if temp_df.sum() == 0:
			raise ValueError(f'There are no dates in the dates df that match the storm month {key}.')
		indicies = temp_df[temp_df == True].index.tolist()
		evaluation_dict[key]['Date_UTC'] = dates['Date_UTC'][indicies].reset_index(drop=True, inplace=False)
		evaluation_dict[key]['xtest'] = xtest[indicies]
		evaluation_dict[key]['ytest'] = ytest[indicies]

	return evaluation_dict


def get_shap_values(model, model_name, training_data, evaluation_dict, background_examples=1000):
	'''
	Function that calculates the shap values for the given model and evaluation data. First checks for previously calculated shap
	values and loads them if they exist. If not, it calculates them and saves them to a pickle file.

	Args:
		model (keras object): trainined neural network model to calculate shap values for.
		background_examples (int, optional): number of background samples to use in calculating shap values. Defaults to 1000.

	Returns:
		np.array or list of np.arrays: shap values for each input feature. Will return a list of arrays if the model has multiple
										inputs. Shape will be the same as the shape of the evaluation data with an additional dimension
										for each of the model outputs.
	'''

	if os.path.exists(f'outputs/shap_values/{model_name}_evaluation_dict.pkl'):
		with open(f'outputs/shap_values/{model_name}_evaluation_dict.pkl', 'rb') as f:
			evaluation_dict = pickle.load(f)

	else:
		# checking to see if the xtrain is a list of multiple inputs. Creates background for each using same random sampling
		if isinstance(training_data, list):
			background = []
			random_indicies = np.random.choice(training_data[0].shape[0], background_examples, replace=False)
			for data in training_data:
				background.append(data[random_indicies])

		else:
			# Get shap values
			background = training_data[np.random.choice(training_data.shape[0], background_examples, replace=False)]

		explainer = shap.DeepExplainer(model, background)

		print('Calculating shap values for each storm month....')
		for key in tqdm(evaluation_dict.keys()):
			shap_values = explainer.shap_values(evaluation_dict[key]['xtest'], check_additivity=False)
			evaluation_dict[key]['shap_values'] = shap_values

		with open(f'outputs/shap_values/{model_name}_evaluation_dict.pkl', 'wb') as f:
			pickle.dump(evaluation_dict, f)

	return evaluation_dict


def converting_shap_to_percentages(shap_values, features):

	if len(shap_values) > 1:
		all_shap_values = []
		for shap in shap_values:
			summed_shap_values = np.sum(shap, axis=1)
			summed_shap_values = summed_shap_values.reshape(summed_shap_values.shape[0], summed_shap_values.shape[1])
			shap_df = pd.DataFrame(summed_shap_values, columns=features)
			perc_df = (shap_df.div(shap_df.abs().sum(axis=1), axis=0))*100
			all_shap_values.append(perc_df)

	else:
		summed_shap_values = np.sum(shap_values, axis=1)
		summed_shap_values = summed_shap_values.reshape(summed_shap_values.shape[0], summed_shap_values.shape[1])
		shap_df = pd.DataFrame(summed_shap_values, columns=features)
		perc_df = (shap_df.div(shap_df.abs().sum(axis=1), axis=0))*100
		all_shap_values = perc_df

	return all_shap_values


def preparing_shap_values_for_plotting(df, dates):

	df = handling_gaps(df, 1000, dates)

	# Seperating the positive contributions from the negative for plotting
	pos_df = df.mask(df < 0, other=0)
	neg_df = df.mask(df > 0, other=0)

	pos_dict, neg_dict = {}, {}

	# Creating numpy arrays for each parameter
	for pos, neg in zip(pos_df, neg_df):
		pos_dict[pos] = pos_df[pos].to_numpy()
		neg_dict[neg] = neg_df[neg].to_numpy()

	return pos_dict, neg_dict, df.index


def handling_gaps(df, threshold, dates):
	'''
	Function for keeping blocks of nans in the data if there is a maximum number of data points between sucessive valid data.
	If the number of nans is too large between sucessive data points it will drop those nans.

	Args:
		df (pd.DataFrame): data to be processed

	Returns:
		pd.DataFrame: processed data
	'''
	df['Date_UTC'] = dates
	df.set_index('Date_UTC', inplace=True)
	df.index = pd.to_datetime(df.index)

	start_time = pd.to_datetime('2009-07-19')
	end_time = pd.to_datetime('2017-12-31')
	date_range = pd.date_range(start_time, end_time, freq='min')

	full_time_df = pd.DataFrame(index=date_range)

	df = full_time_df.join(df, how='left')

	# creting a column in the data frame that labels the size of the gaps
	df['gap_size'] = df[df.columns[1]].isna().groupby(df[df.columns[1]].notna().cumsum()).transform('sum')

	# setting teh gap size column to nan if the value is above the threshold, setting it to 0 otherwise
	df['gap_size'] = np.where(df['gap_size'] > threshold, np.nan, 0)

	# dropping nans from the subset of the gap size column
	df.dropna(inplace=True, subset=['gap_size'])

	# dropping the gap size column
	df.drop(columns=['gap_size'], inplace=True)

	return df


def plotting_shap_values(evaluation_dict, features, region):

	for key in evaluation_dict.keys():

		shap_percentages = converting_shap_to_percentages(evaluation_dict[key]['shap_values'], features)
		mean_pos_dict, mean_neg_dict, mean_dates = preparing_shap_values_for_plotting(shap_percentages[0], evaluation_dict[key]['Date_UTC'])
		std_pos_dict, std_neg_dict, std_dates = preparing_shap_values_for_plotting(shap_percentages[1], evaluation_dict[key]['Date_UTC'])

		colors = sns.color_palette('tab20', len(mean_pos_dict.keys()))

		# Creating the x-axis for the plot
		x = evaluation_dict[key]['Date_UTC'].values

		# Plotting
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('SHAP Values for Mean Predictions')
		pos_values = [val for val in mean_pos_dict.values()]
		neg_values = [val for val in mean_neg_dict.values()]

		# Stacking the positive and negative percent contributions
		plt.stackplot(mean_dates, pos_values, labels=features, colors=colors, alpha=1)
		plt.stackplot(mean_dates, neg_values, colors=colors, alpha=1)
		ax1.margins(x=0, y=0)				# Tightning the plot margins
		plt.ylabel('Percent Contribution')

		# Placing the legend outside of the plot
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/{TARGET}/{key}_twins_mean_region_{region}.png')


		# Plotting
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('SHAP Values for Std Predictions')

		pos_values = [val for val in std_pos_dict.values()]
		neg_values = [val for val in std_neg_dict.values()]

		# Stacking the positive and negative percent contributions
		plt.stackplot(std_dates, pos_values, labels=features, colors=colors, alpha=1)
		plt.stackplot(std_dates, neg_values, colors=colors, alpha=1)
		ax1.margins(x=0, y=0)				# Tightning the plot margins
		plt.ylabel('Percent Contribution')

		# Placing the legend outside of the plot
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/{TARGET}/{key}_twins_std_region_{region}.png')


def getting_feature_importance(evaluation_dict, features):

	feature_importances = []

	for key in evaluation_dict.keys():
		shap_percentages = converting_shap_to_percentages(evaluation_dict[key]['shap_values'], features)

		# seperating mean and std vlaues
		mean_shap_values = shap_percentages[0]
		std_shap_values = shap_percentages[1]

		# getting the mean and std of the mean and std shap values
		mean_mean_shap = mean_shap_values.abs().mean(axis=0)
		mean_std_shap = std_shap_values.abs().mean(axis=0)

		std_mean_shap = mean_shap_values.abs().std(axis=0)
		std_std_shap = std_shap_values.abs().std(axis=0)

		feature_importance_df = pd.DataFrame({'mean_mean_shap':mean_mean_shap, 'mean_std_shap':mean_std_shap,
											'std_mean_shap':std_mean_shap, 'std_std_shap':std_std_shap}, index=features)

		feature_importances.append(feature_importance_df)

	return feature_importance_df


def main():

	feature_importance_dict = {region:{} for region in REGIONS}

	regs, __ = utils.loading_dicts()
	regs = {key:regs[f'region_{key}'] for key in REGIONS}
	for region in REGIONS:
		feature_importance_dict[region]['mean_lat'] = utils.getting_mean_lat(regs[region]['station'])

	del regs
	gc.collect()

	for region in REGIONS:

		if os.path.exists(f'outputs/shap_values/twins_region_{region}_evaluation_dict.pkl'):
			print(f'Shap values for region {region} already exist. Skipping....')
			continue

		print(f'Preparing data....')
		xtrain, ___, xtest, ytrain, ____, ytest, dates_dict, features = modeling.getting_prepared_data(target_var=TARGET, region=region, get_features=True)

		# reshaping the data to match the CNN input
		xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1)
		xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2], 1)

		print('Segmenting the evaluation data....')
		evaluation_dict = segmenting_testing_data(xtest, ytest, dates_dict['test'], storm_months=['2017-09-01', '2012-03-07'])

		print('Loading model....')
		MODEL = loading_model(f'models/{TARGET}/twins_region_{region}_v{VERSION}.h5')

		print('Getting shap values....')
		evaluation_dict = get_shap_values(model=MODEL, model_name=f'twins_region_{region}', training_data=xtrain, evaluation_dict=evaluation_dict)

		print('Plotting shap values....')
		plotting_shap_values(evaluation_dict, features, region)

		print('Getting feature importance....')
		feature_importance_dict[region]['feature_importance'] = getting_feature_importance(evaluation_dict, features)

		gc.collect()

	with open(f'outputs/shap_values/twins_feature_importance_dict.pkl', 'wb') as f:
		pickle.dump(feature_importance_dict, f)

	# keys = [key for key in evaluation_dict.keys()]
	# # plotting feature importance for each feature as a function of mean latitude
	# for feature in features:
	# 	mean_mean_0, mean_std_0, std_mean_0, std_std_0, lat = [], [], [], [], []
	# 	mean_mean_1, mean_std_1, std_mean_1, std_std_1 = [], [], [], []
	# 	for region in REGIONS:
	# 		lat.append(feature_importance_dict[region]['mean_lat'])
	# 		mean_mean_0.append(feature_importance_dict[region]['feature_importance'][0]['mean_mean_shap'][feature])
	# 		mean_std_0.append(feature_importance_dict[region]['feature_importance'][0]['mean_std_shap'][feature])
	# 		std_mean_0.append(feature_importance_dict[region]['feature_importance'][0]['std_mean_shap'][feature])
	# 		std_std_0.append(feature_importance_dict[region]['feature_importance'][0]['std_std_shap'][feature])
	# 		mean_mean_1.append(feature_importance_dict[region]['feature_importance'][1]['mean_mean_shap'][feature])
	# 		mean_std_1.append(feature_importance_dict[region]['feature_importance'][1]['mean_std_shap'][feature])
	# 		std_mean_1.append(feature_importance_dict[region]['feature_importance'][1]['std_mean_shap'][feature])
	# 		std_std_1.append(feature_importance_dict[region]['feature_importance'][1]['std_std_shap'][feature])

	# 	# defining two colors close to each other for each of the storms
	# 	colors = ['#ff0000', '#ff4d4d', '#ff8080', '#ffcccc', '#0000ff', '#4d4dff', '#8080ff', '#ccccff']

	# 	fig = plt.figure(figsize=(20,17))
	# 	ax1 = plt.subplot(211)
	# 	ax1.set_title(f'Mean SHAP Percentage Importance for {feature}')
	# 	plt.scatter(lat, mean_mean_0, label=f'$\mu$ {keys[0]}', color=colors[0])
	# 	plt.scatter(lat, mean_std_0, label=f'$\sigma$ {keys[0]}', color=colors[1])
	# 	plt.scatter(lat, mean_mean_1, label=f'$\mu$ {keys[1]}', color=colors[4])
	# 	plt.scatter(lat, mean_std_1, label=f'$\sigma$ {keys[1]}', color=colors[5])
	# 	plt.ylabel('Mean SHAP Percentage Importance')
	# 	plt.xlabel('Region Latitude')
	# 	plt.legend()

	# 	ax2 = plt.subplot(212)
	# 	ax2.set_title(f'Std SHAP Percentage Importance for {feature}')
	# 	plt.scatter(lat, std_mean_0, label=f'$\mu$ {keys[0]}', color=colors[0])
	# 	plt.scatter(lat, std_std_0, label=f'$\sigma$ {keys[0]}', color=colors[1])
	# 	plt.scatter(lat, std_mean_1, label=f'$\mu$ {keys[1]}', color=colors[4])
	# 	plt.scatter(lat, std_std_1, label=f'$\sigma$ {keys[1]}', color=colors[5])
	# 	plt.ylabel('Std SHAP Percentage Importance')
	# 	plt.xlabel('Region Latitude')
	# 	plt.legend()

	# 	plt.savefig(f'plots/shap/{TARGET}/twins_feature_importance_{feature}.png')



if __name__ == '__main__':
	main()







