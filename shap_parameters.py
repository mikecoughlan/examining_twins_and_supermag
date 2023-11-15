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

import non_twins_modeling_final_version as modeling

MODEL_CONFIG = {'initial_filters': 128,
				'learning_rate': 4.1521558834373335e-07,
				'window_size': 3,
				'stride_length': 1,
				'cnn_layers': 4,
				'dense_layers': 3,
				'cnn_step_up': 2,
				'initial_dense_nodes': 1024,
				'dense_node_decrease_step': 2,
				'dropout_rate': 0.22035812839389704,
				'activation': 'relu',
				'early_stop_patience':25,
				'epochs':500}

TARGET = 'rsd'
VERSION = 'final'
# REGIONS = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
# 			387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
# 			62, 327, 293, 241, 107, 55, 111]
REGIONS = [83]


def loading_model(model_path):

	# Load model
	model = load_model(model_path, compile=False)
	model.compile(loss=modeling.CRPS, optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']))

	return model

def segmenting_testing_data(xtest, ytest, dates, storm_months=['2012-03-01', '2017-09-01']):

	evaluation_dict = {month:{} for month in storm_months}

	# turning storm months into datetime objects
	storm_months = [pd.to_datetime(month) for month in storm_months]

	# Creating daterange for storm months at 1 min frequency
	storm_date_ranges = [pd.date_range(start=month, end=month+pd.DateOffset(months=1), freq='min') for month in storm_months]

	# getting the indicies in the dates df that correspond to the storm months and using those indicies to extract
	# the corresponding arrays from xtest and ytest
	storm_indicies = []
	for key, date_range in zip(evaluation_dict.keys(), storm_date_ranges):
		temp_df = dates['Date_UTC'].isin(date_range)
		indicies = temp_df[temp_df == True].index.tolist()
		evaluation_dict[key]['Date_UTC'] = temp_df[indicies]
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

	if os.path.exists(f'outputs/shap_values/{model_name}_shap_values.pkl'):
		with open(f'outputs/shap_values/{model_name}_shap_values.pkl', 'rb') as f:
			shap_values = pickle.load(f)

	else:
		# checking to see if the xtrain is a list of multiple inputs. Creates background for each using same random sampling
		if isinstance(training_data, list):
			background = []
			random_indicies = np.random.choice(training_data[0].shape[0], background_examples, replace=False)
			for i in range(len(training_data)):
				background.append(training_data[i][random_indicies])

		else:
			# Get shap values
			background = training_data[np.random.choice(training_data.shape[0], background_examples, replace=False)]

		explainer = shap.DeepExplainer(model, background)

		if isinstance(evaluation_dict, dict):
			print('Calculating shap values for each storm month....')
			for key in tqdm(evaluation_dict.keys()):
				evaluation_dict[key]['shap_values'] = explainer.shap_values(evaluation_dict[key]['xtest'], check_additivity=False)

		else:
			shap_values = explainer.shap_values(evaluation_dict, check_additivity=False)

		with open(f'outputs/shap_values/{model_name}_shap_values.pkl', 'wb') as f:
			pickle.dump(shap_values, f)

	return shap_values


def converting_shap_to_percentages(shap_values, features):

	if shap_values.shape[0] > 1:
		all_shap_values = []
		for i in range(shap_values.shape[0]):
			summed_shap_values = np.sum(shap_values[i], axis=1)
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


def preparing_shap_values_for_plotting(df):

	# Seperating the positive contributions from the negative for plotting
	pos_df = df.mask(df < 0, other=0)
	neg_df = df.mask(df > 0, other=0)

	pos_dict, neg_dict = {}, {}

	# Creating numpy arrays for each parameter
	for pos, neg in zip(pos_df, neg_df):
		pos_dict[pos] = pos_df[pos].to_numpy()
		neg_dict[neg] = neg_df[neg].to_numpy()

	return pos_dict, neg_dict


def plotting_shap_values(evaluation_dict, features, region):

	for key in evaluation_dict.keys():

		shap_percentages = converting_shap_to_percentages(evaluation_dict[key]['shap_values'], features)
		mean_pos_dict, mean_neg_dict = preparing_shap_values_for_plotting(shap_percentages[0])
		std_pos_dict, std_neg_dict = preparing_shap_values_for_plotting(shap_percentages[1])

		# Creating the x-axis for the plot
		x = evaluation_dict[key]['Date_UTC']

		# Plotting
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('SHAP Values for Mean Predictions')

		# Stacking the positive and negative percent contributions
		plt.stackplot(x, mean_pos_dict.values(), labels=features, colors=colors, alpha=1)
		plt.stackplot(x, mean_neg_dict.values(), colors=colors, alpha=1)
		ax1.margins(x=0, y=0)				# Tightning the plot margins
		plt.ylabel('Percent Contribution')

		# Placing the legend outside of the plot
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/{TARGET}/{key}_non_twins_mean_region_{region}.png')


		# Plotting
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('SHAP Values for Std Predictions')

		# Stacking the positive and negative percent contributions
		plt.stackplot(x, std_pos_dict.values(), labels=features, colors=colors, alpha=1)
		plt.stackplot(x, std_neg_dict.values(), colors=colors, alpha=1)
		ax1.margins(x=0, y=0)				# Tightning the plot margins
		plt.ylabel('Percent Contribution')

		# Placing the legend outside of the plot
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/{TARGET}/{key}_non_twins_std_region_{region}.png')


def getting_feature_importance(shap_values, features):

	shap_percentages = converting_shap_to_percentages(shap_values, features)

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

	return feature_importance_df


def main():

	feature_importance_dict = {region:{} for region in REGIONS}

	regs, __ = utils.loading_dicts()
	regs = {key:regs[f'region_{key}'] for key in REGIONS}
	for region in REGIONS:
		feature_importance_dict[region]['mean_lat'] = utils.get_mean_lat(regs[region]['station'])

	del regs
	gc.collect()

	for region in REGIONS:
		xtrain, ___, xtest, ytrain, ____, ytest, dates_dict, features = modeling.getting_prepared_data(target_var=TARGET, region=region, get_features=True)
		evaluation_dict = segmenting_testing_data(xtest, ytest, dates_dict['test'], storm_months=['2012-03-01', '2017-09-01'])
		model = load_model(f'models/{TARGET}/non_twins_region_{region}_version_{VERSION}.h5')
		shap_values = get_shap_values(model, f'non_twins_region_{region}_version_{VERSION}', xtrain, evaluation_data)
		plotting_shap_values(evaluation_dict, features, region)
		feature_importance_dict[region]['feature_importance'] = getting_feature_importance(shap_values, features)

	with open(f'outputs/shap_values/non_twins_feature_importance_dict.pkl', 'wb') as f:
		pickle.dump(feature_importance_dict, f)

	# plotting feature importance for each feature as a function of mean latitude
	for feature in features:
		mean_mean, mean_std, std_mean, std_std, lat = [], [], [], [], []
		for region in REGIONS:
			lat.append(feature_importance_dict[region]['mean_lat'])
			mean_mean.append(feature_importance_dict[region]['feature_importance']['mean_mean_shap'][feature])
			mean_std.append(feature_importance_dict[region]['feature_importance']['mean_std_shap'][feature])
			std_mean.append(feature_importance_dict[region]['feature_importance']['std_mean_shap'][feature])
			std_std.append(feature_importance_dict[region]['feature_importance']['std_std_shap'][feature])

		fig = plt.figure(figsize=(20,17))
		ax1 = plt.subplot(211)
		ax1.set_title(f'Mean SHAP Percentage Importance for {feature}')
		plt.scatter(lat, mean_mean, label='$\mu$')
		plt.scatter(lat, mean_std, label='$\sigma$')
		plt.ylabel('Mean SHAP Percentage Importance')
		plt.xlabel('Region Latitude')
		plt.legend()

		ax2 = plt.subplot(212)
		ax2.set_title(f'Std SHAP Percentage Importance for {feature}')
		plt.scatter(lat, std_mean, label='$\mu$')
		plt.scatter(lat, std_std, label='$\sigma$')
		plt.ylabel('Std SHAP Percentage Importance')
		plt.xlabel('Region Latitude')
		plt.legend()

		plt.savefig(f'plots/shap/{TARGET}/non_twins_feature_importance_{feature}.png')



if __name__ == '__main__':
	main()







