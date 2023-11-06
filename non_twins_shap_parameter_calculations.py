##############################################################################################################
#
#	multi-station-dbdt-risk-assessment/calculating_shap_values.py
#
#   Calculates the SHAP values for the solar wind and combined models for each station. 10 randomly chosen
# 	of the 100 split models have their resulting 2D SHAP value arrays averaged. The sum of the values across
# 	the time dimension producing only one value per input parameter. The total values are then normalized to
# 	a percentage of total SHAP values for that input array, giving a percent contribution for each parameter.
# 	Plots the results in a stackplot for the combined and the solar wind models. Also calculates the rolling
# 	average of these percent contributions to smooth the plots. This was not used in analysis.
#
##############################################################################################################



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
import utils
import non_twins_modeling_v2 as modeling

TARGET = 'rsd'
REGIONS = [194, 270, 287, 207, 62, 241, 366, 387, 223, 19, 163]
VERSION = 2

MODEL_CONFIG = {'filters':128,
				'initial_learning_rate':1e-6,
				'epochs':500,
				'loss':'mse',
				'early_stop_patience':25}

def main(region):

	# re-processing the training and testing data using the same random seed to generate the same data used for creating the models
	# This is done to ensure the same data is used for calculating the SHAP values as was used for training the models

	xtrain, xval, xtest, ytrain, yval, ytest, dates_dict, features = modeling.getting_prepared_data(target_var=TARGET, region=region, get_features=True)

	# reshaping the data for a CNN with one channel
	xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
	xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))
	xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], xtest.shape[2], 1))

	# Loading the models and the prediction results
	model = load_model(f'models/{TARGET}/non_twins_region_{region}_v{VERSION}.h5', compile=False)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['initial_learning_rate']), loss=modeling.CRPS)
	predictions = pd.read_feather(f'outputs/{TARGET}/non_twins_modeling_region_{region}_version_{VERSION}.feather')

	if not os.path.exists(f'outputs/shap_values/{TARGET}'):
		os.mkdir(f'outputs/shap_values/{TARGET}')
	# Checking to see if the SHAP values for the model have been calculated already
	if os.path.exists(f'outputs/shap_values/{TARGET}/non_twins_shap_values_region_{region}.pkl'):
		with open(f'outputs/shap_values/{TARGET}/non_twins_shap_values_region_{region}.pkl', 'rb') as c:
			shap_values = pickle.load(c)

	else:
		# SHAP documentation list 1000 background samples as being sufficient for accurate SHAP values
		background = xtrain[np.random.choice(xtrain.shape[0], 1000, replace=False)]

		# initalizing the explainer for the combined model
		explainer = shap.DeepExplainer(model, background)

		# Calculating the SHAP values
		shap_values = explainer.shap_values(xtest, check_additivity=True)

		with open(f'outputs/shap_values/{TARGET}/non_twins_shap_values_region_{region}.pkl', 'wb') as c:
			pickle.dump(shap_values, c)

		# Freeing up memory
		gc.collect()

	# Adding the "crossing" arrays to a list
	mean_shap_values = shap_values[0]
	std_shap_vlues = shap_values[1]

	# Summing of the SHAP values across the time dimension
	summed_mean_shap_values = np.sum(mean_shap_values, axis=1)
	summed_std_shap_values = np.sum(std_shap_values, axis=1)

	# Reshaping the arrays
	summed_mean_shap_values = summed_mean_shap_values.reshape(summed_mean_shap_values.shape[0], summed_mean_shap_values.shape[1])
	summed_std_shap_values = summed_std_shap_values.reshape(summed_std_shap_values.shape[0], summed_std_shap_values.shape[1])

	# Turning the arrays into a dataframe for plotting
	mean_df = pd.DataFrame(summed_mean_shap_values, columns=features)
	std_df = pd.DataFrame(summed_std_shap_values, columns=features)

	# Changing the SHAP values into percentage contributions
	perc_mean_df = (mean_df.div(mean_df.abs().sum(axis=1), axis=0))*100
	perc_std_df = (std_df.div(std_df.abs().sum(axis=1), axis=0))*100

	# Calculates a rolling average
	perc_mean_rolling = perc_mean_df.rolling(10).mean()
	perc_std_rolling = perc_std_df.rolling(10).mean()

	# Seperating the positive contributions from the negative for plotting
	perc_mean_pos_df = perc_mean_df.mask(perc_mean_df < 0, other=0)
	perc_mean_neg_df = perc_mean_df.mask(perc_mean_df > 0, other=0)
	perc_std_pos_df = perc_std_df.mask(perc_std_df < 0, other=0)
	perc_std_neg_df = perc_std_df.mask(perc_std_df > 0, other=0)

	perc_mean_pos_dict, perc_mean_neg_dict, perc_std_pos_dict, perc_std_neg_dict = {}, {}, {}, {}

	# Creating numpy arrays for each parameter
	for pos, neg in zip(perc_mean_pos_df, perc_mean_neg_df):
		perc_mean_pos_dict[pos] = perc_mean_pos_df[pos].to_numpy()
		perc_mean_neg_dict[neg] = perc_mean_neg_df[neg].to_numpy()

	for pos, neg in zip(perc_std_pos_df, perc_std_neg_df):
		perc_std_pos_dict[pos] = perc_std_pos_df[pos].to_numpy()
		perc_std_neg_dict[neg] = perc_std_neg_df[neg].to_numpy()

	# Reordering the parameters in the combined model so the colors of
	# each parameter are the same in both plots

	perc_mean_rolling['Date_UTC'] = predictions['dates']
	perc_std_rolling['Date_UTC'] = predictions['dates']

	perc_mean_rolling.set_index('Date_UTC', inplace=True)
	perc_std_rolling.set_index('Date_UTC', inplace=True)

	perc_x = perc_mean_rolling.index

	# Defining the color pallet for the stack plots
	colors = sns.color_palette('tab20', len(features))

	if not os.path.exists(f'plots/shap/{TARGET}'):
		os.mkdir(f'plots/shap/{TARGET}')

	# Plotting
	fig = plt.figure(figsize=(20,17))

	ax1 = plt.subplot(111)
	ax1.set_title('SHAP Values for Mean Predictions')

	# Stacking the positive and negative percent contributions
	plt.stackplot(prec_x, perc_mean_pos_dict.values(), labels=features, colors=colors, alpha=1)
	plt.stackplot(prec_x, perc_mean_neg_dict.values(), colors=colors, alpha=1)
	ax1.margins(x=0, y=0)				# Tightning the plot margins
	plt.ylabel('Percent Contribution')

	# Placing the legend outside of the plot
	plt.legend(bbox_to_anchor=(1,1), loc='upper left')
	plt.axhline(0, color='black')

	plt.savefig(f'plots/shap/{TARGET}/non_twins_mean_prediction_percent_contribution_region_{region}.png')


	# Plotting
	fig = plt.figure(figsize=(20,17))

	ax1 = plt.subplot(111)
	ax1.set_title('SHAP Values for Std Predictions')

	# Stacking the positive and negative percent contributions
	plt.stackplot(prec_x, perc_std_pos_dict.values(), labels=features, colors=colors, alpha=1)
	plt.stackplot(prec_x, perc_std_neg_dict.values(), colors=colors, alpha=1)
	ax1.margins(x=0, y=0)				# Tightning the plot margins
	plt.ylabel('Percent Contribution')

	# Placing the legend outside of the plot
	plt.legend(bbox_to_anchor=(1,1), loc='upper left')
	plt.axhline(0, color='black')

	plt.savefig(f'plots/shap/{TARGET}/non_twins_std_prediction_percent_contribution_region_{region}.png')


	# Plotting the same two as above but in the same figure
	fig = plt.figure(figsize=(20,17))

	ax1 = plt.subplot(211)
	ax1.set_title('SHAP Values for Mean Predictions')

	# Stacking the positive and negative percent contributions
	plt.stackplot(prec_x, perc_mean_pos_dict.values(), labels=features, colors=colors, alpha=1)
	plt.stackplot(prec_x, perc_mean_neg_dict.values(), colors=colors, alpha=1)
	ax1.margins(x=0, y=0)				# Tightning the plot margins
	plt.ylabel('Percent Contribution')

	# Placing the legend outside of the plot
	plt.legend(bbox_to_anchor=(1,1), loc='upper left')
	plt.axhline(0, color='black')


	ax2 = plt.subplot(212, sharex=ax1)
	ax2.set_title('SHAP Values for Std Predictions')

	# Stacking the positive and negative percent contributions
	plt.stackplot(prec_x, perc_std_pos_dict.values(), labels=features, colors=colors, alpha=1)
	plt.stackplot(prec_x, perc_std_neg_dict.values(), colors=colors, alpha=1)
	ax2.margins(x=0, y=0)				# Tightning the plot margins
	plt.ylabel('Percent Contribution')

	# Placing the legend outside of the plot
	plt.legend(bbox_to_anchor=(1,1), loc='upper left')
	plt.axhline(0, color='black')

	plt.legend(bbox_to_anchor=(1,1), loc='upper left')
	plt.savefig(f'plots/shap/{TARGET}/non_twins_stacked_plots_region_{region}.png', bbox_inches='tight')


	# Same as above but with the rolling averages
	fig = plt.figure(figsize=(20,17))

	ax1 = plt.subplot(111)
	ax1.set_title('SHAP Values for Mean Predictions')
	# Highlihting area of interest
	for param, label, color in zip(perc_mean_rolling.columns, features, colors):
		plt.plot(perc_mean_rolling[param], label=label, color=color)
	ax1.margins(x=0, y=0)
	plt.ylabel('Percent Contribution')
	plt.legend(bbox_to_anchor=(1,1), loc='upper left')
	plt.axhline(0, color='black')

	plt.savefig(f'plots/shap/{TARGET}/rolling_mean_percent_contribution_region_{region}.png')


	fig = plt.figure(figsize=(20,17))

	ax2 = plt.subplot(111)
	ax2.set_title('SHAP Values for Std Predictions')
	# Highlihting area of interest
	for param, label, color in zip(perc_std_rolling.columns, features, colors):
		plt.plot(perc_std_rolling[param], label=label, color=color)
	ax2.margins(x=0, y=0)
	plt.ylabel('Percent Contribution')
	plt.legend(bbox_to_anchor=(1,1), loc='upper left')
	plt.axhline(0, color='black')

	plt.savefig(f'plots/shap/{TARGET}/rolling_std_percent_contribution_region_{region}.png')


if __name__ == '__main__':

	# Lists the stations
	for region in REGIONS:
		print(f'Starting {region}....')
		main(region)
		print(f'Finished {region}')

	print('It ran. Good job!')









