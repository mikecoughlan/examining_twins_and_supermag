import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, brier_score_loss, confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_curve)

import utils

REGIONS = [194, 270, 287, 207, 62, 241, 366, 387, 223, 19, 163]

data_dir = '../../../../data/'
supermag_dir = data_dir+'supermag/feather_files/'
regions_dict = data_dir+'mike_working_dir/identifying_regions_data/adjusted_regions.pkl'

VERSION = 0
TARGET = 'rsd'

def load_predictions(region):

	if os.path.exists(f'outputs/{TARGET}/non_twins_modeling_region_{region}_version_{VERSION}.feather'):
		predictions = pd.read_feather(f'outputs/{TARGET}/non_twins_modeling_region_{region}_version_{VERSION}.feather')
		predictions.set_index('dates', inplace=True)
		predictions.index = pd.to_datetime(predictions.index, format = '%Y-%m-%d %H:%M:%S')

	else:
		raise(f'Fool of a Took! You need to run the script modeling_v{VERSION}.py first. Throw yourself down next time and rid us of your stupidity!')

	return predictions


def attaching_mlt(predictions, region):

	''' Function that attaches the MLT information to the predictions.'''

	# loading the segmented MLT data
	with open(regions_dict, 'rb') as f:
		regions = pickle.load(f)

	stations = regions[f'region_{region}']['station']

	latitude = utils.getting_mean_lat(stations)

	longitudes = {}
	latitudes = []
	for station in stations:
		temp_df = pd.read_feather(f'{supermag_dir}{station}.feather')
		longitudes[station] = temp_df['GEOLON'].iloc[0]

	# getting the key of the maximum value of the longitude
	max_longitude_station = max(longitudes, key=longitudes.get)

	mlt = pd.read_feather(f'{supermag_dir}{max_longitude_station}.feather')
	mlt.set_index('Date_UTC', inplace=True)
	mlt = mlt['MLT']

	# getting the MLT information for each prediction
	predictions = predictions.join(mlt, how='left')

	return predictions, round(latitude, 2)


def plotting_simple_scatter(all_predictions):

	''' Function that plots a simple scatter plot of the predictions vs the actual values'''

	for region in all_predictions.keys():
		predictions = all_predictions[region]['dataframe']
		fig = plt.figure(figsize=(20,10))
		ax = fig.add_subplot(111)
		ax.scatter(predictions['actual'], predictions['predicted'], s=10)
		ax.set_xlabel('Actual Values')
		ax.set_ylabel('Predictions')
		ax.set_title(f'Predictions vs Actual Values {region}')
		ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='r')

		plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{VERSION}/simple_scatter_plot.png')


def prediction_error_vs_MLT(all_predictions):

	for region in all_predictions.keys():
		fig = plt.figure(figsize=(20,10))
		ax = fig.add_subplot(111)
		predictions = all_predictions[region]['dataframe']
		error = predictions['actual'] - predictions['predicted']
		temp_df = pd.DataFrame({'MLT':predictions['MLT'], 'error':error})
		temp_df.dropna(inplace=True)
		plt.hist2d(temp_df['MLT'], temp_df['error'], bins=25, cmap='magma', norm='log')
		ax.set_xlabel('Actual Values')
		ax.set_ylabel('Predictions')
		ax.set_title(f'Error vs MLT   Lat: {all_predictions[region]["average_mlat"]}')

		plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{VERSION}/hist2d_plot_with_latitude_{region}.png')


def plotting_reliability_diagram(all_predictions):

	''' Function that plots the reliability diagram for the predictions.'''

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	for region in all_predictions.keys():
		# getting the reliability diagram
		predictions = all_predictions[region]['dataframe']
		fraction_of_positives, mean_predicted_value = calibration_curve(predictions['actual'], predictions['predicted'], n_bins=10)
		plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=all_predictions[region]['average_mlat'])
	ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
	ax.set_xlabel('Mean Predicted Value')
	ax.set_ylabel('Fraction of Positives')
	ax.set_title('Reliability Diagram')
	ax.legend()

	plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{VERSION}/reliability_diagram.png')


def plotting_precision_recall_curve(all_predictions):

	''' Function that plots the precision recall curve for the predictions.'''

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	for region in all_predictions.keys():
		predictions = all_predictions[region]['dataframe']
		# getting the precision recall curve
		precision, recall, thresholds = precision_recall_curve(predictions['actual'], predictions['predicted'])
		# getting the area under the curve
		auc_score = auc(recall, precision)

		plt.plot(recall, precision, label=f'{all_predictions[region]["average_mlat"]}; AUC: {auc_score:.3f}')
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_title(f'Precision Recall Curve; AUC: {auc_score:.3f}')
	ax.legend()

	plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{VERSION}/precision_recall_curve.png')


def plotting_predicted_values_vs_latitude(all_predictions):

	''' Function that plots the predicted values vs the latitude for each prediction.'''

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	for region in all_predictions.keys():
		predictions = all_predictions[region]['dataframe']
		plt.scatter([all_predictions[region]['average_mlat'] for i in range(len(predictions))], predictions['predicted'], s=10)
	ax.set_xlabel('Latitude')
	ax.set_ylabel('Predicted Values')
	ax.set_title('Predicted Values vs Latitude')

	plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{VERSION}/predicted_values_vs_latitude.png')


def calculating_scores(all_predictions):

	'''
	Function that calculates the scores for the predictions.
	'''
	all_scores = pd.DataFrame()
	for region in all_predictions.keys():
		predictions = all_predictions[region]['dataframe']

		# manually calculating the elements of the confusion matrix using 0.5 as the probabalistic prediction threshold
		Tp = len(predictions[(predictions['actual'] == 1) & (predictions['predicted'] >= 0.5)])
		Tn = len(predictions[(predictions['actual'] == 0) & (predictions['predicted'] < 0.5)])
		Fp = len(predictions[(predictions['actual'] == 0) & (predictions['predicted'] >= 0.5)])
		Fn = len(predictions[(predictions['actual'] == 1) & (predictions['predicted'] < 0.5)])

		# calculating brier score
		Bs = (brier_score_loss(predictions['actual'], predictions['predicted']))

		# calculating the heidke skill score
		Hss = 2*((Tp*Tn)-(Fp*Fn))/(((Tp+Fn)*(Tn+Fn))+((Tp+Fp)*(Tn+Fp)))

		# calculating the precision and recall
		try:
			precision = Tp/(Tp+Fp)
		except ZeroDivisionError:
			precision = 0

		try:
			recall = Tp/(Tp+Fn)
		except ZeroDivisionError:
			recall = 0

		# calculating the f1 score
		try:
			f1 = 2*((precision*recall)/(precision+recall))
		except ZeroDivisionError:
			f1 = 0

		# calculating the rmse
		rmse = np.sqrt(mean_squared_error(predictions['actual'], predictions['predicted']))

		# putting it all into a dataframe
		scores = pd.DataFrame({'True_pos':Tp, 'False_pos':Fp, 'False_neg': Fn, 'True_neg':Tn,
									'Brier Score':Bs, 'Hss':Hss, 'precision':precision,
									'recall':recall, 'f1':f1, 'rmse':rmse}, index=[region])

		all_scores = pd.concat([all_scores, scores.T], axis=1)

	return all_scores


def main():

	if not os.path.exists(f'plots/{TARGET}/analysis_plots_modeling_v{VERSION}/'):
		os.makedirs(f'plots/{TARGET}/analysis_plots_modeling_v{VERSION}/')

	if not os.path.exists(f'outputs/{TARGET}/analysis_plots_modeling_v{VERSION}/'):
		os.makedirs(f'outputs/{TARGET}/analysis_plots_modeling_v{VERSION}/')

	all_predictions = {region:{} for region in REGIONS}

	for region in REGIONS:
		predictions = load_predictions(region)
		predictions, average_mlat = attaching_mlt(predictions, region)
		all_predictions[region]['dataframe'] = predictions
		all_predictions[region]['average_mlat'] = average_mlat

	# plotting a simple scatter plot of the predictions vs the actual values
	plotting_simple_scatter(all_predictions)

	# plotting a simple scatter plot of the predictions vs the actual values with the latitude information for each prediction
	prediction_error_vs_MLT(all_predictions)

	# plotting the reliability diagram for the predictions
	plotting_reliability_diagram(all_predictions)

	# plotting the precision recall curve for the predictions
	plotting_precision_recall_curve(all_predictions)

	# plotting the predicted values vs the latitude for each prediction
	plotting_predicted_values_vs_latitude(all_predictions)

	# calculating the scores for the predictions
	scores = calculating_scores(all_predictions)
	scores.to_csv(f'outputs/{TARGET}/analysis_plots_modeling_v{VERSION}/scores.csv')

	print('Done!')



if __name__ == '__main__':
	main()
