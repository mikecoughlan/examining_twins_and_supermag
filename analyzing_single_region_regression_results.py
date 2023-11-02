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

VERSION = 1
TARGET = 'rsd'

def load_predictions(region, version=VERSION):

	if os.path.exists(f'outputs/{TARGET}/non_twins_modeling_region_{region}_version_{version}.feather'):
		predictions = pd.read_feather(f'outputs/{TARGET}/non_twins_modeling_region_{region}_version_{version}.feather')
		predictions.set_index('dates', inplace=True)
		predictions.index = pd.to_datetime(predictions.index, format = '%Y-%m-%d %H:%M:%S')

	else:
		raise(f'Fool of a Took! You need to run the script modeling_v{version}.py first. Throw yourself down next time and rid us of your stupidity!')

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


def plotting_simple_scatter(all_predictions, version=VERSION):

	''' Function that plots a simple scatter plot of the predictions vs the actual values'''

	for region in all_predictions.keys():
		predictions = all_predictions[region]['dataframe'].dropna(inplace=False, subset=['actual', 'predicted'])
		fig = plt.figure(figsize=(20,10))
		ax = fig.add_subplot(111)
		ax.scatter(predictions['actual'], predictions['predicted'], s=10, label=f'R^2: {r2_score(predictions["actual"], predictions["predicted"]):.3f}')
		plt.legend()
		ax.set_xlabel('Actual Values')
		ax.set_ylabel('Predictions')
		ax.set_title(f'Predictions vs Actual Values {region}')
		ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='r')

		plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{version}/{region}_simple_scatter_plot.png')


def prediction_error_vs_MLT(all_predictions, version=VERSION):

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

		plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{version}/hist2d_plot_with_latitude_{region}.png')


def correlations_vs_mlat(all_predictions, version=VERSION):

	r2 = []

	for region in all_predictions:
		predictions = all_predictions[region]['dataframe'].dropna(inplace=False, subset=['actual', 'predicted'])
		r2.append(r2_score(predictions['actual'], predictions['predicted']))

	mlat = [all_predictions[region]['average_mlat'] for region in all_predictions.keys()]

	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111)
	ax.scatter(mlat, r2, s=10)
	ax.set_xlabel('Latitude')
	ax.set_ylabel('R^2')
	ax.set_title('R^2 vs Latitude')

	plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{version}/r2_vs_latitude.png')


def plotting_reliability_diagram(all_predictions, version=VERSION):

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

	plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{version}/reliability_diagram.png')


def plotting_precision_recall_curve(all_predictions, version=VERSION):

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

	plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{version}/precision_recall_curve.png')


def plotting_predicted_values_vs_latitude(all_predictions, version=VERSION):

	''' Function that plots the predicted values vs the latitude for each prediction.'''

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	for region in all_predictions.keys():
		predictions = all_predictions[region]['dataframe']
		plt.scatter([all_predictions[region]['average_mlat'] for i in range(len(predictions))], predictions['predicted'], s=10)
	ax.set_xlabel('Latitude')
	ax.set_ylabel('Predicted Values')
	ax.set_title('Predicted Values vs Latitude')

	plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{version}/predicted_values_vs_latitude.png')


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


def handling_gaps(df, threshold):
	'''
	Function for keeping blocks of nans in the data if there is a maximum number of data points between sucessive valid data.
	If the number of nans is too large between sucessive data points it will drop those nans.

	Args:
		df (pd.DataFrame): data to be processed

	Returns:
		pd.DataFrame: processed data
	'''
	start_time = pd.to_datetime('2009-07-19')
	end_time = pd.to_datetime('2017-12-31')
	date_range = pd.date_range(start_time, end_time, freq='min')

	full_time_df = pd.DataFrame(index=date_range)

	df = full_time_df.join(df, how='left')

	# creting a column in the data frame that labels the size of the gaps
	df['gap_size'] = df['actual'].isna().groupby(df['actual'].notna().cumsum()).transform('sum')

	# setting teh gap size column to nan if the value is above the threshold, setting it to 0 otherwise
	df['gap_size'] = np.where(df['gap_size'] > threshold, np.nan, 0)

	# dropping nans from the subset of the gap size column
	df.dropna(inplace=True, subset=['gap_size'])

	# dropping the gap size column
	df.drop(columns=['gap_size'], inplace=True)

	return df


def line_plot(all_predictions=None, std=False, version=VERSION, multiple_models=False):
	'''
	Function that plots the output predictions in a time series. If the model used the CRPS as a loss function
	the plot will include the standard deviation of the predictions as a shaded region around the mean.

	Args:
		all_predictions (dict): dictonary of the model results and real data for each region
		std (bool, optional): if there is a standard deviation to plot around the mean. Defaults to False.
	'''

	segmenting_int = 2000
	gap_tolerance = 100

	# check if the version keyword is a list or an int
	if multiple_models:
		if not isinstance(version, list):
			raise ValueError('If multiple_models is True, version must be an int.')

	if multiple_models:

		all_predictions = {region:{} for region in REGIONS}
		for region in REGIONS:
			for i in version:
				all_predictions[region][f'version_{i}'] = {}
				prediction = load_predictions(region, version=i)
				prediction, average_mlat = attaching_mlt(prediction, region)
				prediction = handling_gaps(prediction, gap_tolerance)
				all_predictions[region][f'version_{i}']['dataframe'] = prediction
				all_predictions[region][f'version_{i}']['average_mlat'] = average_mlat

		for region in all_predictions.keys():

			fig = plt.figure(figsize=(20,10))
			ax = fig.add_subplot(111)
			ax.plot(all_predictions[region][f'version_{version[0]}']['dataframe']['actual'].iloc[segmenting_int:segmenting_int+1000], label='Actual', color='k')
			for ver in version:
				predictions = all_predictions[region][f'version_{ver}']['dataframe'].iloc[segmenting_int:segmenting_int+1000]
				if 'predicted_mean' in predictions.columns:
					ax.plot(predictions['predicted_mean'], label=f'Predicted Mean version {ver}', color='blue')
				else:
					ax.plot(predictions['predicted'], label=f'Deterministic version {ver}', color='red')
				if 'predicted_std' in predictions.columns:
					ax.fill_between(predictions.index, predictions['predicted_mean']-predictions['predicted_std'],
										predictions['predicted_mean']+predictions['predicted_std'], alpha=0.2, label=f'Predicted Std version {ver}', color='blue')
			ax.set_xlabel('Date')
			ax.set_ylabel('Predicted Values')
			ax.set_title(f'Predicted Values vs Actual Values {region}')
			ax.legend()

			plt.savefig(f'plots/{TARGET}/multiple_versions/{region}_line_plot_versions_{version}.png')

	else:
		for region in all_predictions.keys():
			predictions = all_predictions[region]['dataframe']
			predictions = handling_gaps(predictions, 100)
			predictions = predictions.iloc[i:i+1000]
			fig = plt.figure(figsize=(20,10))
			ax = fig.add_subplot(111)
			ax.plot(predictions['actual'], label='Actual')
			ax.plot(predictions['predicted'], label='Predicted Mean')
			if std:
				ax.fill_between(predictions.index, predictions['predicted']-predictions['predicted_std'], predictions['predicted']+predictions['predicted_std'], color='gray', alpha=0.5, label='Predicted Std')
			ax.set_xlabel('Date')
			ax.set_ylabel('Predicted Values')
			ax.set_title(f'Predicted Values vs Actual Values {region}')
			ax.legend()

			plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{version}/{region}_line_plot.png')

def checking_error_distributions(all_predictions, version=VERSION):

	errors_df = pd.Series()
	for region in all_predictions.keys():
		errors = (all_predictions[region]['dataframe']['actual'] - all_predictions[region]['dataframe']['predicted'])
		errors_df = pd.concat([errors_df, errors])

	fig = plt.figure(figsize=(10,5))
	plt.hist(errors_df)
	plt.xlabel('errors')
	plt.ylabel('count')
	plt.savefig(f'plots/{TARGET}/analysis_plots_modeling_v{version}/error_distributions.png')


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
	# plotting_simple_scatter(all_predictions)

	# plotting a simple scatter plot of the predictions vs the actual values with the latitude information for each prediction
	# prediction_error_vs_MLT(all_predictions)

	# correlations_vs_mlat(all_predictions)
	
	checking_error_distributions(all_predictions)

	#line_plot(multiple_models=True, version=[1,2])

	# plotting the reliability diagram for the predictions
	# plotting_reliability_diagram(all_predictions)

	# plotting the predicted values vs the latitude for each prediction
	# plotting_predicted_values_vs_latitude(all_predictions)

	# calculating the scores for the predictions
	# scores = calculating_scores(all_predictions)
	# scores.to_csv(f'outputs/{TARGET}/analysis_plots_modeling_v{version}/scores.csv')

	print('Done!')



if __name__ == '__main__':
	main()
