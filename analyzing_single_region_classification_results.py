import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, brier_score_loss, confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_curve)

REGIONS = [194, 270, 287, 207, 62, 241, 366, 387, 223, 19, 163]

VERSION = 0

def load_predictions():

	if os.path.exists(f'outputs/non_twins_modeling_region_{region}_version_{VERSION}.feather'):
		predictions = pd.read_feather(f'outputs/non_twins_modeling_region_{region}_version_{VERSION}.feather')
		predictions.set_index('Date_UTC', inplace=True)
	else:
		raise(f'You fool! You need to run the script modeling_v{VERSION}.py first. Throw yourself down next time and rid us of your stupidity!')


	return predictions


def load_segmented_mlt_dict():

	with open(f'outputs/mlt_span_{MLT_SPAN}_dbdt.pkl', 'rb') as f:
		segmented_mlt = pickle.load(f)

	return segmented_mlt


def plotting_simple_scatter(predictions):

	''' Function that plots a simple scatter plot of the predictions vs the actual values'''

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.scatter(predictions['actual'], predictions['predicted'], s=10)
	ax.set_xlabel('Actual Values')
	ax.set_ylabel('Predictions')
	ax.set_title('Predictions vs Actual Values')
	ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='r')

	plt.savefig(f'plots/analysis_plots_modeling_v{VERSION}/simple_scatter_plot.png')


def simple_scatter_with_latitude_information(predictions, segmented_mlt):

	''' Function that plots a simple scatter plot of the predictions vs the actual values
			with the latitude information for each prediction.'''

	# getting the latitude information for each prediction
	predictions['latitude'] = predictions.index.map(lambda x: segmented_mlt[f'{MLT_BIN_TARGET}'].loc[x.strftime(format='%Y-%m-%d %H:%M:%S')]['mean_lat'])

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	scatter = ax.scatter(predictions['actual'], predictions['predicted'], s=10, c=predictions['latitude'], cmap='plasma')
	plt.colorbar(scatter)
	ax.set_xlabel('Actual Values')
	ax.set_ylabel('Predictions')
	ax.set_title('Predictions vs Actual Values')
	ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='r')

	plt.savefig(f'plots/analysis_plots_modeling_v{VERSION}/simple_scatter_plot_with_latitude.png')


def plotting_reliability_diagram(predictions):

	''' Function that plots the reliability diagram for the predictions.'''

	# getting the reliability diagram
	fraction_of_positives, mean_predicted_value = calibration_curve(predictions['actual'], predictions['predicted'], n_bins=10)

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='MLT Bin')
	ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
	ax.set_xlabel('Mean Predicted Value')
	ax.set_ylabel('Fraction of Positives')
	ax.set_title('Reliability Diagram')
	ax.legend()

	plt.savefig(f'plots/analysis_plots_modeling_v{VERSION}/reliability_diagram.png')


def plotting_precision_recall_curve(predictions):

	''' Function that plots the precision recall curve for the predictions.'''

	# getting the precision recall curve
	precision, recall, thresholds = precision_recall_curve(predictions['actual'], predictions['predicted'])

	# getting the area under the curve
	auc_score = auc(recall, precision)

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.plot(recall, precision, label='MLT Bin')
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_title(f'Precision Recall Curve; AUC: {auc_score:.3f}')
	ax.legend()

	plt.savefig(f'plots/analysis_plots_modeling_v{VERSION}/precision_recall_curve.png')


def plotting_error_vs_latitude(predictions, segmented_mlt):

	''' Function that plots the error vs the latitude for each prediction.'''

	# getting the latitude information for each prediction
	predictions['latitude'] = predictions.index.map(lambda x: segmented_mlt[f'{MLT_BIN_TARGET}'].loc[x.strftime(format='%Y-%m-%d %H:%M:%S')]['mean_lat'])

	# getting the error for each prediction
	predictions['error'] = predictions['predicted'] - predictions['actual']

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.scatter(predictions['latitude'], predictions['error'], s=10)
	ax.set_xlabel('Latitude')
	ax.set_ylabel('Error')
	ax.set_title('Error vs Latitude')

	plt.savefig(f'plots/analysis_plots_modeling_v{VERSION}/error_vs_latitude.png')

def plotting_actual_values_vs_latitude(predictions, segmented_mlt):

	''' Function that plots the actual values vs the latitude for each prediction.'''

	# getting the latitude information for each prediction
	predictions['latitude'] = predictions.index.map(lambda x: segmented_mlt[f'{MLT_BIN_TARGET}'].loc[x.strftime(format='%Y-%m-%d %H:%M:%S')]['mean_lat'])

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.scatter(predictions['latitude'], predictions['actual'], s=10)
	ax.set_xlabel('Latitude')
	ax.set_ylabel('Actual Values')
	ax.set_title('Actual Values vs Latitude')

	plt.savefig(f'plots/analysis_plots_modeling_v{VERSION}/actual_values_vs_latitude.png')


def plotting_predicted_values_vs_latitude(predictions, segmented_mlt):

	''' Function that plots the predicted values vs the latitude for each prediction.'''

	# getting the latitude information for each prediction
	predictions['latitude'] = predictions.index.map(lambda x: segmented_mlt[f'{MLT_BIN_TARGET}'].loc[x.strftime(format='%Y-%m-%d %H:%M:%S')]['mean_lat'])

	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.scatter(predictions['latitude'], predictions['predicted'], s=10)
	ax.set_xlabel('Latitude')
	ax.set_ylabel('Predicted Values')
	ax.set_title('Predicted Values vs Latitude')

	plt.savefig(f'plots/analysis_plots_modeling_v{VERSION}/predicted_values_vs_latitude.png')


def calculating_scores(predictions):

	'''
	Function that calculates the scores for the predictions.
	'''

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
								'recall':recall, 'f1':f1, 'rmse':rmse}, index=[0])

	return scores


def main():

	if not os.path.exists(f'plots/analysis_plots_modeling_v{VERSION}/'):
		os.makedirs(f'plots/analysis_plots_modeling_v{VERSION}/')

	if not os.path.exists(f'outputs/analysis_plots_modeling_v{VERSION}/'):
		os.makedirs(f'outputs/analysis_plots_modeling_v{VERSION}/')

	predictions = load_predictions(use_dict=False)
	segmented_mlt = load_segmented_mlt_dict()

	# plotting a simple scatter plot of the predictions vs the actual values
	plotting_simple_scatter(predictions)

	# plotting a simple scatter plot of the predictions vs the actual values with the latitude information for each prediction
	simple_scatter_with_latitude_information(predictions, segmented_mlt)

	# plotting the reliability diagram for the predictions
	plotting_reliability_diagram(predictions)

	# plotting the precision recall curve for the predictions
	plotting_precision_recall_curve(predictions)

	# plotting the error vs the latitude for each prediction
	plotting_error_vs_latitude(predictions, segmented_mlt)

	# plotting the actual values vs the latitude for each prediction
	plotting_actual_values_vs_latitude(predictions, segmented_mlt)

	# plotting the predicted values vs the latitude for each prediction
	plotting_predicted_values_vs_latitude(predictions, segmented_mlt)

	# calculating the scores for the predictions
	scores = calculating_scores(predictions)
	scores.to_csv(f'outputs/analysis_plots_modeling_v{VERSION}/scores.csv', index=False)

	print('Done!')



if __name__ == '__main__':
	main()
