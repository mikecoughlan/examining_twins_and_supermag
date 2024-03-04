####################################################################################
#
# exmining_twins_and_supermag/non_twins_modeling_v0.py
#
# Performing the modeling using the Solar Wind and Ground Magnetomoeter data.
# TWINS data passes through a pre-trained autoencoder that reduces the TWINS maps
# to a reuced dimensionality. This data is then concatenated onto the model after
# both branches of the CNN hae been flattened, and before the dense layers.
# Similar model to Coughlan (2023) but with a different target variable.
#
####################################################################################


# Importing the libraries
import datetime
import gc
import glob
import json
import os
import pickle
import subprocess
import time

# import keras
import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import shapely
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from torch.utils.data import DataLoader, Dataset, TensorDataset

import utils

pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

TARGET = 'rsd'
REGION = 163
RANDOM_SEED = 7
BATCH_SIZE = 16
VERSION = 'non_twins_pytorch_v0-1'

working_dir = '../../../../data/mike_working_dir/'
region_path = working_dir+'identifying_regions_data/adjusted_regions.pkl'
solarwind_path = '../data/SW/omniData.feather'
supermag_dir_path = '../data/supermag/'
twins_times_path = 'outputs/regular_twins_map_dates.feather'
rsd_path = working_dir+'identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

CONFIG = {'region_numbers': [387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
								83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
								62, 327, 293, 241, 107, 55, 111, 401, 400],
			'time_history':30,
			'random_seed':7}



def loading_data(target_var, region, percentiles=[0.5, 0.75, 0.9, 0.99]):

	# loading all the datasets and dictonaries

	regions, stats = utils.loading_dicts()
	solarwind = utils.loading_solarwind(omni=True, limit_to_twins=True)

	# converting the solarwind data to log10
	solarwind['logT'] = np.log10(solarwind['T'])
	solarwind.drop(columns=['T'], inplace=True)

	# reduce the regions dict to be only the ones that have keys in the region_numbers list
	regions = regions[f'region_{region}']
	stats = stats[f'region_{region}']

	# getting dbdt and rsd data for the region
	supermag_df = utils.combining_stations_into_regions(regions['station'], stats, features=['dbht', 'MAGNITUDE', 'theta', 'N', 'E', 'sin_theta', 'cos_theta'], mean=True, std=True, maximum=True, median=True)

	# getting the mean latitude for the region and attaching it to the regions dictionary
	mean_lat = utils.getting_mean_lat(regions['station'])

	thresholds = [supermag_df[target_var].quantile(percentile) for percentile in percentiles]

	merged_df = pd.merge(supermag_df, solarwind, left_index=True, right_index=True, how='inner')


	return merged_df, mean_lat, thresholds



def getting_prepared_data(target_var, region, get_features=False, do_scaling=True):
	'''
	Calling the data prep class without the TWINS data for this version of the model.

	Returns:
		X_train (np.array): training inputs for the model
		X_val (np.array): validation inputs for the model
		X_test (np.array): testing inputs for the model
		y_train (np.array): training targets for the model
		y_val (np.array): validation targets for the model
		y_test (np.array): testing targets for the model

	'''

	merged_df, mean_lat, thresholds = loading_data(target_var=target_var, region=region, percentiles=[0.5, 0.75, 0.9, 0.99])

	# target = merged_df['classification']
	target = merged_df[f'rolling_{target_var}']

	# reducing the dataframe to only the features that will be used in the model plus the target variable
	vars_to_keep = [f'rolling_{target_var}', 'dbht_median', 'MAGNITUDE_median', 'MAGNITUDE_std', 'sin_theta_std', 'cos_theta_std', 'cosMLT', 'sinMLT',
					'B_Total', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'proton_density', 'logT']
	merged_df = merged_df[vars_to_keep]

	print('Columns in Merged Dataframe: '+str(merged_df.columns))

	temp_version = 'final_1'

	# loading the data corresponding to the twins maps if it has already been calculated
	if os.path.exists(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl'):
		with open(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl', 'rb') as f:
			storms_extracted_dict = pickle.load(f)
		storms = storms_extracted_dict['storms']
		target = storms_extracted_dict['target']

	# if not, calculating the twins maps and extracting the storms
	else:
		storms, target = utils.storm_extract(df=merged_df, lead=30, recovery=9, twins=True, target=True, target_var=f'rolling_{target_var}', concat=False)
		storms_extracted_dict = {'storms':storms, 'target':target}
		with open(working_dir+f'twins_method_storm_extraction_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl', 'wb') as f:
			pickle.dump(storms_extracted_dict, f)

	# making sure the target variable has been dropped from the input data
	print('Columns in Dataframe: '+str(storms[0].columns))

	# getting the feature names
	features = storms[0].columns

	# splitting the data on a day to day basis to reduce data leakage
	day_df = pd.date_range(start=pd.to_datetime('2009-07-01'), end=pd.to_datetime('2017-12-01'), freq='D')
	specific_test_days = pd.date_range(start=pd.to_datetime('2012-03-07'), end=pd.to_datetime('2012-03-13'), freq='D')
	day_df = day_df.drop(specific_test_days)

	train_days, test_days = train_test_split(day_df, test_size=0.1, shuffle=True, random_state=CONFIG['random_seed'])
	train_days, val_days = train_test_split(train_days, test_size=0.125, shuffle=True, random_state=CONFIG['random_seed'])

	test_days = test_days.tolist()
	# adding the two dateimte values of interest to the test days df
	test_days = pd.to_datetime(test_days)
	test_days.append(specific_test_days)

	train_dates_df, val_dates_df, test_dates_df = pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]})
	x_train, x_val, x_test, y_train, y_val, y_test, twins_train, twins_val, twins_test = [], [], [], [], [], [], [], [], []

	# using the days to split the data
	for day in train_days:
		train_dates_df = pd.concat([train_dates_df, pd.DataFrame({'dates':pd.date_range(start=day, end=day+pd.DateOffset(days=1), freq='min')})], axis=0)

	for day in val_days:
		val_dates_df = pd.concat([val_dates_df, pd.DataFrame({'dates':pd.date_range(start=day, end=day+pd.DateOffset(days=1), freq='min')})], axis=0)

	for day in test_days:
		test_dates_df = pd.concat([test_dates_df, pd.DataFrame({'dates':pd.date_range(start=day, end=day+pd.DateOffset(days=1), freq='min')})], axis=0)

	train_dates_df.set_index('dates', inplace=True)
	val_dates_df.set_index('dates', inplace=True)
	test_dates_df.set_index('dates', inplace=True)

	train_dates_df.index = pd.to_datetime(train_dates_df.index)
	val_dates_df.index = pd.to_datetime(val_dates_df.index)
	test_dates_df.index = pd.to_datetime(test_dates_df.index)

	date_dict = {'train':pd.DataFrame(), 'val':pd.DataFrame(), 'test':pd.DataFrame()}

	# getting the data corresponding to the dates
	for storm, y in zip(storms, target):

		copied_storm = storm.copy()
		copied_storm = copied_storm.reset_index(inplace=False, drop=False).rename(columns={'index':'Date_UTC'})

		if storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in train_dates_df.index:
			x_train.append(storm)
			y_train.append(y)
			date_dict['train'] = pd.concat([date_dict['train'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in val_dates_df.index:
			x_val.append(storm)
			y_val.append(y)
			date_dict['val'] = pd.concat([date_dict['val'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in test_dates_df.index:
			x_test.append(storm)
			y_test.append(y)
			date_dict['test'] = pd.concat([date_dict['test'], copied_storm['Date_UTC'][-10:]], axis=0)

	date_dict['train'].reset_index(drop=True, inplace=True)
	date_dict['val'].reset_index(drop=True, inplace=True)
	date_dict['test'].reset_index(drop=True, inplace=True)

	date_dict['train'].rename(columns={date_dict['train'].columns[0]:'Date_UTC'}, inplace=True)
	date_dict['val'].rename(columns={date_dict['val'].columns[0]:'Date_UTC'}, inplace=True)
	date_dict['test'].rename(columns={date_dict['test'].columns[0]:'Date_UTC'}, inplace=True)

	to_scale_with = pd.concat(x_train, axis=0)
	scaler = StandardScaler()
	scaler.fit(to_scale_with)
	if do_scaling:
		x_train = [scaler.transform(x) for x in x_train]
		x_val = [scaler.transform(x) for x in x_val]
		x_test = [scaler.transform(x) for x in x_test]

	# saving the scaler
	with open(f'models/{TARGET}/non_twins_region_{region}_version_{VERSION}_scaler.pkl', 'wb') as f:
		pickle.dump(scaler, f)

	print(f'shape of x_train: {len(x_train)}')
	print(f'shape of x_val: {len(x_val)}')
	print(f'shape of x_test: {len(x_test)}')

	# splitting the sequences for input to the CNN
	x_train, y_train, train_dates_to_drop, __ = utils.split_sequences(x_train, y_train, n_steps=CONFIG['time_history'], dates=date_dict['train'], model_type='regression')
	x_val, y_val, val_dates_to_drop, __ = utils.split_sequences(x_val, y_val, n_steps=CONFIG['time_history'], dates=date_dict['val'], model_type='regression')
	x_test, y_test, test_dates_to_drop, __  = utils.split_sequences(x_test, y_test, n_steps=CONFIG['time_history'], dates=date_dict['test'], model_type='regression')

	# dropping the dates that correspond to arrays that would have had nan values
	date_dict['train'].drop(train_dates_to_drop, axis=0, inplace=True)
	date_dict['val'].drop(val_dates_to_drop, axis=0, inplace=True)
	date_dict['test'].drop(test_dates_to_drop, axis=0, inplace=True)

	date_dict['train'].reset_index(drop=True, inplace=True)
	date_dict['val'].reset_index(drop=True, inplace=True)
	date_dict['test'].reset_index(drop=True, inplace=True)

	print(f'Total training dates: {len(date_dict["train"])}')

	print(f'shape of x_train: {x_train.shape}')
	print(f'shape of x_val: {x_val.shape}')
	print(f'shape of x_test: {x_test.shape}')

	print(f'Nans in training data: {np.isnan(x_train).sum()}')
	print(f'Nans in validation data: {np.isnan(x_val).sum()}')
	print(f'Nans in testing data: {np.isnan(x_test).sum()}')

	print(f'Nans in training target: {np.isnan(y_train).sum()}')
	print(f'Nans in validation target: {np.isnan(y_val).sum()}')
	print(f'Nans in testing target: {np.isnan(y_test).sum()}')

	if not get_features:
		return torch.tensor(x_train), torch.tensor(x_val), torch.tensor(x_test), torch.tensor(y_train), torch.tensor(y_val), torch.tensor(y_test), date_dict
	else:
		return torch.tensor(x_train), torch.tensor(x_val), torch.tensor(x_test), torch.tensor(y_train), torch.tensor(y_val), torch.tensor(y_test), date_dict, features

class CRSP(nn.Module):
	'''
	Defining the CRPS loss function for model training.
	'''

	def __init__(self):
		super(CRSP, self).__init__()

	def forward(self, y_pred, y_true):

		print(f'within loss function y_pred size: {y_pred.size()}')

		# splitting the y_pred tensor into mean and std

		mean, std = torch.unbind(y_pred, dim=-1)
		y_true = torch.unbind(y_true, dim=-1)

		# making the arrays the right dimensions
		mean = mean.unsqueeze(-1)
		std = std.unsqueeze(-1)
		y_true = y_true.unsqueeze(-1)

		# calculating the error
		crps = torch.mean(self.calculate_crps(self.epsilon_error(y_true, mean), std))

		return crps

	def epsilon_error(self, y, u):

		epsilon = torch.abs(y - u)

		return epsilon

	def calculate_crps(self, epsilon, sig):

		crps = sig * ((epsilon / sig) * torch.erf((epsilon / (np.sqrt(2) * sig))) + torch.sqrt(torch.tensor(2 / np.pi)) * torch.exp(-epsilon ** 2 / (2 * sig ** 2)) - 1 / torch.sqrt(torch.tensor(np.pi)))

		return crps

class PrintLayer(nn.Module):
	def __init__(self):
		super(PrintLayer, self).__init__()

	def forward(self, x):
		# Do your print / debug stuff here
		print(x.size())
		return x

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.model = nn.Sequential(

			nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2, stride=1, padding='same'),
			nn.ReLU(),
			nn.Dropout(0.2),
			PrintLayer(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			PrintLayer(),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding='same'),
			nn.ReLU(),
			nn.Dropout(0.2),
			PrintLayer(),
			nn.Flatten(),
			nn.Linear(256*15*7, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			PrintLayer(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			PrintLayer(),
			nn.Linear(128, 2),
			PrintLayer()
		)

	def forward(self, x):

		x = self.model(x)

		return x


class Early_Stopping():
	'''
	Class to create an early stopping condition for the model.

	'''

	def __init__(self, decreasing_loss_patience=25):
		self.decreasing_loss_patience = decreasing_loss_patience
		self.loss_counter = 0
		self.best_score = None
		self.early_stop = False
		self.best_epoch = None

	def __call__(self, train_loss, val_loss, model, epoch):
		self.model = model
		if self.best_score is None:
			self.best_score = val_loss
			self.best_loss = val_loss
			self.save_checkpoint(val_loss)
			self.best_epoch = epoch
		elif val_loss > self.best_score:
			self.loss_counter += 1
			if self.loss_counter >= self.decreasing_loss_patience:
				print(f'Engaging Early Stopping due to lack of improvement in validation loss. Best model saved at epoch {self.best_epoch} with a training loss of {self.best_loss} and a validation loss of {self.best_score}')
				return True
		# elif val_loss > (1.5 * train_loss):
		# 	self.training_counter += 1
		# 	if self.training_counter >= self.training_diff_patience:
		# 		print(f'Engaging Early Stopping due to large seperation between train and val loss. Best model saved at epoch {self.best_epoch} with a training loss of {self.best_loss} and a validation loss of {self.best_score}')
		# 		return True

		else:
			self.best_score = val_loss
			self.best_epoch = epoch
			self.save_checkpoint(val_loss)
			self.loss_counter = 0
			self.training_counter = 0

			return False

	def save_checkpoint(self, val_loss):
		if self.best_loss > val_loss:
			self.best_loss = val_loss
			torch.save(self.model.state_dict(), f'models/non_twins_{VERSION}.pt')


def fit_model(model, train, val, val_loss_patience=25, num_epochs=500):

	if not os.path.exists(f'models/non_twins_{VERSION}.pt'):

		train_loss_list, val_loss_list = [], []

		# moving the model to the available device
		model.to(DEVICE)

		# defining the loss function and the optimizer
		criterion = CRSP()
		optimizer = optim.Adam(model.parameters(), lr=1e-7)
		scaler = torch.cuda.amp.GradScaler()

		# initalizing the early stopping class
		early_stopping = Early_Stopping(decreasing_loss_patience=val_loss_patience)

		for epoch in range(num_epochs):

			# starting the clock for the epoch
			stime = time.time()

			# setting the model to training mode
			model.train()

			# initializing the running loss
			running_training_loss, running_val_loss = 0.0, 0.0

			# shuffling data and creating batches
			for x, y in train:
				x = x.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)

				x = x.unsqueeze(1)

				# forward pass
				with torch.cuda.amp.autocast():
					output = model(x)
					print(f'output size: {output.size()}')

					loss = criterion(output, y)
					# loss.requires_grad = True

				# backward pass
				optimizer.zero_grad()
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()

				# adding the loss to the running loss
				running_training_loss += loss.to('cpu').item()

			# setting the model to evaluation mode
			model.eval()

			# using validation set to check for overfitting
			with torch.no_grad():
				for x, y in val:
					x = x.to(DEVICE, dtype=torch.float)
					y = y.to(DEVICE, dtype=torch.float)

					x = x.unsqueeze(1)

					output = model(x)

					val_loss = criterion(output, y)

					# adding the loss to the running loss
					running_val_loss += val_loss.to('cpu').item()

			# getting the average loss for the epoch
			loss = running_training_loss/len(train)
			val_loss = running_val_loss/len(val)

			# adding the loss to the list
			train_loss_list.append(loss)
			val_loss_list.append(val_loss)

			# checking for early stopping
			if early_stopping(train_loss=loss, val_loss=val_loss, model=model, epoch=epoch):
				break

			# getting the time for the epoch
			epoch_time = time.time() - stime

			# if epoch % 5 == 0:
			print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f} Validation Loss: {val_loss:.4f}' + f' Epoch Time: {epoch_time:.2f} seconds')

			# emptying the cuda cache
			torch.cuda.empty_cache()

			# getting the best model

		# getting the best params saved in the Early Stopping class
		model.load_state_dict(torch.load(f'models/non_twins_{VERSION}.pt'))

		# transforming the lists to a dataframe to be saved
		loss_tracker = pd.DataFrame({'train_loss':train_loss_list, 'val_loss':val_loss_list})
		loss_tracker.to_feather(f'outputs/non_twins_{VERSION}_loss_tracker.feather')

	else:
		# loading the model if it has already been trained.
		model.load_state_dict(torch.load(f'models/non_twins_{VERSION}.pt')) 			# loading the models if already trained

	return model


def evaluation(model, test):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test_dict (dict): dictonary with the testing model inputs and the real data for comparison
		split (int): which split is being tested

	Returns:
		dict: test dict now containing columns in the dataframe with the model predictions for this split
	'''

	# creting an array to store the predictions
	predicted_list, xtest_list, ytest_list = [], [], []
	# setting the encoder and decoder into evaluation model
	model.eval()

	# creating a loss value
	running_loss = 0.0

	# making sure the model is on the correct device
	model.to(DEVICE, dtype=torch.float)

	with torch.no_grad():
		for x, y in test:

			x = x.to(DEVICE, dtype=torch.float)
			y = y.to(DEVICE, dtype=torch.float)

			x = x.unsqueeze(1)

			predicted = model(x)
			# getting shape of tensor
			print(predicted.size())
			loss = F.mse_loss(predicted[[0]], y)
			running_loss += loss.item()

			# making sure the predicted value is on the cpu
			if predicted.get_device() != -1:
				predicted = predicted.to('cpu')
			if x.get_device() != -1:
				x = x.to('cpu')
			if y.get_device() != -1:
				y = y.to('cpu')

			# adding the decoded result to the predicted list after removing the channel dimension
			predicted = torch.squeeze(predicted, dim=1).numpy()
			x = torch.squeeze(x, dim=1).numpy()

			predicted_list.append(predicted)
			xtest_list.append(x)
			ytest_list.append(y)

	return np.concatenate(predicted_list, axis=0), np.concatenate(xtest_list, axis=0), np.concatenate(ytest_list, axis=0), running_loss/len(test)


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	xtrain, xval, xtest, ytrain, yval, ytest, ___ = getting_prepared_data(target_var=TARGET, region=REGION)

	# creating the dataloaders

	train = DataLoader(list(zip(xtrain, ytrain)), batch_size=BATCH_SIZE, shuffle=True)
	val = DataLoader(list(zip(xval, yval)), batch_size=BATCH_SIZE, shuffle=True)
	test = DataLoader(list(zip(xtest, ytest)), batch_size=BATCH_SIZE, shuffle=False)

	# creating the model
	print('Creating model....')
	cnn = CNN()

	# fitting the model
	print('Fitting model....')
	cnn = fit_model(cnn, train, val, val_loss_patience=25, num_epochs=500)

	# evaluating the model
	print('Evaluating model....')
	predictions, xtest, ytest, testing_loss = evaluation(cnn, test)

	print(f'Loss: {testing_loss}')


if __name__ == '__main__':
	main()
	print('It ran. God job!')

