####################################################################################
#
# exmining_twins_and_supermag/modeling_v0.py
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
from scipy.special import expit, inv_boxcox
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from torch.utils.data import DataLoader, Dataset, TensorDataset

import utils

# from data_generator import Generator
# from data_prep import DataPrep

TARGET = 'rsd'
REGION = 163
VERSION = 'pytorch_perceptual_v1-19'

CONFIG = {'time_history':30, 'random_seed':7}


os.environ["CDF_LIB"] = "~/CDF/lib"

working_dir = '../../../../data/mike_working_dir/'
region_path = working_dir+'identifying_regions_data/adjusted_regions.pkl'
region_number = '163'
solarwind_path = '../data/SW/omniData.feather'
supermag_dir_path = '../data/supermag/'
twins_times_path = 'outputs/regular_twins_map_dates.feather'
rsd_path = working_dir+'identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'
RANDOM_SEED = 7
BATCH_SIZE = 64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


def loading_data(target_var, region):

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
	supermag_df = utils.combining_stations_into_regions(regions['station'], stats, features=['dbht', 'MAGNITUDE', \
		'theta', 'N', 'E', 'sin_theta', 'cos_theta'], mean=True, std=True, maximum=True, median=True)

	# getting the mean latitude for the region and attaching it to the regions dictionary
	mean_lat = utils.getting_mean_lat(regions['station'])

	merged_df = pd.merge(supermag_df, solarwind, left_index=True, right_index=True, how='inner')

	print('Loading TWINS maps....')
	maps = utils.loading_twins_maps()

	return merged_df, mean_lat, maps


def getting_prepared_data(target_var, region, get_features=False):
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

	maps = utils.loading_twins_maps()

	# merged_df, mean_lat, maps = loading_data(target_var=target_var, region=region)

	# # target = merged_df['classification']
	# target = merged_df[f'rolling_{target_var}']

	# # reducing the dataframe to only the features that will be used in the model plus the target variable
	# vars_to_keep = [f'rolling_{target_var}', 'dbht_median', 'MAGNITUDE_median', 'MAGNITUDE_std', 'sin_theta_std', 'cos_theta_std', 'cosMLT', 'sinMLT',
	# 				'B_Total', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'proton_density', 'logT']
	# merged_df = merged_df[vars_to_keep]

	# print('Columns in Merged Dataframe: '+str(merged_df.columns))

	# print(f'Target value positive percentage: {target.sum()/len(target)}')
	# # merged_df.drop(columns=[f'rolling_{target_var}', 'classification'], inplace=True)

	temp_version = 'pytorch_test'

	if os.path.exists(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl'):
		with open(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl', 'rb') as f:
			storms_extracted_dict = pickle.load(f)
		storms = storms_extracted_dict['storms']
		target = storms_extracted_dict['target']

	else:
		# getting the data corresponding to the twins maps
		storms, target = utils.storm_extract(df=merged_df, lead=30, recovery=9, twins=True, target=True, target_var=f'rolling_{target_var}', concat=False, map_keys=maps.keys())
		storms_extracted_dict = {'storms':storms, 'target':target}
		with open(working_dir+f'twins_method_storm_extraction_map_keys_region_{region}_time_history_{CONFIG["time_history"]}_version_{temp_version}.pkl', 'wb') as f:
			pickle.dump(storms_extracted_dict, f)

	print('Columns in Dataframe: '+str(storms[0].columns))
	features = storms[0].columns

	# splitting the data on a month to month basis to reduce data leakage
	month_df = pd.date_range(start=pd.to_datetime('2009-07-01'), end=pd.to_datetime('2017-12-01'), freq='MS')
	month_df = month_df.drop([pd.to_datetime('2012-03-01'), pd.to_datetime('2017-09-01')])

	train_months, test_months = train_test_split(month_df, test_size=0.1, shuffle=True, random_state=CONFIG['random_seed'])
	train_months, val_months = train_test_split(train_months, test_size=0.125, shuffle=True, random_state=CONFIG['random_seed'])

	test_months = test_months.tolist()
	# adding the two dateimte values of interest to the test months df
	test_months.append(pd.to_datetime('2012-03-01'))
	test_months.append(pd.to_datetime('2017-09-01'))
	test_months = pd.to_datetime(test_months)

	train_dates_df, val_dates_df, test_dates_df = pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]})
	x_train, x_val, x_test, y_train, y_val, y_test, twins_train, twins_val, twins_test = [], [], [], [], [], [], [], [], []

	# using the months to split the data
	for month in train_months:
		train_dates_df = pd.concat([train_dates_df, pd.DataFrame({'dates':pd.date_range(start=month, end=month+pd.DateOffset(months=1), freq='min')})], axis=0)

	for month in val_months:
		val_dates_df = pd.concat([val_dates_df, pd.DataFrame({'dates':pd.date_range(start=month, end=month+pd.DateOffset(months=1), freq='min')})], axis=0)

	for month in test_months:
		test_dates_df = pd.concat([test_dates_df, pd.DataFrame({'dates':pd.date_range(start=month, end=month+pd.DateOffset(months=1), freq='min')})], axis=0)

	train_dates_df.set_index('dates', inplace=True)
	val_dates_df.set_index('dates', inplace=True)
	test_dates_df.set_index('dates', inplace=True)

	train_dates_df.index = pd.to_datetime(train_dates_df.index)
	val_dates_df.index = pd.to_datetime(val_dates_df.index)
	test_dates_df.index = pd.to_datetime(test_dates_df.index)

	date_dict = {'train':pd.DataFrame(), 'val':pd.DataFrame(), 'test':pd.DataFrame()}

	print(f'Size of the training storms: {len(storms)}')
	print(f'Size of the training target: {len(target)}')
	print(f'Size of the twins maps: {len(maps)}')

	# getting the data corresponding to the dates
	for storm, y, twins_map in zip(storms, target, maps):

		copied_storm = storm.copy()
		copied_storm = copied_storm.reset_index(inplace=False, drop=False).rename(columns={'index':'Date_UTC'})

		if storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in train_dates_df.index:
			x_train.append(storm)
			y_train.append(y)
			twins_train.append(maps[twins_map]['map'])
			date_dict['train'] = pd.concat([date_dict['train'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in val_dates_df.index:
			x_val.append(storm)
			y_val.append(y)
			twins_val.append(maps[twins_map]['map'])
			date_dict['val'] = pd.concat([date_dict['val'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in test_dates_df.index:
			x_test.append(storm)
			y_test.append(y)
			twins_test.append(maps[twins_map]['map'])
			date_dict['test'] = pd.concat([date_dict['test'], copied_storm['Date_UTC'][-10:]], axis=0)

	# scaling the twins maps
	# twins_scaler = MinMaxScaler()
	# twins_scaler.fit(twins_scaling_array)
	# twins_train = [twins_scaler.transform(x) for x in twins_train]
	# twins_val = [twins_scaler.transform(x) for x in twins_val]
	# twins_test = [twins_scaler.transform(x) for x in twins_test]


	def standard_scaling(x):
		return (x - scaling_mean) / scaling_std

	def minmax_scaling(x):
		return (x - scaling_min) / (scaling_max - scaling_min)

	def keV_to_eV(x):
		# changing positive values in the array to eV
		return x *1000

	print(f'Twins train mean before converting to eV: {np.array(twins_train).mean()}')
	print(f'Twins train std before converting to eV: {np.array(twins_train).std()}')

	# twins_train = [keV_to_eV(x) for x in twins_train]
	# twins_val = [keV_to_eV(x) for x in twins_val]
	# twins_test = [keV_to_eV(x) for x in twins_test]

	# print(f'Twins train mean after converting to eV: {np.array(twins_train).mean()}')
	# print(f'Twins train std after converting to eV: {np.array(twins_train).std()}')
	# print(f'Twins train min after converting to eV: {np.array(twins_train).min()}')

	twins_scaling_array = np.vstack(twins_train).flatten()

	twins_scaling_array = twins_scaling_array[twins_scaling_array > 0]
	scaling_mean = twins_scaling_array.mean()
	scaling_std = twins_scaling_array.std()
	scaling_min = twins_scaling_array.min()
	scaling_max = twins_scaling_array.max()

	twins_train = [standard_scaling(x) for x in twins_train]
	twins_val = [standard_scaling(x) for x in twins_val]
	twins_test = [standard_scaling(x) for x in twins_test]

	# twins_train = [minmax_scaling(x) for x in twins_train]
	# twins_val = [minmax_scaling(x) for x in twins_val]
	# twins_test = [minmax_scaling(x) for x in twins_test]

	print(f'Twins train mean after standard scaling: {np.array(twins_train).mean()}')
	print(f'Twins train std after standard scaling: {np.array(twins_train).std()}')
	print(f'Twins train min after standard scaling: {np.array(twins_train).min()}')

	if not get_features:
		return torch.tensor(twins_train), torch.tensor(twins_val), torch.tensor(twins_test), date_dict
	else:
		return torch.tensor(twins_train), torch.tensor(twins_val), torch.tensor(twins_test), date_dict, features



class PerceptualLoss(nn.Module):
	'''loss function using the residuals of a pretrained model to calculate the
			loss between of the feature maps between the predicted and real images'''

	def __init__(self, conv_index: str = '22'):

		super(PerceptualLoss, self).__init__()
		self.conv_index = conv_index
		vgg_features = torchvision.models.vgg19(pretrained=True).features
		modules = [m for m in vgg_features]

		if self.conv_index == '22':
			self.vgg = nn.Sequential(*modules[:8]).to(DEVICE)
		elif self.conv_index == '54':
			self.vgg = nn.Sequential(*modules[:35]).to(DEVICE)

		self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

		self.vgg.requires_grad = False


	def forward(self, output, target):

		# copying the output and ytest such that they go from 1 channel to 3 channels
		output = output.repeat(1, 3, 1, 1)
		target = target.repeat(1, 3, 1, 1)

		self.mean = self.mean.to(DEVICE)
		self.std = self.std.to(DEVICE)
		# output = output.to(DEVICE)
		# target = target.to(DEVICE)

		output = (output-self.mean) / self.std
		target = (target-self.mean) / self.std

		# getting the feature maps from the vgg model
		output_features = self.vgg(output)
		target_features = self.vgg(target)

		# calculating the loss
		loss = F.mse_loss(output_features, target_features)

		return loss


class VGGPerceptualLoss(torch.nn.Module):
	def __init__(self, resize=True):
		super(VGGPerceptualLoss, self).__init__()
		blocks = []
		blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
		for bl in blocks:
			for p in bl.parameters():
				p.requires_grad = False
		self.blocks = torch.nn.ModuleList(blocks)
		self.transform = torch.nn.functional.interpolate
		self.resize = resize
		self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

	def forward(self, output, target, feature_layers=[0, 1, 2, 3], style_layers=[]):

		self.mean = self.mean.to(DEVICE)
		self.std = self.std.to(DEVICE)
		output = output.to(DEVICE)
		target = target.to(DEVICE)

		if output.shape[1] != 3:
			output = output.repeat(1, 3, 1, 1)
			target = target.repeat(1, 3, 1, 1)

		output = (output-self.mean) / self.std
		target = (target-self.mean) / self.std
		if self.resize:
			output = self.transform(output, mode='bilinear', size=(224, 224), align_corners=False)
			target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
		loss = 0.0
		for i, block in enumerate(self.blocks):
			block.to(DEVICE)
			output = block(output)
			target = block(target)
			# output = output.detach()
			# target = target.detach()
			if i in feature_layers:
				loss += F.mse_loss(output, target)
		return loss


class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding='same'),
			# nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
			# nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Dropout(0.2),
			# nn.MaxPool2d(kernel_size=2, stride=2),
			# nn.BatchNorm2d(256),
			# nn.ReLU(),
			# nn.Dropout(0.2),
			# nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
			# # nn.BatchNorm2d(256),
			# nn.ReLU(),
			# nn.Dropout(0.2),
			nn.Flatten(),
			nn.Linear(128*90*60, 120)
		)
		self.decoder = nn.Sequential(
			nn.Linear(120, 128*90*60),
			nn.Unflatten(1, (128, 90, 60)),
			# nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
			# # nn.BatchNorm2d(128),
			# nn.ReLU(),
			# nn.Dropout(0.2),
			# nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=3, padding=0),
			# # nn.BatchNorm2d(128),
			# nn.ReLU(),
			# nn.Dropout(0.2),
			# nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
			# nn.BatchNorm2d(128),
			# nn.ReLU(),
			# nn.Dropout(0.2),
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
			# nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
		)

	def forward(self, x, get_latent=False):
		# x = x.unsqueeze(1)
		latent = self.encoder(x)
		if get_latent:
			return latent
		else:
			x = self.decoder(latent)
			return x


class Early_Stopping():
	'''
	Class to create an early stopping condition for the model.

	'''

	def __init__(self, decreasing_loss_patience=25, training_diff_patience=3):
		self.decreasing_loss_patience = decreasing_loss_patience
		self.training_diff_patience = training_diff_patience
		self.loss_counter = 0
		self.training_counter = 0
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
		elif val_loss > (1.5 * train_loss):
			self.training_counter += 1
			if self.training_counter >= self.training_diff_patience:
				print(f'Engaging Early Stopping due to large seperation between train and val loss. Best model saved at epoch {self.best_epoch} with a training loss of {self.best_loss} and a validation loss of {self.best_score}')
				return True

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
			torch.save(self.model.state_dict(), f'models/autoencoder_{VERSION}.pt')


def fit_autoencoder(model, train, val, val_loss_patience=25, overfit_patience=5, num_epochs=500):

	if not os.path.exists(f'models/autoencoder_{VERSION}.pt'):

		train_loss_list, val_loss_list = [], []

		# moving the model to the available device
		model.to(DEVICE)

		# defining the loss function and the optimizer
		# criterion = nn.MSELoss()
		# criterion = PerceptualLoss()
		# criterion = PerceptualLoss(conv_index='22')

		criterion = nn.MSELoss()
		optimizer = optim.Adam(model.parameters(), lr=1e-4)
		scaler = torch.cuda.amp.GradScaler()

		# initalizing the early stopping class
		early_stopping = Early_Stopping(decreasing_loss_patience=val_loss_patience, training_diff_patience=overfit_patience)

		for epoch in range(num_epochs):

			# starting the clock for the epoch
			stime = time.time()

			# setting the model to training mode
			model.train()

			# initializing the running loss
			running_training_loss, running_val_loss = 0.0, 0.0

			# shuffling data and creating batches
			for train_data in train:
				train_data = train_data.to(DEVICE, dtype=torch.float)
				train_data = train_data.unsqueeze(1)
				# forward pass
				with torch.cuda.amp.autocast():
					output = model(train_data)

					loss = criterion(output, train_data)
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
				for val_data in val:
					val_data = val_data.to(DEVICE, dtype=torch.float)
					val_data = val_data.unsqueeze(1)
					output = model(val_data)

					val_loss = criterion(output, val_data)

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
		model.load_state_dict(torch.load(f'models/autoencoder_{VERSION}.pt'))

		# transforming the lists to a dataframe to be saved
		loss_tracker = pd.DataFrame({'train_loss':train_loss_list, 'val_loss':val_loss_list})
		loss_tracker.to_feather(f'outputs/autoencoder_{VERSION}_loss_tracker.feather')

	else:
		# loading the model if it has already been trained.
		model.load_state_dict(torch.load(f'models/autoencoder_{VERSION}.pt')) 			# loading the models if already trained

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
	predicted_list, test_list = [], []
	# setting the encoder and decoder into evaluation model
	model.eval()

	# creating a loss value
	running_loss = 0.0

	# making sure the model is on the correct device
	model.to(DEVICE, dtype=torch.float)

	with torch.no_grad():
		for test_data in test:
			test_data = test_data.to(DEVICE, dtype=torch.float)
			test_data = test_data.unsqueeze(1)
			predicted = model(test_data)
			loss = F.mse_loss(predicted, test_data)
			running_loss += loss.item()

			# making sure the predicted value is on the cpu
			if predicted.get_device() != -1:
				predicted = predicted.to('cpu')
			if test_data.get_device() != -1:
				test_data = test_data.to('cpu')

			# adding the decoded result to the predicted list after removing the channel dimension
			predicted = torch.squeeze(predicted, dim=1).numpy()
			test_data = torch.squeeze(test_data, dim=1).numpy()
			predicted_list.append(predicted)
			test_list.append(test_data)

	return np.concatenate(predicted_list, axis=0), np.concatenate(test_list, axis=0), running_loss/len(test)



def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	train, val, test, ___ = getting_prepared_data(target_var=TARGET, region=REGION)

	# # converting the data to a tensor dataset
	# train = TensorDataset(train, train)
	# val = TensorDataset(val, val)
	# test = TensorDataset(test, test)

	# creating the dataloaders
	train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
	val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
	test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

	# creating the model
	print('Initalizing model....')
	autoencoder = Autoencoder()

	# fitting the model
	print('Fitting model....')
	autoencoder = fit_autoencoder(autoencoder, train, val, val_loss_patience=25, overfit_patience=5, num_epochs=500)

	# evaluating the model
	print('Evaluating model....')
	predictions, test, testing_loss = evaluation(autoencoder, test)

	# turing the test data back into a numpy array
	# test = np.concatenate([arr for arr in test], axis=0)

	vmin = min([predictions[0, :, :].min(), test[0, :, :].min()])
	vmax = max([predictions[0, :, :].max(), test[0, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[0, :, :], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[0, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	plt.show()

	vmin = min([predictions[324, :, :].min(), test[324, :, :].min()])
	vmax = max([predictions[324, :, :].max(), test[324, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[324, :, :], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[324, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	plt.show()

	vmin = min([predictions[256, :, :].min(), test[256, :, :].min()])
	vmax = max([predictions[256, :, :].max(), test[256, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[256, :, :], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[256, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	plt.show()

	vmin = min([predictions[1000, :, :].min(), test[1000, :, :].min()])
	vmax = max([predictions[1000, :, :].max(), test[1000, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[1000, :, :], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[1000, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	plt.show()


	print(f'Loss: {testing_loss}')


if __name__ == '__main__':
	main()
	print('It ran. God job!')
