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
import logging
import os
import pickle
import subprocess
import time
from typing import Optional

import kornia.enhance as ke
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
import torch.nn._reduction as _Reduction
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from matplotlib import patches
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchsummary import summary
from torchvision import models
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)
from torchvision.transforms.functional import rotate

import utils

# defining the version for saving the models
VERSION = 'pytorch_perceptual_v1-42'

# setting the seed for reproducibility, setting the CDF lib path
# and working dir path, and setting the device.

os.environ["CDF_LIB"] = "~/CDF/lib"
working_dir = '../../../../data/mike_working_dir/'
RANDOM_SEED = 7
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


def creating_fake_twins_data(train, scaling_mean, scaling_std):
	'''
	Function to create fake TWINS data for pretraining the model. Includes real
	TWINS data that has been rotated, mirrored, and duplicated within the same image.

	Args:
		train (torch.tensor): the training data for the model
		scaling_mean (float): the mean of the training data used for scaling
		scaling_std (float): the standard deviation of the training data used for scaling

	Returns:
		torch.tensor: the scaled training data for the model
		torch.tensor: the scaled validation data for the model
		torch.tensor: the scaled testing data for the model
	'''

	# splitting the training data into train val and test
	train_data, test_data = train_test_split(train, test_size=0.1, random_state=RANDOM_SEED)
	train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=RANDOM_SEED)


	'''rotating the data 90 degrees with expansion then cutting them to
		the 90,60 size. Results will have "duplicated" parts of the maps.'''
	train_rotated_90 = rotate(train_data, angle=90, expand=True)
	val_rotated_90 = rotate(val_data, angle=90, expand=True)
	test_rotated_90 = rotate(test_data, angle=90, expand=True)

	train_rotated_90 = train_rotated_90[:, 0:90, 0:60]
	val_rotated_90 = val_rotated_90[:, 0:90, 0:60]
	test_rotated_90 = test_rotated_90[:, 0:90, 0:60]

	# adding a 30, 60 slice to the data at the 1 dimension to make it a 90, 60 image
	train_rotated_90 = torch.cat([train_rotated_90, train_rotated_90[:, 0:30, 0:60]], dim=1)
	val_rotated_90 = torch.cat([val_rotated_90, val_rotated_90[:, 0:30, 0:60]], dim=1)
	test_rotated_90 = torch.cat([test_rotated_90, test_rotated_90[:, 0:30, 0:60]], dim=1)


	# rotating the data 180 degrees (much simpler process)
	train_rotated_180 = rotate(train_data, angle=180, expand=False)
	val_rotated_180 = rotate(val_data, angle=180, expand=False)
	test_rotated_180 = rotate(test_data, angle=180, expand=False)


	# same procedure as the 90 degree rotation for a 270 rotation
	train_rotated_270 = rotate(train_data, angle=270, expand=True)
	val_rotated_270 = rotate(val_data, angle=270, expand=True)
	test_rotated_270 = rotate(test_data, angle=270, expand=True)

	train_rotated_270 = train_rotated_270[:, 0:90, 0:60]
	val_rotated_270 = val_rotated_270[:, 0:90, 0:60]
	test_rotated_270 = test_rotated_270[:, 0:90, 0:60]

	# adding a 30, 60 slice to the data at the 1 dimension
	train_rotated_270 = torch.cat([train_rotated_270, train_rotated_270[:, 0:30, 0:60]], dim=1)
	val_rotated_270 = torch.cat([val_rotated_270, val_rotated_270[:, 0:30, 0:60]], dim=1)
	test_rotated_270 = torch.cat([test_rotated_270, test_rotated_270[:, 0:30, 0:60]], dim=1)


	# flipping the data along the 1 axis
	flipped_train_data = torch.flip(train_data, [1])
	flipped_val_data = torch.flip(val_data, [1])
	flipped_test_data = torch.flip(test_data, [1])

	# flipping the data along the 2 axis
	flipped_train_data_2 = torch.flip(train_data, [2])
	flipped_val_data_2 = torch.flip(val_data, [2])
	flipped_test_data_2 = torch.flip(test_data, [2])


	# concatenating the data without the origonal data
	train_data = torch.cat([train_rotated_90, train_rotated_180, train_rotated_270, flipped_train_data, flipped_train_data_2], dim=0)
	val_data = torch.cat([val_rotated_90, val_rotated_180, val_rotated_270, flipped_val_data, flipped_val_data_2], dim=0)
	test_data = torch.cat([test_rotated_90, test_rotated_180, test_rotated_270, flipped_test_data, flipped_test_data_2], dim=0)

	print(f'Train max: {train_data.max()}, Train min: {train_data.min()}')
	print(f'Val max: {val_data.max()}, Val min: {val_data.min()}')
	print(f'Test max: {test_data.max()}, Test min: {test_data.min()}')

	return train_data, val_data, test_data


def standard_scaling(x, scaling_mean, scaling_std):
	# scaling the data to have a mean of 0 and a standard deviation of 1
	return (x - scaling_mean) / scaling_std


def getting_prepared_data():
	'''
	Function to get the prepared data for the model.

	Args:
		get_features (bool): whether to return the features of the data

	Returns:
		twins_train (torch.tensor): the training data for the autoencoder
		twins_val (torch.tensor): the validation data for the autoencoder
		twins_test (torch.tensor): the testing data for the autoencoder
		date_dict (dict): the dates of the data
		scaling_mean (float): the mean of the training data
		scaling_std (float): the standard deviation of the training data

	'''

	# loading the TWINS maps
	maps = utils.loading_twins_maps()

	# changing all negative values in maps to 0
	for key in maps.keys():
		maps[key]['map'][maps[key]['map'] < 0] = 0

	# loading pre-prepared data
	with open(working_dir+f'twins_method_storm_extraction_map_keys_version_pytorch_test.pkl', 'rb') as f:
		storms_extracted_dict = pickle.load(f)

	# getting the storm labels from teh saved data
	storms = storms_extracted_dict['storms']

	# splitting the data on a day to day basis to reduce data leakage
	day_df = pd.date_range(start=pd.to_datetime('2009-07-01'), end=pd.to_datetime('2017-12-01'), freq='D')

	# making sure a specific test day is in the test set then dropping it from the list to be split
	specific_test_days = pd.date_range(start=pd.to_datetime('2012-03-07'), end=pd.to_datetime('2012-03-13'), freq='D')
	day_df = day_df.drop(specific_test_days)

	# splitting the data into training, validation, and testing sets
	train_days, test_days = train_test_split(day_df, test_size=0.1, shuffle=True, random_state=RANDOM_SEED)
	train_days, val_days = train_test_split(train_days, test_size=0.125, shuffle=True, random_state=RANDOM_SEED)

	# adding the test days of interest to the test set
	test_days = test_days.tolist()
	test_days = pd.to_datetime(test_days)
	test_days.append(specific_test_days)

	# creating dataframes to hold the dates
	train_dates_df, val_dates_df, test_dates_df = pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]}), pd.DataFrame({'dates':[]})

	# getting the minute resolution dates for the days to get all maps with those timestamps in those time frames
	for day in train_days:
		train_dates_df = pd.concat([train_dates_df, pd.DataFrame({'dates':pd.date_range(start=day, end=day+pd.DateOffset(days=1), freq='min')})], axis=0)

	for day in val_days:
		val_dates_df = pd.concat([val_dates_df, pd.DataFrame({'dates':pd.date_range(start=day, end=day+pd.DateOffset(days=1), freq='min')})], axis=0)

	for day in test_days:
		test_dates_df = pd.concat([test_dates_df, pd.DataFrame({'dates':pd.date_range(start=day, end=day+pd.DateOffset(days=1), freq='min')})], axis=0)

	# resetting the index of the dataframes
	train_dates_df.set_index('dates', inplace=True)
	val_dates_df.set_index('dates', inplace=True)
	test_dates_df.set_index('dates', inplace=True)

	# setting the index to the datetimes from the date dfs
	train_dates_df.index = pd.to_datetime(train_dates_df.index)
	val_dates_df.index = pd.to_datetime(val_dates_df.index)
	test_dates_df.index = pd.to_datetime(test_dates_df.index)

	# initalizing the dfs to hold the twins data
	date_dict = {'train':pd.DataFrame(), 'val':pd.DataFrame(), 'test':pd.DataFrame()}

	# just printing the size of the data to make sure they're the same
	print(f'Size of the training storms: {len(storms)}')
	print(f'Size of the twins maps: {len(maps)}')

	# initalizing the lists to hold the twins data
	twins_train, twins_val, twins_test = [], [], []

	# getting the data corresponding to the dates
	for storm, twins_map in zip(storms, maps):

		# creating a copy of the storm data
		copied_storm = storm.copy()
		copied_storm = copied_storm.reset_index(inplace=False, drop=False).rename(columns={'index':'Date_UTC'})

		# getting the twins map that corresponds to the storm
		if storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in train_dates_df.index:
			twins_train.append(maps[twins_map]['map'])
			date_dict['train'] = pd.concat([date_dict['train'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in val_dates_df.index:
			twins_val.append(maps[twins_map]['map'])
			date_dict['val'] = pd.concat([date_dict['val'], copied_storm['Date_UTC'][-10:]], axis=0)
		elif storm.index[0].strftime('%Y-%m-%d %H:%M:%S') in test_dates_df.index:
			twins_test.append(maps[twins_map]['map'])
			date_dict['test'] = pd.concat([date_dict['test'], copied_storm['Date_UTC'][-10:]], axis=0)

	# getting the scaling values for the data
	twins_scaling_array = np.vstack(twins_train).flatten()

	# print(f'Train max: {np.max(twins_scaling_array)}, Train min: {np.min(twins_scaling_array)}')


	# # removing any negative values from the data
	# twins_scaling_array = twins_scaling_array[twins_scaling_array > 0]

	# getting the mean and standard deviation of the data for scaling
	scaling_mean = twins_scaling_array.mean()
	scaling_std = twins_scaling_array.std()

	print(f'Train max: {np.max(twins_scaling_array)}, Train min: {np.min(twins_scaling_array)}')

	# scaling the data
	twins_train = [standard_scaling(x, scaling_mean, scaling_std) for x in twins_train]
	twins_val = [standard_scaling(x, scaling_mean, scaling_std) for x in twins_val]
	twins_test = [standard_scaling(x, scaling_mean, scaling_std) for x in twins_test]

	print('After scaling')
	print(f'Train max: {np.max(twins_train)}, Train min: {np.min(twins_train)}')
	print(f'Val max: {np.max(twins_val)}, Val min: {np.min(twins_val)}')
	print(f'Test max: {np.max(twins_test)}, Test min: {np.min(twins_test)}')

	return torch.tensor(twins_train), torch.tensor(twins_val), torch.tensor(twins_test), date_dict, scaling_mean, scaling_std


class PerceptualLoss(nn.Module):
	'''loss function using the residuals of a pretrained model to calculate the
			loss between of the feature maps between the predicted and real images'''

	def __init__(self, conv_index: str = '22'):
		'''
		Initializing the PerceptualLoss class which uses the VGG19 pretrained model. This loss
		function compares the feature maps of the predicted and real images and computes the
		MSE between those instead of the more traditional MSE on the final arrays.

		Args:
			conv_index (str): the index of the convolutional layer to be used to calculate the loss

		'''
		# inheriting the functionality of the nn.Module class
		super(PerceptualLoss, self).__init__()

		self.conv_index = conv_index

		# getting the VGG19 model
		vgg_features = torchvision.models.vgg19(pretrained=True).features

		# getting the modules of the VGG model
		modules = [m for m in vgg_features]

		# getting the specific layers to be used for the loss
		if self.conv_index == '22':
			self.vgg = nn.Sequential(*modules[:8]).to(DEVICE)
		elif self.conv_index == '54':
			self.vgg = nn.Sequential(*modules[:35]).to(DEVICE)

		# setting the mean and standard deviation of the data
		self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

		# setting the model to not require gradients
		self.vgg.requires_grad = False


	def forward(self, output, target):
		'''
		Function to calculate the perceptual loss between the output and target images.

		Args:
			output (torch.tensor): the predicted image
			target (torch.tensor): the real image

		Returns:
			float: the loss between the two images
		'''

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

		# calculating the MSE using the functional version of the loss
		loss = F.mse_loss(output_features, target_features)

		return loss


class VGGPerceptualLoss(nn.Module):
	def __init__(self, resize=True):
		'''
		Initializing the VGGPerceptualLoss class. This version uses the VGG16 model.

		Args:
			resize (bool): whether to resize the images to 224x224

		'''

		# inheriting the functionality of the nn.Module class
		super(VGGPerceptualLoss, self).__init__()

		# getting the VGG16 model and setting the blocks to not require gradients
		blocks = []
		blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
		for bl in blocks:
			for p in bl.parameters():
				p.requires_grad = False

		# initalizing the variables and getting the mean and standard deviation of the data
		self.blocks = torch.nn.ModuleList(blocks)
		self.transform = torch.nn.functional.interpolate
		self.resize = resize
		self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


	def forward(self, output, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
		'''
		Function to calculate the perceptual loss between the output and target images.

		Args:
			output (torch.tensor): the predicted image
			target (torch.tensor): the real image
			feature_layers (list): the layers to be used to calculate the loss

		Returns:
			float: the loss between the two images
		'''

		# sending the data to the device
		self.mean = self.mean.to(DEVICE)
		self.std = self.std.to(DEVICE)
		output = output.to(DEVICE)
		target = target.to(DEVICE)

		# copying the output and ytest such that they go from 1 channel to 3 channels if they are not already
		if output.shape[1] != 3:
			output = output.repeat(1, 3, 1, 1)
			target = target.repeat(1, 3, 1, 1)

		# normalizing the data
		output = (output-self.mean) / self.std
		target = (target-self.mean) / self.std

		# resizing the data if needed
		if self.resize:
			output = self.transform(output, mode='bilinear', size=(224, 224), align_corners=False)
			target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

		# initalizing the loss
		loss = 0.0

		# getting the feature maps from the VGG model
		for i, block in enumerate(self.blocks):
			block.to(DEVICE)
			output = block(output)
			target = block(target)

			# calculating the MSE for the feature maps using the functional form of the loss
			if i in feature_layers:
				loss += F.mse_loss(output, target)

		return loss


class Autoencoder(nn.Module):
	def __init__(self):
		'''
		Initializing the autoencoder model. Defining the layers of the encoder and decoder.

		'''
		# inheriting the functionality of the nn.Module class
		super(Autoencoder, self).__init__()

		# defining the layers of the encoder
		self.encoder = nn.Sequential(

			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding='same'),
			nn.ReLU(),
			nn.Dropout(0.2),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
			nn.ReLU(),
			nn.Dropout(0.2),

			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
			nn.ReLU(),
			nn.Dropout(0.2),


			# flattening the outputs of the last conv layer to go through a linear latent space
			nn.Flatten(),
			nn.Linear(256*45*30, 420),
		)

		# defining the layers of the decoder. Found using linear activation worked best here
		self.decoder = nn.Sequential(

			nn.Linear(420, 256*45*30),

			# reshaping the data to go through the transposed conv layers
			nn.Unflatten(1, (256, 45, 30)),

			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.Dropout(0.2),

			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
			nn.Dropout(0.2),

			nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=1),
			nn.Dropout(0.2),

			nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=1, padding=0),
		)

	def forward(self, x, get_latent=False):
		'''
		Function to pass the input through the model.

		Args:
			x (torch.tensor): the input data
			get_latent (bool): whether to return the latent space representation of the data

		Returns:
			torch.tensor: the output of the model
		'''

		# passing the data through the encoder
		latent = self.encoder(x)

		# if the latent space is requested, return it
		# if get_latent:
		# 	return latent

		# passing the latent space through the decoder
		# else:
		x = self.decoder(latent)

		return x


class Early_Stopping():
	'''
	Class to create an early stopping condition for the model.

	'''

	def __init__(self, decreasing_loss_patience=25, pretraining=False):
		'''
		Initializing the class.

		Args:
			decreasing_loss_patience (int): the number of epochs to wait before stopping the model if the validation loss does not decrease
			pretraining (bool): whether the model is being pre-trained. Just used for saving model names.

		'''

		# initializing the variables
		self.decreasing_loss_patience = decreasing_loss_patience
		self.loss_counter = 0
		self.training_counter = 0
		self.best_score = None
		self.early_stop = False
		self.best_epoch = None
		self.pretraining = pretraining


	def __call__(self, train_loss, val_loss, model, optimizer, epoch, pretraining=False):
		'''
		Function to call the early stopping condition.

		Args:
			train_loss (float): the training loss for the model
			val_loss (float): the validation loss for the model
			model (object): the model to be saved
			epoch (int): the current epoch

		Returns:
			bool: whether the model should stop training or not
		'''

		# using the absolute value of the loss for negatively orientied loss functions
		val_loss = abs(val_loss)

		# initializing the best score if it is not already
		self.model = model
		self.optimizer = optimizer
		if self.best_score is None:
			self.best_train_loss = train_loss
			self.best_score = val_loss
			self.best_loss = val_loss
			self.save_checkpoint(val_loss)
			self.best_epoch = epoch

		# if the validation loss greater than the best score add one to the loss counter
		elif val_loss > self.best_score:
			self.loss_counter += 1

			# if the loss counter is greater than the patience, stop the model training
			if self.loss_counter >= self.decreasing_loss_patience:
				gc.collect()
				print(f'Engaging Early Stopping due to lack of improvement in validation loss. Best model saved at epoch {self.best_epoch} with a training loss of {self.best_train_loss} and a validation loss of {self.best_score}')
				return True

		# if the validation loss is less than the best score, reset the loss counter and use the new validation loss as the best score
		else:
			self.best_train_loss = train_loss
			self.best_score = val_loss
			self.best_epoch = epoch

			# saving the best model as a checkpoint
			self.save_checkpoint(val_loss)
			self.loss_counter = 0
			self.training_counter = 0

			return False

	def save_checkpoint(self, val_loss):
		'''
		Function to continually save the best model.

		Args:
			val_loss (float): the validation loss for the model
		'''

		# saving the model if the validation loss is less than the best loss
		if self.best_loss > val_loss:
			self.best_loss = val_loss
			print('Saving checkpoint!')

			# saving the model dictonary
			if self.pretraining:
				torch.save({'model':self.model.state_dict(),
							'optimizer': self.optimizer.state_dict(),
							'best_epoch':self.best_epoch,
							'finished_training':False},
							f'models/autoencoder_pretraining_{VERSION}.pt')
			else:
				torch.save({'model': self.model.state_dict(),
							'optimizer':self.optimizer.state_dict(),
							'best_epoch':self.best_epoch,
							'finished_training':False},
							f'models/autoencoder_{VERSION}.pt')


def resume_training(model, optimizer, pretraining=False):
	'''
	Function to resume training of a model if it was interupted without completeing.

	Args:
		model (object): the model to be trained
		optimizer (object): the optimizer to be used
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the model to be trained
		object: the optimizer to be used
		int: the epoch to resume training from
	'''

	if pretraining:
		# try-except block to load the model and optimizer depending on how it was saved
		try:
			checkpoint = torch.load(f'models/autoencoder_pretraining_{VERSION}.pt')
			model.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			epoch = checkpoint['best_epoch']
			finished_training = checkpoint['finished_training']

		# will produce a key error if the model is an earlier version
		except KeyError:
			model.load_state_dict(torch.load(f'models/autoencoder_pretraining_{VERSION}.pt'))
			optimizer=None
			epoch = 0
			finished_training = True

	else:
		try:
			checkpoint = torch.load(f'models/autoencoder_{VERSION}.pt')
			model.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			epoch = checkpoint['best_epoch']
			finished_training = checkpoint['finished_training']
		except KeyError:
			model.load_state_dict(torch.load(f'models/autoencoder_{VERSION}.pt'))
			optimizer=None
			epoch = 0
			finished_training = True

	return model, optimizer, epoch, finished_training


class JSD(nn.Module):
	'''
	Class to calculate the Jensen-Shannon Divergence between two images. First
		the maximum and minimum of the inputs is found. Then the images are binned
		into histograms using the same bins defined by the max and min values to keep
		the bins consistent between the y and y_hat arrays. The histograms use a smoothing
		function to normalize the values for each bin to probabilities that sum up to 1 which
		is prefered for the calculation of the divergence. The values are cpilled to a
		minimum of 1e-45 to avoid -inf values when taking the log. The m parameter is then calculated
		and used as the "input" for the KLDivLoss function. The "target" is the y (P(x))
		and y_hat (Q(x)) for the two kl_divergence calculations. The final loss is the average of the two.
		For more information on the Jensen-Shannon Divergence see: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

	Args:
		nn.Module: the base class for all neural network
					modules. Inherits the backpropagation functionality
	'''


	def __init__(self):
		'''
		Initializing the JSD class.

		'''

		# using super to inherit the functionality of the nn.Module class
		super(JSD, self).__init__()

		# defining the KLDivLoss function to be used to calculate the loss
		self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)


	def matching_scales(self, y_hat: torch.tensor, y: torch.tensor):
		'''
		Function to match the scales of the y and y_hat arrays and bin them into histograms.

		Args:
			y_hat (torch.tensor): the predicted image
			y (torch.tensor): the real image

		Returns:
			torch.tensor: the binned histogram of the predicted image
			torch.tensor: the binned histogram of the real image
		'''

		# finding the min and max values of the data
		self.scale_min = torch.min(torch.min(y), torch.min(y_hat))
		self.scale_max = torch.max(torch.max(y), torch.max(y_hat))

		# creating the bins for the histogram such that they are the same values for each array
		bins = torch.torch.linspace(self.scale_min, self.scale_max, 100).to(DEVICE)

		# reshaping the data to be 1D
		y = torch.reshape(y, (1,y.size(0)))
		y_hat = torch.reshape(y_hat, (1,y_hat.size(0)))

		# binning into probability normalized histograms
		div_y = ke.histogram(y, bins=bins, bandwidth=torch.tensor(0.1))
		div_y_hat = ke.histogram(y_hat, bins=bins, bandwidth=torch.tensor(0.1))

		return div_y_hat, div_y


	def forward(self, q: torch.tensor, p: torch.tensor):
		'''
		Function to calculate the Jensen-Shannon Divergence between the predicted and real images.

		Args:
			q (torch.tensor): the predicted image
			p (torch.tensor): the real image

		Returns:
			float: the JSDiv loss between the two images
		'''

		# calculating the mean squared error between the two images for later use
		mse = F.mse_loss(q, p)

		# matching the scales and binning the data
		q, p = self.matching_scales(q, p)

		# calculating the m parameter
		m = (0.5 * (p + q))

		# clipping to avoid -inf in the log
		q, p, m = torch.clamp(q, min=1e-45), torch.clamp(p, min=1e-45), torch.clamp(m, min=1e-45)

		# taking the log of the data
		p, q, m = torch.log(p), torch.log(q), torch.log(m)

		# calculating and returning the loss
		return torch.mul(mse, torch.add(1, (0.5 * (self.kl(m, p) + self.kl(m, q)))))


def fitting_the_model(model, data, train_or_val=None):

	''' will include this function to simplify the training process
	'''

def fit_autoencoder(model, train, val, val_loss_patience=25, overfit_patience=5, num_epochs=500, pretraining=False):

	'''
	_summary_: Function to train the autoencoder model.

	Args:
		model (object): the model to be trained
		train (torch.utils.data.DataLoader): the training data
		val (torch.utils.data.DataLoader): the validation data
		val_loss_patience (int): the number of epochs to wait before stopping the model
									if the validation loss does not decrease
		overfit_patience (int): the number of epochs to wait before stopping the model
									if the training loss is significantly lower than the
									validation loss
		num_epochs (int): the number of epochs to train the model
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the trained model
	'''

	# initializing the Adam optimizer using the model params and setting the initial learning rate
	optimizer = optim.Adam(model.parameters(), lr=1e-5)

	# checking if the model has already been trained, loading it if it exists
	if pretraining:
		if os.path.exists(f'models/autoencoder_pretraining_{VERSION}.pt'):
			model, optimizer, current_epoch, finished_training = resume_training(model=model, optimizer=optimizer, pretraining=pretraining)
		else:
			finished_training = False
			current_epoch = 0
	else:
		if os.path.exists(f'models/autoencoder_{VERSION}.pt'):
			model, optimizer, current_epoch, finished_training = resume_training(model=model, optimizer=optimizer, pretraining=pretraining)
		else:
			finished_training = False
			current_epoch = 0

	# checking to see if the model was already trained or was interupted during training
	if not finished_training:

		# initializing the lists to hold the training and validation loss which will be used to plot the losses as a function of epoch
		train_loss_list, val_loss_list = [], []

		# moving the model to the available device
		model.to(DEVICE)

		# creating a function to calculate the softmax of the data
		def softmax(x):
			return torch.exp(x) / torch.exp(x).sum()

		# defining the loss function
		criterion = JSD()

		# initalizing the early stopping class
		early_stopping = Early_Stopping(decreasing_loss_patience=val_loss_patience, pretraining=pretraining)

		# looping through the epochs
		while current_epoch < num_epochs:

			# starting the clock for the epoch
			stime = time.time()

			# setting the model to training mode
			model.train()

			# initializing the running loss
			running_training_loss, running_val_loss = 0.0, 0.0

			# checking to see if there is a different X and y for the training data. Was used when gaussian noise was added to inputs
			if isinstance(next(iter(train)), list):
				for X, y in train:

					# moving the data to the available device
					X = X.to(DEVICE, dtype=torch.float)
					y = y.to(DEVICE, dtype=torch.float)

					# adding a channel dimension to the data
					X = X.unsqueeze(1)

					# forward pass
					output = model(X)

					# reshaping the data to be 1D
					output, y = output.view(-1, output.size(-1)), y.view(-1, y.size(-1))
					output, y = torch.flatten(output), torch.flatten(y)

					# calculating the loss
					loss = criterion(output, y)

					# backward pass
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					# emptying the cuda cache
					X = X.to('cpu')
					y = y.to('cpu')

					# adding the loss to the running training loss
					running_training_loss += loss.to('cpu').item()

			# if X = y for the training data (pure autoencoder)
			else:
				# looping through the batches
				for train_data in train:

					# moving the data to the available device
					train_data = train_data.to(DEVICE, dtype=torch.float)

					# adding a channel dimension to the data
					train_data = train_data.unsqueeze(1)

					# forward pass
					output = model(train_data)

					# reshaping the data to be 1D for making the histograms in the loss function
					output, train_data = output.view(-1, output.size(-1)), train_data.view(-1, train_data.size(-1))
					output, train_data = torch.flatten(output), torch.flatten(train_data)

					# calculating the loss
					loss = criterion(output, train_data)

					# checking for nans in the workflow
					torch.autograd.set_detect_anomaly(True)

					# backward pass
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					# emptying the cuda cache
					train_data = train_data.to('cpu')

					# adding the loss to the running training loss
					running_training_loss += loss.to('cpu').item()

			# setting the model to eval mode so the dropout layers are not used during validation and weights are not updated
			model.eval()

			# using validation set to check for overfitting
			if isinstance(next(iter(val)), list):

				# looping through the batches
				for X, y in val:

					# moving the data to the available device
					X = X.to(DEVICE, dtype=torch.float)
					y = y.to(DEVICE, dtype=torch.float)

					# adding a channel dimension to the data
					X = X.unsqueeze(1)

					# forward pass with no gradient calculation
					with torch.no_grad():

						output = model(X)

						# reshaping the data to be 1D for making the histograms in the loss function
						output, y = output.view(-1, output.size(-1)), y.view(-1, y.size(-1))
						output, y = torch.flatten(output), torch.flatten(y)

						# calculating the loss
						val_loss = criterion(output, y)

						# emptying the cuda cache
						X = X.to('cpu')
						y = y.to('cpu')

						# adding the loss to the running val loss
						running_val_loss += val_loss.to('cpu').item()

			# if X = y for the validation data (pure autoencoder)
			else:

				# looping through the batches
				for val_data in val:

					# moving the data to the available device
					val_data = val_data.to(DEVICE, dtype=torch.float)

					# adding a channel dimension to the data
					val_data = val_data.unsqueeze(1)

					# forward pass with no gradient calculation
					with torch.no_grad():

						output = model(val_data)

						# reshaping the data to be 1D for making the histograms in the loss function
						output, val_data = output.view(-1, output.size(-1)), val_data.view(-1, val_data.size(-1))
						output, val_data = torch.flatten(output), torch.flatten(val_data)

						# calculating the loss
						val_loss = criterion(output, val_data)

						# emptying the cuda cache
						val_data = val_data.to('cpu')

						# adding the loss to the running loss
						running_val_loss += val_loss.to('cpu').item()

			# getting the average loss for the epoch
			loss = running_training_loss/len(train)
			val_loss = running_val_loss/len(val)

			# adding the loss to the list
			train_loss_list.append(loss)
			val_loss_list.append(val_loss)

			# checking for early stopping or the end of the training epochs
			if (early_stopping(train_loss=loss, val_loss=val_loss, model=model, optimizer=optimizer, epoch=current_epoch)) or (current_epoch == num_epochs-1):

				# saving the final model
				gc.collect()

				# model = Autoencoder()

				# clearing the cuda cache
				torch.cuda.empty_cache()
				gc.collect()

				if pretraining:

					# loading the best model version
					final = torch.load(f'models/autoencoder_pretraining_{VERSION}.pt')

					# setting the finished training flag to True
					final['finished_training'] = True

					# saving the final model
					torch.save(final, f'models/autoencoder_pretraining_{VERSION}.pt')

				else:

					# loading the best model version
					final = torch.load(f'models/autoencoder_{VERSION}.pt')

					# setting the finished training flag to True
					final['finished_training'] = True

					# saving the final model
					torch.save(final, f'models/autoencoder_{VERSION}.pt')

				# breaking the loop
				break

			# getting the time for the epoch
			epoch_time = time.time() - stime

			# printing the loss for the epoch
			print(f'Epoch [{current_epoch}/{num_epochs}], Loss: {loss:.4f} Validation Loss: {val_loss:.4f}' + f' Epoch Time: {epoch_time:.2f} seconds')
			print(f'Model encoder weights example for Epoch {current_epoch}: {model.encoder[0].weight[0]}')

			# emptying the cuda cache
			torch.cuda.empty_cache()

			# updating the epoch
			current_epoch += 1

		# transforming the lists to a dataframe to be saved
		loss_tracker = pd.DataFrame({'train_loss':train_loss_list, 'val_loss':val_loss_list})
		loss_tracker.to_feather(f'outputs/autoencoder_{VERSION}_loss_tracker.feather')

		gc.collect()
		# getting the best params saved in the Early Stopping class
		if pretraining:
			model.load_state_dict(torch.load(f'models/autoencoder_pretraining_{VERSION}.pt'))
		else:
			model.load_state_dict(torch.load(f'models/autoencoder_{VERSION}.pt'))

	else:
		# loading the model if it has already been trained.
		if pretraining:
			try:
				final = torch.load(f'models/autoencoder_pretraining_{VERSION}.pt')
				model.load_state_dict(final['model'])
			except KeyError:
				model.load_state_dict(torch.load(f'models/autoencoder_pretraining_{VERSION}.pt'))
		else:
			try:
				final = torch.load(f'models/autoencoder_{VERSION}.pt')
				model.load_state_dict(final['model'])
			except KeyError:
				model.load_state_dict(torch.load(f'models/autoencoder_{VERSION}.pt'))

	return model


def evaluation(model, test, get_layer_outputs=False):
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

	layers = {"encoder.1":"conv1",
			"encoder.4":"conv2",
			"encoder.7":"conv3",
			"encoder.10":"fc1",
			"decoder.0":"fc2",
			"decoder.2":"deconv1",
			"decoder.4":"deconv2",
			"decoder.6":"deconv3",
			"decoder.8":"deconv4"}

	output_lists = {layer:[] for layer in layers.values()}

	with torch.no_grad():
		if isinstance(next(iter(test)), list):
			for X, y in test:
				X = X.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)
				X = X.unsqueeze(1)
				predicted = model(X)

				if get_layer_outputs:
					model_layers = create_feature_extractor(model, return_nodes=layers)
					intermediate_layers = model_layers(X)
					for layer in layers.values():
						output_lists[layer].append(intermediate_layers[layer].to('cpu').numpy())

				loss = F.mse_loss(predicted, y)
				running_loss += loss.item()

				# making sure the predicted value is on the cpu
				if predicted.get_device() != -1:
					predicted = predicted.to('cpu')
				if y.get_device() != -1:
					y = y.to('cpu')

				# adding the decoded result to the predicted list after removing the channel dimension
				predicted = torch.squeeze(predicted, dim=1).numpy()
				y = torch.squeeze(y, dim=1).numpy()
				predicted_list.append(predicted)
				test_list.append(y)

		else:
			for test_data in test:
				test_data = test_data.to(DEVICE, dtype=torch.float)

				test_data = test_data.unsqueeze(1)
				predicted = model(test_data)
				loss = F.mse_loss(predicted, test_data)
				running_loss += loss.item()


				if get_layer_outputs:
					model_layers = create_feature_extractor(model, return_nodes=layers)
					intermediate_layers = model_layers(test_data)
					for layer in layers.values():
						output_lists[layer].append(intermediate_layers[layer].to('cpu').numpy())

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


	if get_layer_outputs:
		for layer in output_lists.keys():
			output_lists[layer] = np.concatenate(output_lists[layer], axis=0)
		return np.concatenate(predicted_list, axis=0), np.concatenate(test_list, axis=0), running_loss/len(test), output_lists, layers
	else:
		return np.concatenate(predicted_list, axis=0), np.concatenate(test_list, axis=0), running_loss/len(test)


def plotting_some_examples(predictions, test, pretraining=False):

	vmin = min([predictions[0, :, :].min(), test[0, :, :].min()])
	vmax = max([predictions[0, :, :].max(), test[0, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[0, :, :], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[0, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	if pretraining:
		plt.savefig(f'plots/pretraining_{VERSION}_example_1.png')
	else:
		plt.savefig(f'plots/{VERSION}_example_1.png')

	vmin = min([predictions[324, :, :].min(), test[324, :, :].min()])
	vmax = max([predictions[324, :, :].max(), test[324, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[324, :, :], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[324, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	if pretraining:
		plt.savefig(f'plots/pretraining_{VERSION}_example_2.png')
	else:
		plt.savefig(f'plots/{VERSION}_example_2.png')

	vmin = min([predictions[256, :, :].min(), test[256, :, :].min()])
	vmax = max([predictions[256, :, :].max(), test[256, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[256, :, :], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[256, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	if pretraining:
		plt.savefig(f'plots/pretraining_{VERSION}_example_3.png')
	else:
		plt.savefig(f'plots/{VERSION}_example_3.png')

	vmin = min([predictions[1000, :, :].min(), test[1000, :, :].min()])
	vmax = max([predictions[1000, :, :].max(), test[1000, :, :].max()])
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(predictions[1000, :, :], vmin=vmin, vmax=vmax)
	ax1.set_title('Prediction')
	ax2 = fig.add_subplot(122)
	ax2.imshow(test[1000, :, :], vmin=vmin, vmax=vmax)
	ax2.set_title('Actual')
	if pretraining:
		plt.savefig(f'plots/pretraining_{VERSION}_example_4.png')
	else:
		plt.savefig(f'plots/{VERSION}_example_4.png')


def examining_distributions_in_parts_of_the_predictions(prediction, noise_dims, temp_dims):

	# defining the areas of the bounding boxes
	noise_left, noise_right, noise_top, noise_bottom = noise_dims
	temp_left, temp_right, temp_top, temp_bottom = temp_dims

	noise_box = prediction[noise_top:noise_bottom, noise_left:noise_right]
	temp_box = prediction[temp_top:temp_bottom, temp_left:temp_right]

	print(f'Noise Box: Mean: {noise_box.mean()}, Std: {noise_box.std()}')

	# plotting histograms of the areas of interest
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(111)
	ax1.hist(noise_box.flatten(), bins=100, alpha=0.5, label='Noise')
	ax1.hist(temp_box.flatten(), bins=100, alpha=0.5, label='Temperature')
	ax1.legend()
	plt.show()

	# plotting the prediction including bounding boxes around the areas of interest.
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(111)
	ax1.imshow(prediction, vmin=prediction.min(), vmax=prediction.max())
	ax1.add_patch(patches.Rectangle((noise_left, noise_top), noise_right - noise_left, noise_bottom - noise_top, edgecolor='red', facecolor='none'))
	ax1.add_patch(patches.Rectangle((temp_left, temp_top), temp_right - temp_left, temp_bottom - temp_top, edgecolor='red', facecolor='none'))
	plt.show()


def comparing_distributions(predictions, test):

	# plotting the histograms
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(111)

	ax1.hist(predictions.flatten(), bins=100, alpha=0.5, label='Predictions', log=True)
	ax1.hist(test.flatten(), bins=100, alpha=0.5, label='Test', log=True)
	# plt.xlim(-4, 10)
	ax1.legend()
	plt.savefig(f'plots/agu_presentation_plots/{VERSION}_distributions.png')

def plotting_histograms_for_each_layer_output(outputs, labels):

	fig, axes = plt.subplots(len(outputs.keys()), 1, figsize=(20, 20))
	for ax, label, output in zip(axes, labels.keys(), outputs.values()):
		ax.hist(output.flatten(), bins=100, alpha=0.5, log=True)
		ax.set_title(f'Layer {label}')

	plt.savefig(f'plots/{VERSION}_layer_outputs.png')


def examining_explained_variance_ratio(model, data, n_components=10):

	from sklearn.decomposition import PCA

	# getting the model layers
	model_layers = create_feature_extractor(model, return_nodes={'encoder.10':'fc1'})

	# getting the latent space
	latent = model_layers(data)

	# reshaping the latent space
	latent = latent.view(latent.size(0), -1)

	# initializing the PCA model
	pca = PCA(n_components=n_components)

	# fitting the PCA model
	pca.fit(latent)

	# getting the explained variance ratio
	explained_variance = pca.explained_variance_ratio_

	# plotting the explained variance
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(111)
	ax1.plot(explained_variance)
	ax1.set_title('Explained Variance Ratio')
	plt.savefig(f'plots/{VERSION}_explained_variance_ratio.png')

def examining_extreme_values(predictions, test):

	threshold = test.max()

	# getting the indices of the extreme values
	indices = np.where(predictions > threshold)

	# getting the values of the extreme values from the predictions
	values = predictions[indices]

	# getting the values of the extreme values from the test data
	test_values = test[indices]

	# plotting the values
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(111)
	ax1.scatter(values, test_values)
	ax1.set_xlabel('Predictions')
	ax1.set_ylabel('Test Data')
	plt.savefig(f'plots/{VERSION}_extreme_values.png')



def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data
	print('Loading data...')
	train, val, test, ___, scaling_mean, scaling_std = getting_prepared_data()

	# getting the shape of the tensor to show the model summary
	train_size = list(train.size())

	# getting the pretraining data
	# pretrain_train, pretrain_val, pretrain_test = creating_fake_twins_data(train, scaling_mean, scaling_std)

	# creating the dataloaders which prepares the data in batches for training
	train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
	val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
	test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

	# pretrain_train = DataLoader(pretrain_train, batch_size=BATCH_SIZE, shuffle=True)
	# pretrain_val = DataLoader(pretrain_val, batch_size=BATCH_SIZE, shuffle=True)
	# pretrain_test = DataLoader(pretrain_test, batch_size=BATCH_SIZE, shuffle=False)

	# creating the model
	print('Initalizing model....')
	autoencoder = Autoencoder()
	autoencoder.to(DEVICE)
	print(summary(autoencoder, (1, train_size[1], train_size[2])))

	# pretraining the model
	print('Pretraining model....')
	# autoencoder = fit_autoencoder(autoencoder, pretrain_train, pretrain_val, val_loss_patience=50, num_epochs=500, pretraining=True)

	# testing the pretrained model to make sure it works
	print('Testing pretrained model....')
	# pretrained_predictions, pretrained_test, testing_loss = evaluation(autoencoder, pretrain_test)

	# unscaling the predictions and test data
	# pretrained_predictions = (pretrained_predictions * scaling_std) + scaling_mean
	# pretrained_test = (pretrained_test * scaling_std) + scaling_mean


	# creating a directory to save the plots if it doesn't already exist
	if not os.path.exists('plots'):
		os.makedirs('plots')

	# plotting some examples
	print('Plotting some examples from pretrained....')
	# plotting_some_examples(pretrained_predictions, pretrained_test, pretraining=True)

	# examining the distributions in parts of the predictions
	print('Examining the distributions of the predictions....')
	# comparing_distributions(pretrained_predictions, pretrained_test)


	# fitting the model
	print('Fitting model....')
	autoencoder = fit_autoencoder(autoencoder, train, val, val_loss_patience=50, num_epochs=500)

	# evaluating the model
	print('Evaluating model....')
	predictions, test, testing_loss, output_layers, layers = evaluation(autoencoder, test, get_layer_outputs=True)

	plotting_histograms_for_each_layer_output(output_layers, layers)

	# unscaling the predictions and test data
	predictions = (predictions * scaling_std) + scaling_mean
	test = (test * scaling_std) + scaling_mean

	# plotting some examples
	print('Plotting some examples....')
	plotting_some_examples(predictions, test)

	# examining the distributions in parts of the predictions
	print('Examining the distributions of the predictions....')
	# examining_distributions_in_parts_of_the_predictions(predictions[324, :, :], noise_dims=(0, 40, 0, 90), temp_dims=(30, 60, 60, 80))

	# comparing the distributions of the predictions and the test data
	print('Comparing the distributions of the predictions and the test data....')
	comparing_distributions(predictions, test)

	# examining the explained variance ratio of the latent space
	print('Examining the explained variance ratio of the latent space....')
	# examining_explained_variance_ratio(autoencoder, test)

	# examining the extreme values
	print('Examining the extreme values....')
	examining_extreme_values(predictions, test)

	print(f'Loss: {testing_loss}')

if __name__ == '__main__':
	main()
	print('It ran. God job!')
