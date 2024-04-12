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
from matplotlib import patches
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchsummary import summary
from torchvision import models
from torchvision.transforms.functional import rotate

import utils

# from data_generator import Generator
# from data_prep import DataPrep

logging.basicConfig(level=logging.INFO, format='')

TARGET = 'rsd'
REGION = 163
VERSION = 'pytorch_perceptual_v1-39'

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
BATCH_SIZE = 16

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


class Logger:
	def __init__(self):
		self.entries = {}

	def add_entry(self, entry):
		self.entries[len(self.entries) + 1] = entry

	def __str__(self):
		return json.dumps(self.entries, sort_keys=True, indent=4)


def loading_data(target_var, region):
	'''
	Function to load the data for the model.

	Args:
		target_var (str): the target variable to be used in the model
		region (int): the region to be used in the model

	Returns:
		pd.DataFrame: the merged dataframe
		float: the mean latitude of the region
		dict: the TWINS maps

	'''

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


def generate_gaussian_2d(num_sample, shape, max_value, peak_location=None, peak_std=None):
	"""
	Generate a 2D NumPy array representing a Gaussian distribution with a specified maximum value and optional random peak location.

	Args:
		shape (tuple): Shape of the array (height, width).
		max_value (float): Maximum value of the Gaussian distribution.
		peak_location (tuple): Peak location coordinates (row, column). If None, a random location is chosen.
		peak_std (float): Standard deviation of the Gaussian peak.

	Returns:
		numpy.ndarray: 2D array representing the Gaussian distribution.
	"""

	# initalizing the np.array to store the gaussian distributions
	gaussian_array = np.zeros((num_sample, shape[0], shape[1]))

	for i in range(num_sample):
		# Generate random peak location and standard deviation if not specified
		if peak_location is None:
			peak_location = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
		if peak_std is None:
			peak_std = np.random.uniform(0.01, 0.9) * shape[0]

		# Generate 2D Gaussian distribution
		x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
		x0, y0 = peak_location
		gaussian = max_value * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * peak_std ** 2))

		gaussian_array[i, :, :] = gaussian

		# resetting the peak location and peak std
		peak_location = None
		peak_std = None

	return gaussian_array


def creating_pretraining_data(tensor_shape, train_max, train_min, scaling_mean, scaling_std, num_samples=10000):
	'''
	Function to create the data to pretrain the autoencoder. This data is used to train the autoencoder
	to reduce the dimensionality of the TWINS maps.

	Args:
		tensor_shape (tuple): the shape of the tensor to be created
		train_max (float): the maximum value of the training data
		train_min (float): the minimum value of the training data
		scaling_mean (float): the mean of the training data
		scaling_std (float): the standard deviation of the training data
		num_samples (int): the number of samples to be created
		distribution (str): the distribution of the data to be created

	Returns:
		torch.tensor: the training data for the autoencoder
		torch.tensor: the validation data for the autoencoder
		torch.tensor: the testing data for the autoencoder

	'''

	# train_data = np.random.normal(low=train_min, high=train_max, size=(int(num_samples*0.7), tensor_shape[0], tensor_shape[1]))
	# val_data = np.random.normal(low=train_min, high=train_max, size=(int(num_samples*0.2), tensor_shape[0], tensor_shape[1]))
	# test_data = np.random.normal(low=train_min, high=train_max, size=(int(num_samples*0.1), tensor_shape[0], tensor_shape[1]))

	train_data = generate_gaussian_2d(int(num_samples*0.7), tensor_shape, train_max)
	val_data = generate_gaussian_2d(int(num_samples*0.2), tensor_shape, train_max)
	test_data = generate_gaussian_2d(int(num_samples*0.1), tensor_shape, train_max)

	# plotting 9 random examples of training data
	fig, axes = plt.subplots(3, 3, figsize=(10, 10))
	for i, ax in enumerate(axes.flatten()):
		ax.imshow(train_data[i, :, :])
		ax.set_title(f'Example {i+1}')
	plt.show()

	# plotting 9 random examples of validation data
	fig, axes = plt.subplots(3, 3, figsize=(10, 10))
	for i, ax in enumerate(axes.flatten()):
		ax.imshow(val_data[i, :, :])
		ax.set_title(f'Example {i+1}')
	plt.show()

	# scaling the data
	train_data = standard_scaling(train_data, scaling_mean, scaling_std)
	val_data = standard_scaling(val_data, scaling_mean, scaling_std)
	test_data = standard_scaling(test_data, scaling_mean, scaling_std)

	# making figures of examples of the data
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(121)
	ax1.imshow(train_data[10, :, :])
	ax1.set_title('Training Example')
	ax2 = fig.add_subplot(122)
	ax2.imshow(val_data[10, :, :])
	ax2.set_title('Validation Example')
	plt.show()

	# converting the data to tensors
	train_data = torch.tensor(train_data, dtype=torch.float)
	val_data = torch.tensor(val_data, dtype=torch.float)
	test_data = torch.tensor(test_data, dtype=torch.float)

	return train_data, val_data, test_data


def creating_fake_twins_data(train, scaling_mean, scaling_std):

	# splitting the training data into train val and test
	train_data, test_data = train_test_split(train, test_size=0.1, random_state=RANDOM_SEED)
	train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=RANDOM_SEED)

	# rotating the data 90 degrees with expansion then cutting them to the 90,60 size
	train_rotated_90 = rotate(train_data, angle=90, expand=True)
	val_rotated_90 = rotate(val_data, angle=90, expand=True)
	test_rotated_90 = rotate(test_data, angle=90, expand=True)

	train_rotated_90 = train_rotated_90[:, 0:90, 0:60]
	val_rotated_90 = val_rotated_90[:, 0:90, 0:60]
	test_rotated_90 = test_rotated_90[:, 0:90, 0:60]

	# adding a 30, 60 slice to the data at the 1 dimension
	train_rotated_90 = torch.cat([train_rotated_90, train_rotated_90[:, 0:30, 0:60]], dim=1)
	val_rotated_90 = torch.cat([val_rotated_90, val_rotated_90[:, 0:30, 0:60]], dim=1)
	test_rotated_90 = torch.cat([test_rotated_90, test_rotated_90[:, 0:30, 0:60]], dim=1)

	# rotating the data 180 degrees
	train_rotated_180 = rotate(train_data, angle=180, expand=False)
	val_rotated_180 = rotate(val_data, angle=180, expand=False)
	test_rotated_180 = rotate(test_data, angle=180, expand=False)

	# rotating the data 270 with an expansion and then cutting it down to the 90, 60 size
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

	# flipping the data
	flipped_train_data = torch.flip(train_data, [1])
	flipped_val_data = torch.flip(val_data, [1])
	flipped_test_data = torch.flip(test_data, [1])

	# flipping the data
	flipped_train_data_2 = torch.flip(train_data, [2])
	flipped_val_data_2 = torch.flip(val_data, [2])
	flipped_test_data_2 = torch.flip(test_data, [2])

	# concatenating the data without the origonal data
	train_data = torch.cat([train_rotated_90, train_rotated_180, train_rotated_270, flipped_train_data, flipped_train_data_2], dim=0)
	val_data = torch.cat([val_rotated_90, val_rotated_180, val_rotated_270, flipped_val_data, flipped_val_data_2], dim=0)
	test_data = torch.cat([test_rotated_90, test_rotated_180, test_rotated_270, flipped_test_data, flipped_test_data_2], dim=0)

	mean, std = -0.9, 0.2141

	# X_train = train_data + torch.tensor(np.random.normal(loc=mean, scale=std, size=train_data.shape))
	# X_val = val_data + torch.tensor(np.random.normal(loc=mean, scale=std, size=val_data.shape))
	# X_test = test_data + torch.tensor(np.random.normal(loc=mean, scale=std, size=test_data.shape))

	# scaling the data
	train_data = standard_scaling(train_data, scaling_mean, scaling_std)
	val_data = standard_scaling(val_data, scaling_mean, scaling_std)
	test_data = standard_scaling(test_data, scaling_mean, scaling_std)

	# X_train = standard_scaling(X_train, scaling_mean, scaling_std)
	# X_val = standard_scaling(X_val, scaling_mean, scaling_std)
	# X_test = standard_scaling(X_test, scaling_mean, scaling_std)

	# # plotting some examples of the data
	# fig, axes = plt.subplots(3, 3, figsize=(10, 10))
	# for i, ax in enumerate(axes.flatten()):
	# 	ax.imshow(X_train[i, :, :])
	# 	ax.set_title(f'Example {i+1}')
	# plt.show()

	# return X_train, X_val, X_test, train_data, val_data, test_data
	return train_data, val_data, test_data


def standard_scaling(x, scaling_mean, scaling_std):
	# scaling the data to have a mean of 0 and a standard deviation of 1
	return (x - scaling_mean) / scaling_std


def minmax_scaling(x, scaling_min, scaling_max):
	# scaling the data to be between 0 and 1
	return (x - scaling_min) / (scaling_max - scaling_min)


def keV_to_eV(x):
	# changing positive values in the array to eV
	return x *1000


def getting_prepared_data(get_features=False):
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
		features (list): the features of the data

	'''

	maps = utils.loading_twins_maps()

	temp_version = 'pytorch_test'

	with open(working_dir+f'twins_method_storm_extraction_map_keys_version_{temp_version}.pkl', 'rb') as f:
		storms_extracted_dict = pickle.load(f)
	storms = storms_extracted_dict['storms']
	target = storms_extracted_dict['target']
	features = storms[0].columns

	# splitting the data on a day to day basis to reduce data leakage
	day_df = pd.date_range(start=pd.to_datetime('2009-07-01'), end=pd.to_datetime('2017-12-01'), freq='D')
	specific_test_days = pd.date_range(start=pd.to_datetime('2012-03-07'), end=pd.to_datetime('2012-03-13'), freq='D')
	day_df = day_df.drop(specific_test_days)

	train_days, test_days = train_test_split(day_df, test_size=0.1, shuffle=True, random_state=RANDOM_SEED)
	train_days, val_days = train_test_split(train_days, test_size=0.125, shuffle=True, random_state=RANDOM_SEED)

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

	twins_scaling_array = np.vstack(twins_train).flatten()

	twins_scaling_array = twins_scaling_array[twins_scaling_array > 0]
	scaling_mean = twins_scaling_array.mean()
	scaling_std = twins_scaling_array.std()

	twins_train = [standard_scaling(x, scaling_mean, scaling_std) for x in twins_train]
	twins_val = [standard_scaling(x, scaling_mean, scaling_std) for x in twins_val]
	twins_test = [standard_scaling(x, scaling_mean, scaling_std) for x in twins_test]

	if not get_features:
		return torch.tensor(twins_train), torch.tensor(twins_val), torch.tensor(twins_test), date_dict, scaling_mean, scaling_std
	else:
		return torch.tensor(twins_train), torch.tensor(twins_val), torch.tensor(twins_test), date_dict, scaling_mean, scaling_std, features



class PerceptualLoss(nn.Module):
	'''loss function using the residuals of a pretrained model to calculate the
			loss between of the feature maps between the predicted and real images'''

	def __init__(self, conv_index: str = '22'):
		'''
		Initializing the PerceptualLoss class.

		Args:
			conv_index (str): the index of the convolutional layer to be used to calculate the loss

		'''

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

		# calculating the loss
		loss = F.mse_loss(output_features, target_features)

		return loss


class VGGPerceptualLoss(torch.nn.Module):
	def __init__(self, resize=True):
		'''
		Initializing the VGGPerceptualLoss class.

		Args:
			resize (bool): whether to resize the images to 224x224

		'''

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
		'''
		Function to calculate the perceptual loss between the output and target images.

		Args:
			output (torch.tensor): the predicted image
			target (torch.tensor): the real image
			feature_layers (list): the layers to be used to calculate the loss

		Returns:
			float: the loss between the two images
		'''

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
			if i in feature_layers:
				loss += F.mse_loss(output, target)
		return loss


class Autoencoder(nn.Module):
	def __init__(self):
		'''
		Initializing the autoencoder model.

		'''
		super(Autoencoder, self).__init__()
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

			nn.Flatten(),
			nn.Linear(256*45*30, 420),
		)
		self.decoder = nn.Sequential(

			nn.Linear(420, 256*45*30),
			nn.Unflatten(1, (256, 45, 30)),

			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.Dropout(0.2),

			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
			nn.Dropout(0.2),

			nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=1),
			nn.Dropout(0.2),

			nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=1, padding=0),
			nn.ReLU()
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

		# x = x.unsqueeze(1)
		latent = self.encoder(x)
		if get_latent:
			return latent
		else:
			x = self.decoder(latent)
			return x


class BaseModel(nn.Module):
	'''
	_summary_: Base class for all models.

	Args:
		nn (nn): the base class for all neural network modules
	'''
	def __init__(self):
		super(BaseModel, self).__init__()
		self.logger = logging.getLogger(self.__class__.__name__)

	def forward(self):
		raise NotImplementedError

	def summary(self):
		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		nbr_params = sum([np.prod(p.size()) for p in model_parameters])
		self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

	def __str__(self):
		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		nbr_params = sum([np.prod(p.size()) for p in model_parameters])
		return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'


class SegNet(BaseModel):
	def __init__(self, latent_dims=120, in_channels=1, pretrained=True, freeze_bn=False, **_):
		super(SegNet, self).__init__()
		vgg_bn = models.vgg16_bn(pretrained= pretrained)
		encoder = list(vgg_bn.features.children())
		self.latent_dims = latent_dims

		encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

		# Encoder, VGG without any maxpooling
		self.stage1_encoder = nn.Sequential(*encoder[:6])
		self.stage2_encoder = nn.Sequential(*encoder[7:13])
		self.stage3_encoder = nn.Sequential(*encoder[14:23])
		self.stage4_encoder = nn.Sequential(*encoder[24:33])
		self.stage5_encoder = nn.Sequential(*encoder[34:-1])
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.flatten = nn.Flatten()

		# Decoder, same as the encoder but reversed, maxpool will not be used
		decoder = encoder
		decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
		# Replace the last conv layer
		decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		# When reversing, we also reversed conv->batchN->relu, correct it
		decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
		# Replace some conv layers & batchN after them
		for i, module in enumerate(decoder):
			if isinstance(module, nn.Conv2d):
				if module.in_channels != module.out_channels:
					decoder[i+1] = nn.BatchNorm2d(module.in_channels)
					decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

		self.stage1_decoder = nn.Sequential(*decoder[0:9])
		self.stage2_decoder = nn.Sequential(*decoder[9:18])
		self.stage3_decoder = nn.Sequential(*decoder[18:27])
		self.stage4_decoder = nn.Sequential(*decoder[27:33])
		self.stage5_decoder = nn.Sequential(*decoder[33:],
				nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
		)
		self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

		self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
									self.stage4_decoder, self.stage5_decoder)
		if freeze_bn: self.freeze_bn()

	def linear(self, in_channels, out_channels):
		return nn.Sequential(nn.Linear(in_channels, out_channels)).to(DEVICE)


	def _initialize_weights(self, *stages):
		for modules in stages:
			for module in modules.modules():
				if isinstance(module, nn.Conv2d):
					nn.init.kaiming_normal_(module.weight)
					if module.bias is not None:
						module.bias.data.zero_()
				elif isinstance(module, nn.BatchNorm2d):
					module.weight.data.fill_(1)
					module.bias.data.zero_()

	def forward(self, x, get_latent=False):

		# x = x.repeat(1, 3, 1, 1)

		# Encoder
		x = self.stage1_encoder(x)
		x1_size = x.size()
		x, indices1 = self.pool(x)

		x = self.stage2_encoder(x)
		x2_size = x.size()
		x, indices2 = self.pool(x)

		x = self.stage3_encoder(x)
		x3_size = x.size()
		x, indices3 = self.pool(x)

		x = self.stage4_encoder(x)
		x4_size = x.size()
		x, indices4 = self.pool(x)

		x = self.stage5_encoder(x)
		x5_size = x.size()
		x, indices5 = self.pool(x)

		last_pool_size = x.size()
		# last_pool_size.to(DEVICE)

		# Latent space
		x = self.flatten(x)

		latent = self.linear(last_pool_size[1]*last_pool_size[2]*last_pool_size[3], self.latent_dims)(x)

		# Decoder
		x = self.linear(self.latent_dims, last_pool_size[1]*last_pool_size[2]*last_pool_size[3])(latent)
		x = x.view(last_pool_size)

		x = self.unpool(x, indices=indices5, output_size=x5_size)
		x = self.stage1_decoder(x)

		x = self.unpool(x, indices=indices4, output_size=x4_size)
		x = self.stage2_decoder(x)

		x = self.unpool(x, indices=indices3, output_size=x3_size)
		x = self.stage3_decoder(x)

		x = self.unpool(x, indices=indices2, output_size=x2_size)
		x = self.stage4_decoder(x)

		x = self.unpool(x, indices=indices1, output_size=x1_size)
		x = self.stage5_decoder(x)

		if get_latent:
			return latent
		else:
			return x

	def get_backbone_params(self):
		return []

	def get_decoder_params(self):
		return self.parameters()

	def freeze_bn(self):
		for module in self.modules():
			if isinstance(module, nn.BatchNorm2d): module.eval()


class Early_Stopping():
	'''
	Class to create an early stopping condition for the model.

	'''

	def __init__(self, decreasing_loss_patience=25, training_diff_patience=3, pretraining=False):
		'''
		Initializing the class.

		Args:
			decreasing_loss_patience (int): the number of epochs to wait before stopping the model if the validation loss does not decrease
			training_diff_patience (int): the number of epochs to wait before stopping the model if the training loss is significantly lower than the validation loss
			pretraining (bool): whether the model is being pre-trained. Just used for saving model names.

		'''
		self.decreasing_loss_patience = decreasing_loss_patience
		self.training_diff_patience = training_diff_patience
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

		self.model = model
		self.optimizer = optimizer
		if self.best_score is None:
			self.best_score = val_loss
			self.best_loss = val_loss
			self.save_checkpoint(val_loss)
			self.best_epoch = epoch
		elif val_loss > self.best_score:
			self.loss_counter += 1
			if self.loss_counter >= self.decreasing_loss_patience:
				gc.collect()
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
		'''
		Function to continually save the best model.

		Args:
			val_loss (float): the validation loss for the model
		'''

		if self.best_loss > val_loss:
			self.best_loss = val_loss
			print('Saving checkpoint!')
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
	Function to resume training of a model.

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
		try:
			checkpoint = torch.load(f'models/autoencoder_pretraining_{VERSION}.pt')
			model.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			epoch = checkpoint['best_epoch']
			finished_training = checkpoint['finished_training']
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


def fit_autoencoder(model, train, val, val_loss_patience=25, overfit_patience=5, num_epochs=500, pretraining=False):

	'''
	_summary_: Function to train the autoencoder model.

	Args:
		model (object): the model to be trained
		train (torch.utils.data.DataLoader): the training data
		val (torch.utils.data.DataLoader): the validation data
		val_loss_patience (int): the number of epochs to wait before stopping the model if the validation loss does not decrease
		overfit_patience (int): the number of epochs to wait before stopping the model if the training loss is significantly lower than the validation loss
		num_epochs (int): the number of epochs to train the model
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the trained model
	'''

	optimizer = optim.Adam(model.parameters(), lr=1e-6)

	# checking if the model has already been trained
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

	if not finished_training:

		train_loss_list, val_loss_list = [], []

		# moving the model to the available device
		model.to(DEVICE)

		# # defining the loss function and the optimizer
		# if pretraining:
		mse = nn.MSELoss()
		kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

		def softmax(x):
			return torch.exp(x) / torch.exp(x).sum()

		# def criterion(y_hat, y):
		# 	mse_loss = mse(y_hat, y)

		# 	# div_y = minmax_scaling(y, scale_min, scale_max)
		# 	# div_y_hat = minmax_scaling(y_hat, scale_min, scale_max)

		# 	scale_min = min(y.min(), y_hat.min())
		# 	scale_max = max(y.max(), y_hat.max())

		# 	div_y = torch.histc(y, bins=100, min=scale_min.item(), max=scale_max.item())
		# 	div_y_hat = torch.histc(y_hat, bins=100, min=scale_min.item(), max=scale_max.item())

		# 	scale_min = min(div_y.min(), div_y_hat.min())
		# 	scale_max = max(div_y.max(), div_y_hat.max())

		# 	div_y = minmax_scaling(div_y, scale_min, scale_max)
		# 	div_y_hat = minmax_scaling(div_y_hat, scale_min, scale_max)

		# 	# div_y = softmax(div_y)
		# 	# div_y_hat = softmax(div_y_hat)

		# 	# quickly plotting the histograms

		# 	# # changing zeros to very small value
		# 	div_y_non_zero_min = div_y[div_y != 0].min()
		# 	div_y_hat_non_zero_min = div_y_hat[div_y_hat != 0].min()

		# 	div_y[div_y == 0] = 0.1*div_y_non_zero_min
		# 	div_y_hat[div_y_hat == 0] = 0.1*div_y_hat_non_zero_min

		# 	# plt.plot(div_y.cpu().detach().numpy(), label='test')
		# 	# plt.plot(div_y_hat.cpu().detach().numpy(), label='pred')
		# 	# plt.legend()
		# 	# plt.show()

		# 	div_loss = kl_loss(div_y_hat.log(), div_y.log())
		# 	loss = mse_loss + div_loss
		# 	return loss


		# 	# criterion = nn.L1Loss() 		# this calculates the mean absolute error losses
		# else:
		# 	criterion = VGGPerceptualLoss()

		# criterion = nn.MSELoss()

		# scaler = torch.cuda.amp.GradScaler()

		# initalizing the early stopping class
		early_stopping = Early_Stopping(decreasing_loss_patience=val_loss_patience, training_diff_patience=overfit_patience, pretraining=pretraining)

		while current_epoch < num_epochs:

			# starting the clock for the epoch
			stime = time.time()

			# if not pretraining, freezing all but the last layer for the first 10 epochs
			# if not pretraining:
			# 	if epoch < 10:

			# 		# freezing all layers of encoder
			# 		for param in model.encoder.parameters():
			# 			param.requires_grad = False
			# 		# freezing all layers of decoder except the last layer
			# 		for param in model.decoder.parameters():
			# 			param.requires_grad = False
			# 		for param in model.decoder[-1].parameters():
			# 			param.requires_grad = True

			# 	else:
			# 		# unfreezing all layers of encoder
			# 		for param in model.encoder.parameters():
			# 			param.requires_grad = True
			# 		# unfreezing all layers of decoder
			# 		for param in model.decoder.parameters():
			# 			param.requires_grad = True

			# setting the model to training mode
			model.train()

			# initializing the running loss
			running_training_loss, running_val_loss = 0.0, 0.0

			# shuffling data and creating batches
			if isinstance(next(iter(train)), list):
				for X, y in train:
					X = X.to(DEVICE, dtype=torch.float)
					y = y.to(DEVICE, dtype=torch.float)
					X = X.unsqueeze(1)
					# forward pass
					# with torch.cuda.amp.autocast():
					output = model(X)

					loss = criterion(output, y)
						# loss.requires_grad = True

					# backward pass
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					X = X.to('cpu')
					y = y.to('cpu')
					# adding the loss to the running loss
					running_training_loss += loss.to('cpu').item()

			else:
				for train_data in train:
					train_data = train_data.to(DEVICE, dtype=torch.float)
					train_data = train_data.unsqueeze(1)
					# forward pass
					# with torch.cuda.amp.autocast():
					output = model(train_data)

					loss = criterion(output, train_data)
						# loss.requires_grad = True

					# backward pass
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					train_data = train_data.to('cpu')

					# adding the loss to the running loss
					running_training_loss += loss.to('cpu').item()

			# setting the model to evaluation mode
			model.eval()

			# using validation set to check for overfitting
			if isinstance(next(iter(val)), list):
				for X, y in val:
					X = X.to(DEVICE, dtype=torch.float)
					y = y.to(DEVICE, dtype=torch.float)
					X = X.unsqueeze(1)
					with torch.no_grad():
						output = model(X)
						val_loss = criterion(output, y)

						X = X.to('cpu')
						y = y.to('cpu')
						running_val_loss += val_loss.to('cpu').item()
			else:
				with torch.no_grad():
					for val_data in val:
						val_data = val_data.to(DEVICE, dtype=torch.float)
						val_data = val_data.unsqueeze(1)
						output = model(val_data)

						val_loss = criterion(output, val_data)

						val_data = val_data.to('cpu')

						# adding the loss to the running loss
						running_val_loss += val_loss.to('cpu').item()

			# getting the average loss for the epoch
			loss = running_training_loss/len(train)
			val_loss = running_val_loss/len(val)

			# adding the loss to the list
			train_loss_list.append(loss)
			val_loss_list.append(val_loss)

			# checking for early stopping
			if (early_stopping(train_loss=loss, val_loss=val_loss, model=model, optimizer=optimizer, epoch=current_epoch)) or (current_epoch == num_epochs-1):
				final = torch.load(f'models/autoencoder_pretraining_{VERSION}.pt')
				final['finished_training'] = True
				torch.save(final, f'models/autoencoder_pretraining_{VERSION}.pt')
				break

			# getting the time for the epoch
			epoch_time = time.time() - stime

			# if epoch % 5 == 0:
			print(f'Epoch [{current_epoch}/{num_epochs}], Loss: {loss:.4f} Validation Loss: {val_loss:.4f}' + f' Epoch Time: {epoch_time:.2f} seconds')

			# emptying the cuda cache
			torch.cuda.empty_cache()

			# updating the epoch
			current_epoch += 1

		gc.collect()
		# getting the best params saved in the Early Stopping class
		if pretraining:
			model.load_state_dict(torch.load(f'models/autoencoder_pretraining_{VERSION}.pt'))
		else:
			model.load_state_dict(torch.load(f'models/autoencoder_{VERSION}.pt'))

		# transforming the lists to a dataframe to be saved
		loss_tracker = pd.DataFrame({'train_loss':train_loss_list, 'val_loss':val_loss_list})
		loss_tracker.to_feather(f'outputs/autoencoder_{VERSION}_loss_tracker.feather')

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
		if isinstance(next(iter(test)), list):
			for X, y in test:
				X = X.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)
				X = X.unsqueeze(1)
				predicted = model(X)
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

	ax1.hist(predictions.flatten(), bins=100, alpha=0.5, label='Predictions', density=True)
	ax1.hist(test.flatten(), bins=100, alpha=0.5, label='Test', density=True)
	ax1.legend()
	plt.show()

def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	train, val, test, ___, scaling_mean, scaling_std = getting_prepared_data()

	# getting the shape of the tensor
	train_size = list(train.size())

	# # getting the pretraining data
	# pretrain_train, pretrain_val, pretrain_test = creating_pretraining_data(tensor_shape=(train_size[1], train_size[2]),
	# 																			train_max=torch.max(train).item(), train_min=torch.min(train).item(),
	# 																			scaling_mean=scaling_mean, scaling_std=scaling_std,
	# 																			num_samples=100000)

	# getting the pretraining data
	# X_pretrain_train, X_pretrain_val, X_pretrain_test, y_pretrain_train, y_pretrain_val, y_pretrain_test = creating_fake_twins_data(train, scaling_mean, scaling_std)

	pretrain_train, pretrain_val, pretrain_test = creating_fake_twins_data(train, scaling_mean, scaling_std)

	# creating the dataloaders
	train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
	val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
	test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

	# pretrain_train = DataLoader(list(zip(X_pretrain_train, y_pretrain_train)), batch_size=BATCH_SIZE, shuffle=True)
	# pretrain_val = DataLoader(list(zip(X_pretrain_val, y_pretrain_val)), batch_size=BATCH_SIZE, shuffle=True)
	# pretrain_test = DataLoader(list(zip(X_pretrain_test, y_pretrain_test)), batch_size=BATCH_SIZE, shuffle=False)

	pretrain_train = DataLoader(pretrain_train, batch_size=BATCH_SIZE, shuffle=True)
	pretrain_val = DataLoader(pretrain_val, batch_size=BATCH_SIZE, shuffle=True)
	pretrain_test = DataLoader(pretrain_test, batch_size=BATCH_SIZE, shuffle=False)

	# creating the model
	print('Initalizing model....')
	autoencoder = Autoencoder()
	autoencoder.to(DEVICE)
	print(summary(autoencoder, (1, train_size[1], train_size[2])))

	# pretraining the model
	print('Pretraining model....')
	autoencoder = fit_autoencoder(autoencoder, pretrain_train, pretrain_val, val_loss_patience=50, num_epochs=500, pretraining=True)

	# testing the pretrained model to make sure it works
	print('Testing pretrained model....')
	pretrained_predictions, pretrained_test, testing_loss = evaluation(autoencoder, pretrain_test)

	# unscaling the predictions and test data
	pretrained_predictions = (pretrained_predictions * scaling_std) + scaling_mean
	pretrained_test = (pretrained_test * scaling_std) + scaling_mean

	# creating a directory to save the plots if it doesn't already exist
	if not os.path.exists('plots'):
		os.makedirs('plots')

	# plotting some examples
	print('Plotting some examples from pretrained....')
	plotting_some_examples(pretrained_predictions, pretrained_test, pretraining=True)

	# fitting the model
	print('Fitting model....')
	autoencoder = fit_autoencoder(autoencoder, train, val, val_loss_patience=50, num_epochs=500)

	# evaluating the model
	print('Evaluating model....')
	predictions, test, testing_loss = evaluation(autoencoder, test)

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

	print(f'Loss: {testing_loss}')

if __name__ == '__main__':
	main()
	print('It ran. God job!')
