import datetime as dt
import inspect

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence


class Generator(Sequence):
	"""Generates data for Keras
	Sequence based data generator. Suitable for building data generator for training and prediction.
	"""
	def __init__(self, features, results, to_fit=True, batch_size=32, shuffle=False):
		"""Initialization
		:param list_IDs: list of all 'label' ids to use in the generator
		:param image_path: path to images location
		:param mask_path: path to masks location
		:param to_fit: True to return X and y, False to return X only
		:param batch_size: batch size at each iteration
		:param shuffle: True to shuffle label indexes after every epoch
		"""
		self.features = features
		self.results = results
		self.to_fit = to_fit
		self.batch_size = batch_size
		self.shuffle = shuffle

	def __len__(self):
		"""Denotes the number of batches per epoch
		:return: number of batches per epoch
		"""
		return int(np.floor(len(self.features[0]) / self.batch_size))


	def __getitem__(self, index):
		"""Generate one batch of data
		:param index: index of the batch
		:return: X and y when fitting. X only when predicting
		"""

		# Generating data from batch indices. Putting the multipe features into a list
		X = []
		for i in range(len(self.features)):
			x = np.empty((self.batch_size, self.features[i].shape[1], self.features[i].shape[2], 1))
			x = self.features[i][index * self.batch_size:(index + 1) * self.batch_size]
			X.append(x)

		# Returning X and y when fitting
		if self.to_fit:
			y = np.empty((self.batch_size, 1))
			y = self.results[index * self.batch_size:(index + 1) * self.batch_size]
			return X, y
		else:
			return X