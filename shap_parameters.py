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


class SHAP_values():

	def __init__(self, model_path, model_name, loss_function, training_data, evaluation_data, features, model_config):

		self.model_path = model_path
		self.model_name = model_name
		self.loss_function = loss_function
		self.training_data = training_data
		self.evaluation_data = evaluation_data
		self.features = features
		self.model_config = model_config


	def load_model(self):

		# Load model
		self.model = load_model(self.model_path, compile=False)
		self.model.compile(loss=self.loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate']))

		return self.model

	def get_shap_values(self, model=None, background_examples=1000):
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

		if model is None:
			model = self.model

		if os.path.exists(f'outputs/shap_values/{self.model_name}_shap_values.pkl'):
			with open(f'outputs/shap_values/{self.model_name}_shap_values.pkl', 'rb') as f:
				self.shap_values = pickle.load(f)

		else:
			# checking to see if the xtrain is a list of multiple inputs. Creates background for each using same random sampling
			if isinstance(self.training_data, list):
				background = []
				random_indicies = np.random.choice(self.training_data[0].shape[0], background_examples, replace=False)
				for i in range(len(self.training_data)):
					background.append(self.training_data[i][random_indicies])

			else:
				# Get shap values
				background = self.training_data[np.random.choice(self.training_data.shape[0], background_examples, replace=False)]

			explainer = shap.DeepExplainer(model, background)

			if isinstance(self.evaluation_data, list):
				self.shap_values = []
				for i in range(len(self.evaluation_data)):
					self.shap_values.append(explainer.shap_values(self.evaluation_data[i], check_additivity=False))

			else:
				self.shap_values = explainer.shap_values(self.evaluation_data, check_additivity=False)

			with open(f'outputs/shap_values/{self.model_name}_shap_values.pkl', 'wb') as f:
				pickle.dump(shap_values, f)

		return self.shap_values


	def converting_shap_to_percentages(self, shap_values=None):

		if shap_values is None:
			shap_values = self.shap_values

		if shap_values.shape[0] > 1:
			self.all_shap_values = []
			for i in range(shap_values.shape[0]):
				summed_shap_values = np.sum(shap_values[i], axis=1)
				summed_shap_values = summed_shap_values.reshape(summed_shap_values.shape[0], summed_shap_values.shape[1])
				shap_df = pd.DataFrame(summed_shap_values, columns=self.features)
				perc_df = (shap_df.div(shap_df.abs().sum(axis=1), axis=0))*100
				self.all_shap_values.append(perc_df)

		else:
			summed_shap_values = np.sum(shap_values, axis=1)
			summed_shap_values = summed_shap_values.reshape(summed_shap_values.shape[0], summed_shap_values.shape[1])
			shap_df = pd.DataFrame(summed_shap_values, columns=self.features)
			perc_df = (shap_df.div(shap_df.abs().sum(axis=1), axis=0))*100
			self.all_shap_values = perc_df

		return self.all_shap_values









