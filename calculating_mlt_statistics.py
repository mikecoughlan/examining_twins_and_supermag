import gc
import glob
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm



class Calculating_MLT_Statistics():
	'''
	Creates a dataframe of statistics calulated within particular MLT bins.
	'''

	def __init__(self, df, mlt_min, mlt_max, mlt_step, param):
		'''
		Initialization function for the class.

		Args:
			df (pd.DataFrame): dataframe containing the parameters to calculate the stats for. Must contain an 'MLT' column.
			mlt_min (int/float): minimum MLT bin to calculate stats
			mlt_max (int/float): maximum MLT bin to calculate stats
			mlt_step (int/float): bin size in MLT
			param (string): the parameter the statistics will be calculated for.
		'''
		
		self.df = df
		self.mlt_min = mlt_min
		self.mlt_max = mlt_max
		self.mlt_step = mlt_step
		self.param = param
	

	def filter_data(self, mlt_min_bin, mlt_max_bin):
		'''
		Filter a pandas data frame to MLT bins.

		Args:
			mlt_min_bin (int/float): MLT bin min value (inclusive)
			mlt_max_bin (int/float): MLT bin max value (exclusive)

		Returns:
			pd.DataFrame: dataframe containing only data from within the MLT bin limits
		'''

		mlt_df = self.df[(self.df['MLT'] >= mlt_min_bin) & (self.df['MLT'] < mlt_max_bin)]

		mlt_df.reset_index(inplace=True, drop=True)

		return mlt_df


	def compute_statistics(self, filtered_df, mlt):
		'''
		Compute the statistics of the parameter for each degree bins.

		Args:
			filtered_df (pd.DataFrame): dataframe filtered into the MLT bin
			mlt (int/float): minimim MLT bin, used as index

		Returns:
			pd.Dataframe: dataframe containing the resulting stats. Only contains one row with min MLT as index.
		'''

		filtered_df = filtered_df[filtered_df[self.param].notna()]
		stats_df = pd.DataFrame({'MLT': mlt,
								'count':len(filtered_df),
								'mean': filtered_df[self.param].mean(),
								'median':filtered_df[self.param].median(),
								'std': filtered_df[self.param].std(),
								'max':filtered_df[self.param].max()},
								index=[0])

		return stats_df


	def process_directory(self):
		'''
		Process all feather files in a directory and return a list
		of filtered data frames for each degree bins.

		Returns:
			pd.dataframe: dataframe with the stats for all the MLT bins
		'''

		temp_df = pd.DataFrame()
		for mlt in np.arange(self.mlt_min, self.mlt_max, self.mlt_step):
			mlt_min_bin = mlt
			mlt_max_bin = mlt + self.mlt_step
			df_filtered = self.filter_data(mlt_min_bin, mlt_max_bin)
			if not df_filtered.empty:
				stat = self.compute_statistics(df_filtered, mlt)
				temp_df = pd.concat([temp_df, stat], axis=0, ignore_index=True)

		if temp_df.empty:
			temp_df = pd.DataFrame({'count':np.nan,
									'mean':np.nan,
									'median':np.nan,
									'std':np.nan,
									'max':np.nan},
									index=np.arange(self.mlt_min, self.mlt_max, self.mlt_step))
		else:
			temp_df.set_index("MLT", inplace=True)


		return temp_df