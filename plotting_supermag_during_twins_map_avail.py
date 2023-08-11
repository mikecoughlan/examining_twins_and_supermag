import gc
import os
import pickle

import scipy
from scipy.io import netcdf_file

os.environ["CDF_LIB"] = "~/CDF/lib"
import datetime
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacepy
from spacepy import pycdf
from tqdm import tqdm


def plotting(region, name):
	'''
	plots a heatmap of a particular parameter using imshow. First transforms the data frame into a 2d array for plotting

	Args:
		stats (pd.df): dataframe containing the locations and values
	'''
	params = ['mean', 'median', 'max', 'std']

	stations = [key for key in region.keys() if key not in ['rsd', 'rsd_stats']]

	colors = sns.color_palette('tab20', len(stations))
	color_map = dict(zip(stations, colors))
	color_map.update({np.nan:(0,0,0)})

	fig, axs = plt.subplots(4, 2, figsize=(20,15))
	fig.suptitle(f'{name} - Stations: {str(stations)[1:-1]}', fontsize=25)
	for i, param in zip(range(1,9,2),params):

		ax = plt.subplot(4,2,i)
		plt.ylabel(param, fontsize=15)
		for col, stat in zip(colors, stations):
			if i ==1:
				plt.plot(region[stat]['statistics'][param], label=f'{stat} {np.round(np.log10(region[stat]["statistics"]["count"].sum()), 1)}', color=col)
			else:
				plt.plot(region[stat]['statistics'][param], label=stat, color=col)
		plt.xlabel('MLT')
		plt.legend()
		plt.margins(x=0)

		ax = plt.subplot(4,2,i+1)
		plt.ylabel(param, fontsize=15)
		plt.plot(region['rsd_stats'][param])
		plt.xlabel('MLT')
		plt.legend()
		plt.margins(x=0)

	plt.savefig(f'plots/{name}_twins_map_avail_segmented_stats.png')
	plt.close()
	gc.collect()


def main():

	with open('outputs/twins_time_segmented_data_and_stats.pkl', 'rb') as f:
		segmented = pickle.load(f)

	for region in segmented.keys():
		plotting(segmented[region], region)


if __name__ == '__main__':
	main()

