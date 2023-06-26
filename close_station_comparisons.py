import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the directory containing the CSV files
data_dir = '../data/supermag/'

# Define the degree grid
mlat_min = -90
mlat_max = 90
mlt_min = 0
mlt_max = 24
mlat_step = 1
mlt_step = (1/60)

def creating_dict_of_stations(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step):
	'''
	Creates a dictonary of which stations are in each grid. Used so we don't have to loop over each file every time only once.
	'''
	stations_dict = {}
	for mlat in tqdm(np.arange(mlat_min, mlat_max, mlat_step)):
		stats = []
		mlat_min_bin = mlat
		mlat_max_bin = mlat + mlat_step
		for filename in glob.glob(data_dir+'*.feather', recursive=True):
			df = pd.read_feather(filename)
			if df['MLAT'].between(mlat_min_bin, mlat_max_bin, inclusive='left').any():
				file_name = os.path.basename(filename)
				station = file_name.split('.')[0]
				stats.append(station)

		if not stats:
			stations_dict[f'mlat_{mlat}'] = stats

	with open(f'outputs/stations_dict_{mlat_step}_MLAT.pkl', 'wb') as f:
		pickle.dump(stations_dict, f)

	return stations_dict


def load_data(filepath):
	'''
	Load a feather file into a pandas data frame.
	'''
	df = pd.read_feather(filepath)

	return df


def filter_data(df, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin):
	'''
	Filter a pandas data frame to degree bins.
	'''
	df = df[(df['MLAT'] >= mlat_min_bin) & (df['MLAT'] < mlat_max_bin) & (df['MLT'] >= mlt_min_bin) & (df['MLT'] < mlt_max_bin)]

	return df


def process_file(filepath, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin):
	'''
	Process a single feather file and return a filtered data frame for the degree bins.
	'''
	df = load_data(filepath)
	df_filtered = filter_data(df, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin)
	df_filtered.reset_index(inplace=True, drop=True)

	return df_filtered


def process_directory(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step, stations_dict):
	'''
	Process all feather files in a directory and return a list of filtered data frames for each degree bins.
	'''
	stats_df = {}
	for mlat in np.arange(mlat_min, mlat_max, mlat_step):
		stats_df[mlat] = {}
		for stats in stations_dict[f'mlat_{mlat}']:
			mlat_min_bin = mlat
			mlat_max_bin = mlat + mlat_step
			temp_df = pd.DataFrame()
			for mlt in np.arange(mlt_min, mlt_max, mlt_step):
				print(f'MLAT: {mlat}' + f' MLT: {mlt}')
				mlt_min_bin = mlt
				mlt_max_bin = mlt + mlt_step
				filepath = os.path.join(data_dir, f'{stats}.feather')
				df_filtered = process_file(filepath, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin)
				if not df_filtered.empty:
					stat = compute_statistics(df_filtered, mlt)
					temp_df = pd.concat([temp_df, stat], axis=0, ignore_index=True)

			temp_df.set_index("MLT", inplace=True)
			stats_df[mlat][stats] = temp_df

	return stats_df


def compute_statistics(df_combined, mlt):
	'''
	Compute the statistics of the 'dbht' parameter for each degree bins.
	'''

	df_combined = df_combined[df_combined['dbht'].notna()]
	stats_df = pd.DataFrame({'MLT': mlt,
							'count':len(df_combined),
							'mean': df_combined['dbht'].mean(),
							'median':df_combined['dbht'].median(),
							'std': df_combined['dbht'].std(),
							'max':df_combined['dbht'].max()},
							index=[0])

	return stats_df

def plotting(stats, mlat):
	'''
	plots a heatmap of a particular parameter using imshow. First transforms the data frame into a 2d array for plotting

	Args:
		stats (pd.df): dataframe containing the locations and values
	'''

	params = ['mean', 'median', 'max', 'std' ,'count']

	xticks = [0, 24, 48, 72, 95]
	xtick_labels = [0, 6, 12, 18, 24]

	fig = plt.figure(figsize=(20,15))
	plt.title(f'MLAT {mlat}')

	for i, param in enumerate(params):

		ax = plt.subplot(2,3,i+1)
		for stat in stats:
			plt.plot(stats[stat][param], label=stat)
		plt.xlabel('MLT')
		plt.ylabel(param)
		plt.xticks(xticks, labels=xtick_labels)
		plt.legend()

	plt.savefig(f'plots/station_comparison_mlat_{mlat}.png')


def main():
	# Process the directory of feather files and compute the statistics for each 5 degree bin
	if not os.path.exists(f'outputs/stations_dict_{mlat_step}_MLAT.pkl'):
		stations_dict = creating_dict_of_stations(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step)
	else:
		with open(f'outputs/stations_dict_{mlat_step}_MLAT.pkl', 'rb') as f:
			stations_dict = pickle.load(f)

	if not os.path.exists(f'outputs/stats_dict_{mlat_step}_stats.pkl'):
		stats_dict = process_directory(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step, stations_dict)
		# stats = compute_statistics(data_frames)

		with open(f'outputs/stats_dict_{mlat_step}_stats.pkl', 'wb') as s:
			pickle.dump(stats_dict, s)

	else:
		with open(f'outputs/stats_dict_{mlat_step}_stats.pkl', 'rb') as s:
			stats_dict = pickle.load(s)

	for mlat in stats_dict.keys():
		# Plot the results
		plotting(stats_dict[mlat], mlat)



if __name__ == '__main__':
	main()
