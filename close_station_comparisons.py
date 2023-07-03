import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def filter_data(df, mlat_min_bin, mlat_max_bin, mlt_min_bin=None, mlt_max_bin=None):
	'''
	Filter a pandas data frame to degree bins.
	'''
	if mlt_min_bin:
		df = df[(df['MLAT'] >= mlat_min_bin) & (df['MLAT'] < mlat_max_bin) & (df['MLT'] >= mlt_min_bin) & (df['MLT'] < mlt_max_bin)]
	else:
		df = df[(df['MLAT'] >= mlat_min_bin) & (df['MLAT'] < mlat_max_bin)]

	return df


def process_file(df, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin):
	'''
	Process a single feather file and return a filtered data frame for the degree bins.
	'''

	df_filtered = filter_data(df, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin)
	df_filtered.reset_index(inplace=True, drop=True)

	return df_filtered


def process_directory(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step, stations_dict):
	'''
	Process all feather files in a directory and return a list of filtered data frames for each degree bins.
	'''
	stats_df = {}
	for mlat in np.arange(mlat_min, mlat_max, mlat_step):
		if f'mlat_{mlat}' in stations_dict:
			stats_df[mlat] = {}
			for stats in stations_dict[f'mlat_{mlat}']:
				mlat_min_bin = mlat
				mlat_max_bin = mlat + mlat_step
				temp_df = pd.DataFrame()
				filepath = os.path.join(data_dir, f'{stats}.feather')
				df = load_data(filepath)
				stats_df[mlat][f'{stats}_dates'] = df.copy().dropna().Date_UTC
				for mlt in np.arange(mlt_min, mlt_max, mlt_step):
					print(f'MLAT: {mlat}' + f' MLT: {mlt}')
					mlt_min_bin = mlt
					mlt_max_bin = mlt + mlt_step
					df_filtered = process_file(df, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin)
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

def extracting_dates(data_dir, stats, mlat, mlat_step):

	start_date = pd.to_datetime('1995-01-01 00:00:00')
	end_date = pd.to_datetime('2019-12-31 23:59:00')
	time_period = pd.date_range(start=start_date, end=end_date, freq='min')
	time_period = pd.Series(range(len(time_period)), index=time_period)
	dates = []
	for stat in stats:
		df = load_data(data_dir+f'{stat}.feather')
		df = filter_data(df, mlat, mlat+mlat_step)
		df.dropna(subset=['dbht'], inplace=True)
		df.reset_index(inplace=True, drop=True)
		date = pd.DataFrame(index=df['Date_UTC'])
		date['value'] = int(1)
		temp_df = pd.merge(time_period.to_frame(), date, left_index=True, right_index=True, how='left').drop(0, axis=1)
		temp_df.fillna(0, inplace=True)
		dates.append(temp_df)

	return dates

def getting_solar_cycle():

	solar = pd.read_json('../data/observed-solar-cycle-indices.json')
	solar['time-tag'] = pd.to_datetime(solar['time-tag'])
	solar.set_index(solar['time-tag'], inplace=True)

	return solar

def getting_geo_location(stat, geo_df):

	temp_df = geo_df[geo_df['IAGA'] == stat]
	lat = temp_df['GEOLAT']
	lon = temp_df['GEOLON']

	return lat, lon


def plotting(stats, mlat, mlat_step, data_dir, solar, geo_df):
	'''
	plots a heatmap of a particular parameter using imshow. First transforms the data frame into a 2d array for plotting

	Args:
		stats (pd.df): dataframe containing the locations and values
	'''
	params = ['mean', 'median', 'max', 'std']
	twins_start = pd.to_datetime('2010-01-01')
	twins_end = pd.to_datetime('2017-12-31')
	start_date = pd.to_datetime('1995-01-01')
	end_date = pd.to_datetime('2019-12-31')
	twins_period = pd.date_range(start=twins_start, end=twins_end)
	twins_period = pd.Series(range(len(twins_period)), index=twins_period)

	# xticks = [0, 24, 48, 72, 95]
	# xtick_labels = [0, 6, 12, 18, 24]

	color_map = sns.color_palette('tab20', len(stats))

	# fig = plt.figure(figsize=(20,15))
	fig, axs = plt.subplots(4, 2, figsize=(20,15))
	fig.suptitle(f'MLAT: {mlat} - Stations: {str(list(stats.keys()))[1:-1]}', fontsize=25)
	for i, param in enumerate(params):

		ax = plt.subplot(4,2,i+1)
		plt.ylabel(param, fontsize=15)
		for col, stat in zip(color_map, stats):
			if i ==0:
				plt.plot(stats[stat][param], label=f'{stat} {np.round(np.log10(stats[stat]["count"].sum()), 1)}', color=col)
			else:
				plt.plot(stats[stat][param], label=stat, color=col)
		plt.xlabel('MLT')
		# plt.xticks(xticks, labels=xtick_labels)
		plt.legend()
		plt.margins(x=0)
	ax = plt.subplot(4,1,3)

	plt.xlim(start_date, end_date)
	dates = extracting_dates(data_dir, stats, mlat, mlat_step)
	for j, (col, date, stat) in enumerate(zip(color_map, dates, stats)):
		plt.fill_between(date.index, (j+0.1)*date['value'], (j+1)*date['value'], color=col, alpha=0.7, label=stat)
		plt.yticks([])

	plt.title('data availability')
	ax2 = ax.twinx()
	plt.fill_between(twins_period.index, 0, solar['smoothed_ssn'].max(), color='black', alpha=0.2, label='TWINS period')
	ax2.plot(solar['smoothed_ssn'], color='black', label='Solar Cycle')
	plt.margins(y=0)
	plt.yticks([])
	plt.legend()

	ax = plt.subplot(4,1,4)
	plt.title('Station locations')
	plt.xlim(geo_df['GEOLON'].min()-5, geo_df['GEOLON'].max()+5)
	plt.ylim(geo_df['GEOLAT'].min()-5, geo_df['GEOLAT'].max()+5)
	plt.xlabel('geolon')
	plt.ylabel('geolat')
	for col, stat in zip(color_map, stats):
		lat, lon = getting_geo_location(stat, geo_df)
		plt.scatter(lon, lat, color=col, s=70)

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
		with open(f'outputs/stats_dict_{mlat_step}_stats.pkl', 'rb') as o:
			stats_dict = pickle.load(o)

	solar = getting_solar_cycle()
	start_date = pd.to_datetime('1995-01-01')
	end_date = pd.to_datetime('2019-12-31')
	solar = solar[(solar.index > start_date) & (solar.index < end_date)]

	geo_df = pd.read_csv('supermag-stations-info.csv')

	for mlat in tqdm(stats_dict.keys()):
		# Plot the results
		plotting(stats_dict[mlat], mlat, mlat_step, data_dir, solar, geo_df)



if __name__ == '__main__':
	main()
