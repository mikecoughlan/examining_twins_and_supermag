import warnings

import matplotlib.pyplot as plt
import numpy as nu
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')
import gc
import glob
import os

data_dir = '../../../../data/'
twins_dir = '../data/twins/'
supermag_dir = data_dir+'supermag/feather_files/'

station_files = glob.glob(supermag_dir+'*.feather')

# read in the station data

def load_supermag(station_file):

	station = pd.read_feather(station_file)
	station.set_index('Date_UTC', inplace=True, drop=True)
	station['dN'] = station['N'].diff(1)
	station['dE'] = station['E'].diff(1)
	station.index = pd.to_datetime(station.index, format='%Y-%m-%d %H:%M:$S')

	return station

def creating_time_step_dict(station_files, start_time, end_time):

	# creating a dictionary with the keys being the time steps and the values being empty dataframes
	times_dict = {time.strftime('%Y-%m-%d %H:%M:%S'): pd.DataFrame() for time in pd.date_range(start_time, end_time, freq='1min')}

	for station_file in tqdm(station_files):

		# laoding supermag file
		station = load_supermag(station_file)
		station = station[start_time:end_time]
		# checking to see if the sataion has any data in the time range
		if station.empty:
			continue

		# adding the row corresponding to the time step to the dataframe in the dictonary
		for time in times_dict.keys():
			times_dict[time] = pd.concat([times_dict[time], station.loc[[time]]], axis=0, ignore_index=True)

	return times_dict

def plotting_time_series(times_dict, stime, geo_or_mag='mag', color_var='dbht'):

	hlines = [60, 30, 0, -30, -60]
	val_max = max([times_dict[time][color_var].max() for time in times_dict.keys()])
	val_min = min([times_dict[time][color_var].min() for time in times_dict.keys()])
	vmax = max(abs(val_max), abs(val_min))
	vmin = -vmax

	# creating a scatter plot for each time step in the dictornay with the color and size of the points corresponding to the value of the color_var
	for time in times_dict.keys():
		fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
		if geo_or_mag == 'geo':
			scatter = axes.scatter(times_dict[time]['GEOLON'], times_dict[time]['GEOLAT'], s=times_dict[time][color_var], c=times_dict[time][color_var], cmap='coolwarm', vmax=vmax, vmin=vmin)
			vlines = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
			axes.set_xlabel('GEOLON')
			axes.set_ylabel('GEOLAT')
		elif geo_or_mag == 'mag':
			scatter = axes.scatter(times_dict[time]['MLT'], times_dict[time]['MLAT'], s=times_dict[time][color_var], c=times_dict[time][color_var], cmap='coolwarm', vmax=vmax, vmin=vmin)
			vlines = [0, 6, 12, 18]
			axes.set_xlabel('MLT')
			axes.set_ylabel('MLAT')
		for hline in hlines:
			axes.axhline(y=hline, color='k', linestyle='--', alpha=0.5)
		for vline in vlines:
			axes.axvline(x=vline, color='k', linestyle='--', alpha=0.5)
		cbar = axes.figure.colorbar(scatter, ax=axes)
		cbar.set_label(color_var, fontsize=12)
		axes.set_title(time, fontsize=15)
		axes.set_xticks(vlines)
		axes.set_yticks(hlines)
		axes.set_xlim(0, 24)
		axes.margins(x=0)
		plt.savefig(f'plots/{stime}/{geo_or_mag}_{time}.png')
		plt.close()
		gc.collect()


def main():

	start_time = '2017-01-07 04:00:00'
	end_time = '2017-01-07 07:00:00'

	if not os.path.exists(f'plots/{start_time}'):
		os.makedirs(f'plots/{start_time}')

	times_dict = creating_time_step_dict(station_files, start_time, end_time)

	plotting_time_series(times_dict, start_time, geo_or_mag='mag', color_var='dN')


if __name__ == '__main__':
	main()