import gc
import glob
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspedas
import pyspedas.geopack as pygeo
import shapely
from dateutil import parser
from geopack import geopack, t89
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge
from spacepy import pycdf

os.environ["CDF_LIB"] = "~/CDF/lib"

twins_dir = '../data/twins/'
supermag_dir = '../data/supermag/'
regions_dict = '../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = '../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'

region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]


def loading_dicts():
	'''
	Loads the regional dictionaries and stats dictionaries

	Returns:
		regions (dict): dictionary containing the regional dictionaries
		stats (dict): dictionary containing the regional stats dictionaries including rsd and mlt data
	'''

	print('Loading regional dictionaries....')

	with open(regions_dict, 'rb') as f:
		regions = pickle.load(f)

	regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in region_numbers}

	with open(regions_stat_dict, 'rb') as g:
		stats = pickle.load(g)

	stats = {f'region_{reg}': stats[f'region_{reg}'] for reg in region_numbers}

	return regions, stats


def loading_twins_maps():


	print('Loading twins maps....')
	times = pd.read_feather('outputs/regular_twins_map_dates.feather')
	twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

	maps = {}

	for file in twins_files:
		twins_map = pycdf.CDF(file)
		for i, date in enumerate(twins_map['Epoch']):
			if len(np.unique(twins_map['Ion_Temperature'][i][50:140,40:100])) == 1:
				continue
			check = pd.to_datetime(date.strftime(format='%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
			if check in times.values:
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')] = {}
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')]['map'] = twins_map['Ion_Temperature'][i][35:125,40:130]

	return maps


def loading_solarwind():

	print('Loading solar wind data....')
	df = pd.read_feather('../data/SW/ace_data.feather')
	df.set_index('ACEepoch', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')

	return df


def loading_supermag(station):

	print(f'Loading station {station}....')
	df = pd.read_feather(supermag_dir+station+'.feather')

	# limiting the analysis to the nightside
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')

	return df

def combining_regional_dfs(stations, rsd, map_keys, delay=None):


	print('Combining regional data....')
	start_time = pd.to_datetime('2009-07-20')
	end_time = pd.to_datetime('2017-12-31')
	twins_time_period = pd.date_range(start=start_time, end=end_time, freq='min')

	combined_stations = pd.DataFrame(index=twins_time_period)

	for station in stations:
		stat = loading_supermag(station)
		stat = stat[start_time:end_time]
		stat = stat[['dbht']]
		if delay:
			stat[f'{station}_delay_{delay}'] = stat['dbht'].shift(-delay)
			combined_stations = pd.concat([combined_stations, stat[f'delay_{delay}']], axis=1, ignore_index=False)
		else:
			stat[f'{station}_dbdt'] = stat['dbht']
			combined_stations = pd.concat([combined_stations, stat[f'{station}_dbdt']], axis=1, ignore_index=False)

	mean_dbht = combined_stations.mean(axis=1)
	max_dbht = combined_stations.max(axis=1)

	combined_stations['reg_mean'] = mean_dbht
	combined_stations['reg_max'] = max_dbht
	combined_stations['rsd'] = rsd['max_rsd']['max_rsd']
	combined_stations['MLT'] = rsd['max_rsd']['MLT']

	segmented_df = combined_stations[combined_stations.index.isin(map_keys)]

	return segmented_df


def get_seconds(dt):
	'''
	Converts a date and time into seconds from the 1970's

	Args:
		dt (string): date and time of interest

	Returns:
		ut (float): time in seconds from the 1970's
	'''
	t0 = datetime(1970,1,1)
	t1 = parser.parse(dt)
	ut = (t1-t0).total_seconds()

	return ut


def get_footpoint(xx=None, yy=None, zz=None, x_gsm=None, y_gsm=None, z_gsm=None, vx=None, vy=None, vz=None, ut=None, dt=None):
	'''
	# Trace field line from (x,y,z) to equatorial plane and locates teh footpoint

	Args:
		xx (np.array): x component of the field line in GSM
		yy (np.array): y component of the field line in GSM
		zz (np.array): z component of the field line in GSM
		x_gsm (float): x component of the station in GSM
		y_gsm (float): y component of the station in GSM
		z_gsm (float): z component of the station in GSM
		vx (float): x component of the solar wind velocity in GSE used to calculate the dipole tilt angle
		vy (float): y component of the solar wind velocity in GSE used to calculate the dipole tilt angle
		vz (float): z component of the solar wind velocity in GSE used to calculate the dipole tilt angle
		ut (float): time in seconds from the 1970's
		dt (string): date and time of interest

	Returns:
		xf (float): x component of the footpoint in GSM
		yf (float): y component of the footpoint in GSM
		zmin (float): z component of the footpoint in GSM (should be zero or close to zero)
	'''
	if x_gsm:
		# Calculate dipole tilt angle
		if dt:
			ut = get_seconds(dt)
		ps = geopack.recalc(ut, vxgse=vx, vygse=vy, vzgse=vz)
		# Calculate field line (both directions)
		x,y,z,xx,yy,zz = geopack.trace(x_gsm,y_gsm,z_gsm,dir=1,rlim=21,r0=.99999,
									parmod=2,exname='t89',inname='igrf',maxloop=1000)

    # Check that field lines start and terminate at Earth
	if (abs(xx[0]) > 1):
		print(f'Field line failed to terminate at Earth. UT: {ut}')

    # Find index where field line goes closest to z=0 and its value
	idx = np.argmin(np.abs(zz))

	# Return index where field line goes closest to z=0 and its value
	zmin = zz[idx]
	xf = xx[idx]
	yf = yy[idx]

	return xf, yf, zmin


def field_line_tracing(date, geolat, geolon, vx, vy, vz):
	'''
	Ties together all the field line tracing elements and gets
	the footpoints for a given station at a given time.

	Args:
		date (string): date and time of interest
		geolat (float): geographic latitude of station
		geolon (float): geographic longitude of station
		vx (float): x component of the solar wind velocity in GSE used to calculate the dipole tilt angle
		vy (float): y component of the solar wind velocity in GSE used to calculate the dipole tilt angle
		vz (float): z component of the solar wind velocity in GSE used to calculate the dipole tilt angle

	Returns:
		xf (float): x component of the footpoint in GSM
		yf (float): y component of the footpoint in GSM
		zmin (float): z component of the footpoint in GSM (should be zero or close to zero)
	'''
	# getting time in seconds from date in the 1970's. Seems like a silly way to do this but that's the requirement.
	ut = get_seconds(date)

	# Getting the dipole tile angle
	ps = geopack.recalc(ut, vxgse=vx, vygse=vy, vzgse=vz)

	# convert degrees to radians
	lat_rad = np.deg2rad(geolat)
	lon_rad = np.deg2rad(geolon)
	# print(lat_rad, lon_rad)

	# Convert Geodetic to geocentric spherical
	r, theta_rad = geopack.geodgeo(0, lat_rad, 1)
	# print(r, theta_rad, lon_rad)

	# Converting Geocentric Spherical to Geocentric Cartesian
	x_gc, y_gc, z_gc = geopack.sphcar(1, theta_rad, lon_rad, 1)
	# print('GC:  ', x_gc,y_gc,z_gc,' R=',np.sqrt(x_gc**2+y_gc**2+z_gc**2))

	# Convert Geocentric Cartesian to GSM
	x_gsm, y_gsm, z_gsm = geopack.geogsm(x_gc, y_gc, z_gc, 1)
	# print('GSM: ', x_gsm,y_gsm,z_gsm,' R=',np.sqrt(x_gsm**2+y_gsm**2+z_gsm**2))

	# perfroming the trace
	x, y, z, xx, yy, zz = geopack.trace(x_gsm, y_gsm, z_gsm, dir=1, rlim=1000, r0=.99999, parmod=2, exname='t89', inname='igrf', maxloop=10000)

	# getting the footpoints in the equatorial plane
	xf, yf, zmin = get_footpoint(xx=xx, yy=yy, zz=zz)
	print(f'Footprints: {xf}, {yf}, {zmin}')

	return {'xf':xf, 'yf':yf, 'zmin':zmin}


def preparing_region_footpoints_for_plotting(region_df, footpoints):
	'''
	putting all the footpoints, dbdt, station names, and mean subtracted dbdt info into a dataframe
	for plotting. Also converting the footpoint coordinates to the correct scale based on how the
	twins maps are stored in basic np.arrays. The additions to the footpoint coordinates would have
	to be adjusted if the twins maps were trimmed differently. The additions coorespond to the
	x and y locations of Earth in the twins maps. The zmin values are not adjusted because this is
	in Re and just used for reference. The plotting_color is the mean subtracted dbdt value for the
	station at the time of the twins map. This is done to garner some insight into the relationship
	between the footpoint location and the station's impact on the RSD.

	Args:
		region_df (pd.dataframe): dataframe containing the regional dbdt, rsd, and mlt data for a given date
		footpoints (dict): dictionary containing the footpoint locations for each station in the region

	Returns:
		scatter_plotting_df (pd.dataframe): dataframe containing the footpoint locations, dbdt, station names,
											mean subtracted dbdt values
	'''

	print('Preparing footpoints for plotting....')
	scatter_plotting_df = pd.DataFrame(columns=['xf', 'yf', 'zmin', 'station', 'dbdt', 'plotting_color'])

	for station, foot in footpoints.items():

		scatter_plotting_df = scatter_plotting_df.append({'xf':((foot['xf']*2)+80), 'yf':((foot['yf']*2)+45),
														'zmin':foot['zmin'], 'station':station,
														'dbdt':region_df[f'{station}_dbdt'],
														'plotting_color':(region_df[station]-region_df['reg_mean'])},
														ignore_index=True)

	return scatter_plotting_df


def plotting_footpoints_on_twins_maps(twins_dict, region_df, date, region):
	'''
	Plots the footpoints on top of the twins maps

	Args:
		twins_dict (dict): dictionary entry containing the twins maps and footpoints
		region_df (pd.dataframe): dataframe containing the regional dbdt, rsd, and mlt data for a given date

	'''

	print('Plotting footpoints on twins maps....')
	scatter_plotting_df = preparing_region_footpoints_for_plotting(region_df[date], twins_dict[date][f'{region}_footpoints'])

	fig = plt.subplots(figsize=(20,15))
	ax1=plt.subplot(111)
	twins_cmap = plt.get_cmap('viridis')
	twins_sc = ScalarMappable(cmap=twins_cmap)
	twins_sc.set_array([])
	ax1.imshow(twins_dict[date]['map'], cmap=twins_sc, aspect='auto', origin='lower')

	# adding twins colorabar
	twins_cbar = plt.colorbar(twins_sc, ax=ax1, label='Twins')

	# plotting station footpoints
	ax2 = ax1.twinx()
	footpoint_cmap = plt.get_cmap('bwr')
	footpoint_normalize = plt.Normalize(vmin=scatter_plotting_df['plotting_color'].min(), vmax=scatter_plotting_df['plotting_color'].max())
	footpoint_sc = ScalarMappable(cmap=footpoint_cmap, norm=footpoint_normalize)
	footpoint_sc.set_array([])
	ax2.scatter(scatter_plotting_df['xf'], scatter_plotting_df['yf'], c=scatter_plotting_df['plotting_color'], s=100,
				marker=(5,1), cmap=footpoint_cmap)

	# adding annotation to the station footpoints
	for i, txt in enumerate(scatter_plotting_df['station']):
		ax2.annotate(txt, (scatter_plotting_df['xf'][i], scatter_plotting_df['yf'][i]), textcoords='offset points', xytext=(0,10), ha='center')

	# adding footpoints colorbar
	footpoint_cbar = plt.colorbar(footpoint_sc, ax=ax2, label='Mean Subtracted dB/dt')

	plt.title(f'{date} - {region} Max RSD: {np.round(region_df[date]["rsd"], 2)} MLT: {np.round(region_df[date]["MLT"], 2)}', fontsize=25)
	plt.savefig(f'plots/footpoints_on_twins_maps/{region}_{date}.png')
	plt.close()
	gc.collect()



# okay, in order I need to:
# load the twins maps
# load the regions
# load the solar wind data
# choose a region, and then for each station in that region
	# load the supermag data
	# go through the process of finding the footpoints for each time that cooresponds to a twins map
	# save the footpoints in a dictionary
# plot the footpoints on top of the data from the twins maps


def main():

	# loading all the datasets and dictonaries
	twins = loading_twins_maps()
	regions, stats = loading_dicts()
	solarwind = loading_solarwind()

	# selecting one region at a time for analysis
	region = 'region_166'
	test_Region = regions[region]
	test_Stats = stats[region]

	# getting dbdt and rsd data for the region
	region_df = combining_regional_dfs(test_Region['station'], test_Stats, twins.keys())

	# Getting the geographic coordiantes of the stations in the region
	stations_geo_locations = {}
	for station in test_Region['station']:
		df = loading_supermag(station)
		stations_geo_locations[station] = {'GEOLAT': df['GEOLAT'].mean(), 'GEOLON': df['GEOLON'].mean()}

	# getting the footpoints for each station in the region for each of
	# the twins maps and storing them in the maps dictionary
	print('Getting footpoints....')
	for date, entry in twins.items():
		if f'{region}_footpoints' in entry:
			continue
		print(f'Working on {date}')
		footpoints = {}
		for station, station_info in stations_geo_locations.items():
			footpoints[station] = field_line_tracing(date, station_info['GEOLAT'], \
														station_info['GEOLON'], solarwind.loc[date]['Vx'], \
														solarwind.loc[date]['Vy'], solarwind.loc[date]['Vz'])
		entry[f'{region}_footpoints'] = footpoints

	# saving the updated twins dictionaryx
	with open('../outputs/twins_maps_with_footpoints.pkl', 'wb') as f:
		pickle.dump(twins, f)

	# plotting the footpoints on top of the twins maps
	date = '2012-03-12 09:40:00'
	plotting_footpoints_on_twins_maps(twins, region_df, date, region)



if __name__ == '__main__':
	main()