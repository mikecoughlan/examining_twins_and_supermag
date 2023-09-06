import glob
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspedas
import pyspedas.geopack as pygeo
from dateutil import parser
from geopack import geopack, t89
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge
from spacepy import pycdf

os.environ["CDF_LIB"] = "~/CDF/lib"

twins_dir = '../../data/twins/'
supermag_dir = '../../data/supermag/'
regions_dict = '../../identifying_regions/outputs/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = '../../identifying_regions/outputs/twins_era_stats_dict_radius_regions_min_2.pkl'

region_numbers = [83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
						387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
						62, 327, 293, 241, 107, 55, 111]


def loading_dicts():

	with open(regions_dict, 'rb') as f:
		regions = pickle.load(f)

	regions = {f'region_{reg}': regions[f'region_{reg}'] for reg in region_numbers}

	with open(regions_stat_dict, 'rb') as g:
		stats = pickle.load(g)

	stats = {f'region_{reg}': stats[f'region_{reg}'] for reg in region_numbers}

	return regions, stats


def loading_twins_maps():

	times = pd.read_feather('../outputs/regular_twins_map_dates.feather')
	twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

	maps = {}

	for file in twins_files:
		twins_map = pycdf.CDF(file)
		for i, date in enumerate(twins_map['Epoch']):
			if len(np.unique(twins_map['Ion_Temperature'][i][50:140,40:100])) == 1:
				continue
			check = pd.to_datetime(date.strftime(format='%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
			if check in times.values:
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')] = twins_map['Ion_Temperature'][i][50:140,40:100]

	return maps


def loading_solarwind():

	df = pd.read_feather('../../data/SW/ace_data.feather')
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')

	return df


def loading_supermag(station):

	df = pd.read_feather(supermag_dir+station+'.feather')

	# limiting the analysis to the nightside
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')

	return df


def dual_half_circle(center=(0,0), radius=1, angle=90, ax=None, colors=('w','k','k'),
                     **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    #w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    #w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)

    w1 = Wedge(center, radius, theta1, theta2, fc=colors[1], **kwargs)
    w2 = Wedge(center, radius, theta2, theta1, fc=colors[0], **kwargs)

    cr = Circle(center, radius, fc=colors[2], fill=False, **kwargs)
    for wedge in [w1, w2, cr]:
        ax.add_artist(wedge)

    return [w1, w2, cr]


def setup_fig(plotting_y=True, xlim=(10,-30),ylim=(-20,20)):

    fig = plt.figure(figsize=(15,10))
    ax  = fig.add_subplot(111)
    ax.axvline(0,ls=':',color='k')
    ax.axhline(0,ls=':',color='k')
    ax.set_xlabel('X GSM [Re]')
    if plotting_y:
        ax.set_ylabel('Y GSM [Re]')
    else:
        ax.set_ylabel('Z GSM [Re]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_xaxis()
    ax.set_aspect('equal')
    w1,w2,cr = dual_half_circle(ax=ax)

    return ax


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
	print('Doing field line tracing....')
	# getting time in seconds from date in the 1970's. Seems like a silly way to do this but that's the requirement.
	ut = get_seconds(date)

	# Getting the dipole tile angle
	ps = geopack.recalc(ut, vxgse=vx, vygse=vy, vzgse=vz)

	# convert degrees to radians
	lat_rad = np.deg2rad(geolat)
	lon_rad = np.deg2rad(geolon)
	print(lat_rad, lon_rad)

	# Convert Geodetic to geocentric spherical
	r, theta_rad = geopack.geodgeo(0, lat_rad, 1)
	print(r, theta_rad, lon_rad)

	# Converting Geocentric Spherical to Geocentric Cartesian
	x_gc, y_gc, z_gc = geopack.sphcar(1, theta_rad, lon_rad, 1)
	print('GC:  ', x_gc,y_gc,z_gc,' R=',np.sqrt(x_gc**2+y_gc**2+z_gc**2))

	# Convert Geocentric Cartesian to GSM
	x_gsm, y_gsm, z_gsm = geopack.geogsm(x_gc, y_gc, z_gc, 1)
	print('GSM: ', x_gsm,y_gsm,z_gsm,' R=',np.sqrt(x_gsm**2+y_gsm**2+z_gsm**2))

	# perfroming the trace
	x, y, z, xx, yy, zz = geopack.trace(x_gsm, y_gsm, z_gsm, dir=1, rlim=1000, r0=.99999, parmod=2, exname='t89', inname='igrf', maxloop=10000)

	# getting the footpoints in the equatorial plane
	xf, yf, zmin = get_footpoint(xx=xx, yy=yy, zz=zz)
	print(f'Footprints: {xf}, {yf}, {zmin}')

	return {'xf':xf, 'yf':yf, 'zmin':zmin}


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
	region = 'region_83'
	test_Region = regions[region]
	test_Stats = stats[region]

	# Getting the geographic coordiantes of the stations in the region
	stations_geo_locations = {}
	for station in test_Region['stations']:
		df = loading_supermag(station)
		stations_geo_locations[station] = {'GEOLAT': df['GEOLAT'].mean(), 'GEOLON': df['GEOLON'].mean()}
	
	# getting the footpoints for each station in the region for each of 
	# the twins maps and storing them in the maps dictionary
	for date, entry in twins.items():
		print(f'Working on {date}')
		footpoints = {}
		for station, station_info in stations_geo_locations.items():
			footpoints[station] = field_line_tracing(date, station_info['GEOLAT'], \
														station_info['GEOLON'], solarwind.loc[date]['Vx'], \
														solarwind.loc[date]['Vy'], solarwind.loc[date]['Vz'])
		entry[f'{region}_footpoints'] = footpoints
	




if __name__ == '__main__':
	main()