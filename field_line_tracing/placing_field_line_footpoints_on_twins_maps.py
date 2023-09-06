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


def loading_supermag(station, start_time, end_time):

	df = pd.read_feather(supermag_dir+station+'.feather')

	# limiting the analysis to the nightside
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
	df = df[start_time:end_time]

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

    t0 = datetime(1970,1,1)
    t1 = parser.parse(dt)
    ut = (t1-t0).total_seconds()

    return ut


# Return index where field line goes closest to z=0 and its value
def find_nearest_z(z_arr, value):
    array = np.asarray(z_arr)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# Takes: dt, th_loc[x,y,z] (assumes get_th_xyz() has already been called)
# Returns: x,y coordinates of the traced equatorial "footpoint"
def get_footpoint(x_gsm, y_gsm, z_gsm, vx, vy, vz, dt):
    # Calculate dipole tilt angle
    ut = get_seconds(dt)
    ps = geopack.recalc(ut, vxgse=vx, vygse=vy, vzgse=vz)
    # Calculate field line (both directions)
    x,y,z,xx,yy,zz = geopack.trace(x_gsm,y_gsm,z_gsm,dir=1,rlim=21,r0=.99999,
                                   parmod=2,exname='t89',inname='igrf',maxloop=1000)
    # Check that field lines start and terminate at Earth
    if (abs(xx[0]) > 1):
        print(f'Field line failed to terminate at Earth. UT: {ut}')
    mindex, z_min = find_nearest_z(zz, 0)
    xf = xx[mindex]
    yf = yy[mindex]
    return xf, yf, z_min