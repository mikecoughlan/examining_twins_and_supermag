import gc
import os

import scipy
from scipy.io import netcdf_file

os.environ["CDF_LIB"] = "~/CDF/lib"
import datetime
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacepy
from spacepy import pycdf
from tqdm import tqdm

dir = '../data/twins/'
file = 'twins_m2_ena_20090721_v02.cdf'
filepath = os.path.join(dir, file)


twins_files = sorted(glob.glob(dir+'*.cdf', recursive=True))
batch_size = 32
time_threshold = pd.Timedelta(minutes=9.5)

dates = []

for file in twins_files:
	twins_map = pycdf.CDF(file)
	for date in twins_map['Epoch']:
		dates.append(date.strftime(format='%Y-%m-%d %H:%M:%S'))


df = pd.DataFrame({'dates':dates})
df['dates'] = pd.to_datetime(df['dates'])

expected_intervals = pd.Timedelta(minutes=9, seconds=58)
over_intervals = pd.Timedelta(minutes=10, seconds=30)

irr_dates = []
i = 1
while i < len(df)-1:

	foreward_difference = (df['dates'].iloc[i+1] - df['dates'].iloc[i])
	backward_difference = (df['dates'].iloc[i] - df['dates'].iloc[i-1])

	if (foreward_difference < expected_intervals) & (backward_difference < expected_intervals) \
		|(foreward_difference < expected_intervals) & (backward_difference > over_intervals) \
		|(foreward_difference > over_intervals) & (backward_difference < expected_intervals):

		if ((df['dates'].iloc[i].minute % 10 == 0) & (df['dates'].iloc[i].second == 00)) \
			| ((df['dates'].iloc[i].minute % 10 == 9) & (df['dates'].iloc[i].second == 59)):
			print(df['dates'].iloc[i])
			i+=1
			continue

		irr_dates.append(df['dates'].iloc[i])
		df.drop(index=i, inplace=True)
		df.reset_index(inplace=True, drop=True)

	i+=1

irregs = pd.DataFrame({'dates': irr_dates})

regular_dates = df[~df['dates'].isin(irregs['dates'])]

regular_dates.to_feather('outputs/regular_twins_map_dates.feather')

