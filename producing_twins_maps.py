import gc
import os

import scipy
from scipy.io import netcdf_file

os.environ["CDF_LIB"] = "~/CDF/lib"
import datetime
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import spacepy
from spacepy import pycdf

dir = '../data/twins/'
file = 'twins_m2_ena_20090721_v02.cdf'
filepath = os.path.join(dir, file)

twins_files = sorted(glob.glob(dir+'*.cdf', recursive=True))
batch_size = 32

for files in tqdm(range(0, len(twins_files), batch_size)):
	gc.collect()
	temps = []
	dates = []
	batch_of_files = twins_files[files:files+batch_size]
	for file in batch_of_files:
		twins_map = pycdf.CDF(file)
		temps.append(np.array(twins_map['Ion_Temperature']))
		for date in twins_map['Epoch']:
			dates.append(date.strftime(format='%Y-%m-%d %H:%M:%S'))

	temps = np.concatenate(temps, axis=0)

	vmin = np.min(temps)
	vmax = np.max(temps)

	for i in range(temps.shape[0]):
		fig = plt.figure()
		plt.imshow(temps[i])
		plt.clim(vmin, vmax)
		plt.colorbar()
		plt.xticks([])
		plt.yticks([])
		plt.title(dates[i])
		plt.savefig(f'plots/temp_maps/{dates[i]}')
		plt.close()