import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils

# loading the region data
with open('../../../../data/mike_working_dir/identifying_regions_data/adjusted_regions.pkl', 'rb') as f:
		regions = pickle.load(f)
with open('../../../../data/mike_working_dir/identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl', 'rb') as f:
		stats = pickle.load(f)


data_dir = '../../../../data/'
working_dir = data_dir + 'mike_working_dir/rsd_stats/'
twins_dir = '../data/twins/'
supermag_dir = data_dir+'supermag/feather_files/'

# loading the solar wind data
omni = pd.read_feather('../data/SW/omniData.feather')
omni = omni.set_index('Epoch', drop=True)
omni.index = pd.to_datetime(omni.index)

storm_list = pd.read_csv('StormList_types.csv')
storm_list['initial_phase'] = pd.to_datetime(storm_list['initial_phase'])
storm_list['main_phase'] = pd.to_datetime(storm_list['main_phase'])
storm_list['minimumSymH'] = pd.to_datetime(storm_list['minimumSymH'])
storm_list['end_recovery_phase'] = pd.to_datetime(storm_list['end_recovery_phase'])
storm_list['new_begining_times'] = pd.to_datetime(storm_list['new_begining_times'])
storm_list['new_ending_times'] = pd.to_datetime(storm_list['new_ending_times'])


recalculate_regions = False

clusters = {
	'canadian_cluster':{
		'regions':{
		'CAN-0': {'stations':['NEW', 'T19', 'C10', 'LET', 'T03', 'T43']},
		'CAN-1': {'stations':['LET', 'T03', 'T43', 'RED' , 'C06']},
		'CAN-2': {'stations':['T43', 'RED', 'C06', 'MEA', 'T36']}
		}
	},
	'greenland_cluster':{
		'regions':{
		'GRL-0': {'stations':['GHB', 'SKT', 'STF', 'ATU']},
		'GRL-1': {'stations':['SKT', 'STF', 'ATU', 'GDH']},
		'GRL-2': {'stations':['STF', 'ATU', 'GDH', 'UMQ']},
		'GRL-3': {'stations':['GHB', 'FHB', 'NAQ']},
		}
	},
	'fennoscandinavian_cluster':{
		'regions':{
		'FSC-0': {'stations':['RVK', 'LYC', 'DON', 'JCK']},
		'FSC-1': {'stations':['HAN', 'MEK', 'OUJ', 'NUR']},
		'FSC-2': {'stations':['MAS', 'NOR', 'IVA', 'KEV', 'KIL', 'MUO', 'SOR', 'TRO', 'ABK', 'KIR']},
		'FSC-3': {'stations':['MAS', 'AND', 'KIL', 'MUO', 'SOR', 'TRO', 'ABK', 'KIR']},
		'FSC-4': {'stations':['MAS', 'SOD', 'IVA', 'KEV', 'KIL', 'MUO', 'ABK', 'KIR', 'PEL']},
		'FSC-5': {'stations':['JCK', 'DON', 'ABK', 'KIR', 'LYC']},
		'FSC-6': {'stations':['MAS', 'AND', 'KIL', 'MUO', 'JCK', 'TRO', 'ABK', 'KIR', 'PEL']},
		}
	},
	'central_european_cluster':{
		'regions':{
		'CEU-0': {'stations':['ZAG', 'LVV', 'BEL', 'VYH']},
		'CEU-1': {'stations':['BEL', 'HLP', 'SZC', 'KLD']},
		'CEU-2': {'stations':['THY', 'BDV', 'WIC', 'NCK', 'HRB']},
		'CEU-3': {'stations':['ROE', 'BFE', 'WNG']},
		}
	},
	'non_cluster_regions':{
		'regions':{
		'SVLB': {'stations':['BBG', 'LYR', 'HOR', 'NAL', 'HRN', 'HOP']},
		'JPN-0': {'stations':['KUJ', 'KNY', 'KAG']},
		'JPN-1': {'stations':['MMB', 'ASB', 'RIK', 'MSR']},
		'ALSK': {'stations':['CMO', 'FYU', 'PKR', 'GAK']},
		'HUD-0': {'stations':['PIN', 'ISL', 'C05']},
		'HUD-1': {'stations':['FCC', 'EKP', 'RAN', 'BLC']},
		}
	}
}

regions = {**clusters['greenland_cluster']['regions'], **clusters['canadian_cluster']['regions'],
			**clusters['fennoscandinavian_cluster']['regions'], **clusters['central_european_cluster']['regions'],
			**clusters['non_cluster_regions']['regions']}


def getting_dbdt_dataframe(region):

	dbdt_df = pd.DataFrame(index=pd.date_range(start='2009-07-20', end='2017-12-31', freq='min'))
	for station in region['stations']:
		# loading the station data
		station_df = pd.read_feather(supermag_dir + station + '.feather')
		station_df.set_index('Date_UTC', inplace=True)
		station_df.index = pd.to_datetime(station_df.index)
		# cutting the data to the region time frame
		station_df = station_df[pd.to_datetime('2009-07-20'):pd.to_datetime('2017-12-31')]
		# creating the dbdt time series
		dbdt_df[station] = station_df['dbht']

	return dbdt_df

def calculating_rsd(region):

	if 'dbdt_df' not in region.keys():
		region['dbdt_df'] = getting_dbdt_dataframe(region)

	dbdt_df = region['dbdt_df']
	rsd = pd.DataFrame(index=dbdt_df.index)
	# calculating the RSD
	for col in dbdt_df.columns:
		ss = dbdt_df[col]
		temp_df = dbdt_df.drop(col,axis=1)
		ra = temp_df.mean(axis=1)
		rsd[col] = ss-ra
	max_rsd = rsd.max(axis=1)
	max_station = rsd.idxmax(axis=1)
	rsd['max_rsd'] = max_rsd
	rsd['max_station'] = max_station

	region['rsd_df'] = rsd

	return rsd


if not (os.path.exists(working_dir+'rsd_data.pkl')) or recalculate_regions:
	for region in regions.values():
		region['rsd_df'] = calculating_rsd(region)
		region['mean_lat'] = utils.getting_mean_lat(region['stations'])
	with open(working_dir+'rsd_data.pkl', 'wb') as f:
		pickle.dump(regions, f)
else:
	with open(working_dir+'rsd_data.pkl', 'rb') as f:
		regions = pickle.load(f)


all_stations = []
for region in regions:
	new_stations = [station for station in regions[region]['stations'] if station not in all_stations]
	all_stations += new_stations


stations = {}
for station in all_stations:
	df = pd.read_feather(supermag_dir+station+'.feather')
	df.set_index('Date_UTC', drop=True, inplace=True)
	df.index = pd.to_datetime(df.index)
	df = df['2009-07-20':'2017-12-31']
	stations[station] = {'geolat': df['GEOLAT'].mean()}
	df.drop(columns=['SZA', 'dbn_geo', 'dbe_geo', 'dbz_geo','MAGLON', 'GEOLAT', 'GEOLON'], inplace=True)
	stations[station]['df'] = df


# defining the latitude bins in degrees
lat_delimiter = 5

# setting a date range for the index
index_date_range = pd.date_range('2009-07-20', '2017-12-31', freq='min')

lat_dict = {}
for i in range(20, 90, lat_delimiter):
	print(f'lat: {i}')
	cluster_flag = False
	scandinavia, svalbard, greenland, central_europe, central_canadian = pd.DataFrame(index=index_date_range), \
																		pd.DataFrame(index=index_date_range), \
																		pd.DataFrame(index=index_date_range), \
																		pd.DataFrame(index=index_date_range), \
																		pd.DataFrame(index=index_date_range)
	lat_dict[i] = {'rsd':{'df': pd.DataFrame()}, 'dbht':{'df': pd.DataFrame()}}
	for region in regions:
		temp_df = regions[region]['rsd_df']
		if regions[region]['mean_lat'] >= i and regions[region]['mean_lat'] < i + lat_delimiter:
			# if region in scandinavia_regions:
			# 	scandinavia = pd.concat([scandinavia, temp_df['max_rsd']], axis=1, ignore_index=False)
			# 	scandinavia = scandinavia.rename(columns={'max_rsd': region})
			# 	cluster_flag = True
			# elif region in svalbard_regions:
			# 	svalbard = pd.concat([svalbard, temp_df['max_rsd']], axis=1, ignore_index=False)
			# 	svalbard = svalbard.rename(columns={'max_rsd': region})
			# 	cluster_flag = True
			# elif region in greenland_regions:
			# 	greenland = pd.concat([greenland, temp_df['max_rsd']], axis=1, ignore_index=False)
			# 	greenland = greenland.rename(columns={'max_rsd': region})
			# 	cluster_flag = True
			# elif region in central_europe_regions:
			# 	central_europe = pd.concat([central_europe, temp_df['max_rsd']], axis=1, ignore_index=False)
			# 	central_europe = central_europe.rename(columns={'max_rsd': region})
			# 	cluster_flag = True
			# elif region in central_canadian_regions:
			# 	central_canadian = pd.concat([central_canadian, temp_df['max_rsd']], axis=1, ignore_index=False)
			# 	central_canadian = central_canadian.rename(columns={'max_rsd': region})
			# 	dluster_flag = True
			# else:
			temp_df = pd.concat([temp_df, omni], axis=1, join='inner')
			lat_dict[i]['rsd']['df'] = pd.concat([lat_dict[i]['rsd']['df'], temp_df], axis=0)
	if cluster_flag:
		# finding the non-empty cluster dataframe
		clusters = [scandinavia, svalbard, greenland, central_europe, central_canadian]
		non_empty_clusters = [cluster for cluster in clusters if not cluster.empty]
		for cluster in non_empty_clusters:
			cluster['max'] = cluster.max(axis=1)
			cluster['max_region'] = cluster.idxmax(axis=1)
			cluster.dropna(inplace=True)
			print(cluster)
			# using the idmax to get the data for that timestamp from that region and combining them all into a single dataframe
			max_cluster_df = pd.DataFrame()
			for index, row in cluster.iterrows():
				max_cluster_df = pd.concat([max_cluster_df, stats[row['max_region']]['max_rsd'].loc[index]], axis=0)

			print('max cluster df')
			print(max_cluster_df)

			# adding the omni data to the max_cluster_df
			max_cluster_df = pd.concat([max_cluster_df, omni], axis=1, join='inner')
			lat_dict[i]['rsd']['df'] = pd.concat([lat_dict[i]['rsd']['df'], max_cluster_df], axis=0)

	print('Now the stations....')
	for station in tqdm(stations):
		temp_df = stations[station]['df']
		segmented_df = temp_df[(temp_df['MLAT']>=i) & (temp_df['MLAT']<i+lat_delimiter)]
		if not segmented_df.empty:
			segmented_df = pd.concat([segmented_df, omni], axis=1, join='inner')
			lat_dict[i]['dbht']['df'] = pd.concat([lat_dict[i]['dbht']['df'], segmented_df], axis=0)


hss_df = storm_list[(storm_list['ifHSS'].isnull() == False) & (storm_list['ifCME'].isnull() == True)]
cme_df = storm_list[(storm_list['ifCME'].isnull() == False) & (storm_list['ifHSS'].isnull() == True)]
complex_df = storm_list[storm_list['ifcomplex'].isnull() == False]


def perc95(x):
	return np.percentile(x, 95)

def seperating_into_bins(lat_dict, analyze, dates=None, var=None, var_bins=None):

	if analyze == 'rsd':
		analyze_var_name = 'max_rsd'
	elif analyze == 'dbht':
		analyze_var_name = 'dbht'
	else:
		raise ValueError('analyze must be either rsd or dbht')

	# writing some error handling for the inputs
	if dates is not None:
		if not isinstance(dates, pd.DataFrame):
			raise ValueError('dates must be a pandas dataframe')
		if dates.shape[1] < 2:
			raise ValueError('dates must have column for start time and end time')

	if var:
		if not isinstance(var, str):
			raise ValueError('var must be a string')
		if not isinstance(var_bins, list):
			raise ValueError('var_bins must be a list')
		if len(var_bins) < 2:
			raise ValueError('var_bins must have at least two elements')

	# creating a dataframe to store the combined data
	if var:
		titles = []
		for i in range(len(var_bins)+1):
			if i == 0:
				titles.append(f'{var}<{var_bins[i]}')
			elif i < len(var_bins):
				titles.append(f'{var} between {var_bins[i-1]} and {var_bins[i]}')
			else:
				titles.append(f'{var}>{var_bins[i-1]}')
		combined_aggs = {t: pd.DataFrame() for t in titles}
	else:
		combined_aggs = pd.DataFrame()

	for lat in lat_dict:
		if lat_dict[lat][analyze]['df'].empty:
			continue
		lat_dict[lat][analyze]['df'].dropna(inplace=True)
		lat_dict[lat][analyze]['df']['MLT'] = lat_dict[lat][analyze]['df']['MLT'].astype(int)
		lat_dict[lat][analyze]['df']['MLT'].replace(24, 0, inplace=True)

		# if a dates dataframe is included, the data will be filtered to only
		# include the dates between the starting and ending dates for each row
		# of the dataframe
		if dates is not None:
			temp_df = pd.DataFrame()
			if isinstance(lat_dict[lat][analyze]['df'].index, pd.DatetimeIndex):
				lat_dict[lat][analyze]['df'].reset_index(inplace=True, drop=False)
				lat_dict[lat][analyze]['df'].rename(columns={'index': 'Date_UTC'}, inplace=True)
			lat_dict[lat][analyze]['df']['Date_UTC'] = pd.to_datetime(lat_dict[lat][analyze]['df']['Date_UTC'])
			df = lat_dict[lat][analyze]['df']
			for index, date in dates.iterrows():
				storm = df[(df['Date_UTC'] >= date['initial_phase']) & (df['Date_UTC'] <= date['end_recovery_phase'])][[analyze_var_name, 'MLT']]
				temp_df = pd.concat([temp_df, storm], axis=0)
			temp_df = temp_df.groupby('MLT').agg(['mean', 'std', 'max', 'median', perc95])[analyze_var_name]
			temp_df['lat'] = lat
			combined_aggs = pd.concat([combined_aggs, temp_df], axis=0)

		# if a variable is included, the data will be filtered to only include
		# the rows where the variable is between the values in the var_bins.
		# this will create n+1 dataframes where n is the number of bins
		if var:
			temp_df = lat_dict[lat][analyze]['df']
			if isinstance(temp_df.index, pd.DatetimeIndex):
				temp_df.reset_index(inplace=True, drop=False)
				temp_df.rename(columns={'index': 'Date_UTC'}, inplace=True)
			for i, t in enumerate(titles):
				if i == 0:
					lat_dict[lat][t] = temp_df[temp_df[var] < var_bins[i]][[analyze_var_name, 'MLT']]
				elif i < len(var_bins):
					lat_dict[lat][t] = temp_df.loc[temp_df[var].between(var_bins[i-1], var_bins[i])][[analyze_var_name, 'MLT']]
				else:
					lat_dict[lat][t] = temp_df[temp_df[var] > var_bins[i-1]][[analyze_var_name, 'MLT']]

				lat_dict[lat][t] = lat_dict[lat][t].groupby('MLT').agg(['mean', 'std', 'max', 'median', perc95])[analyze_var_name]
				lat_dict[lat][t]['lat'] = lat
				combined_aggs[t] = pd.concat([combined_aggs[t], lat_dict[lat][t]], axis=0)



		# lat_dict[lat]['rsd']['lat'] = lat

	return lat_dict, combined_aggs

rsd_bz_lat_dict, rsd_bz_combined_aggs = seperating_into_bins(lat_dict, analyze='rsd',  var='BZ_GSM', var_bins=[-2, 2])
rsd_vx_lat_dict, rsd_vx_combined_aggs = seperating_into_bins(lat_dict, analyze='rsd', var='Vx', var_bins=[-700, -400])
rsd_hss_lat_dict, rsd_hss_combined_aggs = seperating_into_bins(lat_dict, analyze='rsd', dates=hss_df[['initial_phase', 'end_recovery_phase']])
rsd_cme_lat_dict, rsd_cme_combined_aggs = seperating_into_bins(lat_dict, analyze='rsd', dates=cme_df[['initial_phase', 'end_recovery_phase']])
rsd_complex_lat_dict, rsd_complex_combined_aggs = seperating_into_bins(lat_dict, analyze='rsd', dates=complex_df[['initial_phase', 'end_recovery_phase']])

dbht_bz_lat_dict, dbht_bz_combined_aggs = seperating_into_bins(lat_dict, analyze='dbht', var='BZ_GSM', var_bins=[-2, 2])
dbht_vx_lat_dict, dbht_vx_combined_aggs = seperating_into_bins(lat_dict, analyze='dbht', var='Vx', var_bins=[-700, -400])
dbht_hss_lat_dict, dbht_hss_combined_aggs = seperating_into_bins(lat_dict, analyze='dbht', dates=hss_df[['initial_phase', 'end_recovery_phase']])
dbht_cme_lat_dict, dbht_cme_combined_aggs = seperating_into_bins(lat_dict, analyze='dbht', dates=cme_df[['initial_phase', 'end_recovery_phase']])
dbht_complex_lat_dict, dbht_complex_combined_aggs = seperating_into_bins(lat_dict, analyze='dbht', dates=complex_df[['initial_phase', 'end_recovery_phase']])


def plot_formatting(combined_aggs, stat_value, name=None):

	# creating a meshgrid to plot the pivot table values
	mltbin = np.arange(0, 24, 1)
	latbin = np.arange(50, 85, 5)

	arr = np.asarray(mltbin)/24.*2*np.pi
	R, th = np.meshgrid(latbin, arr)

	# creating the pivot table
	if isinstance(combined_aggs, pd.DataFrame):
		plotting_dict = {name:{}}
		pivot = combined_aggs.pivot_table(index='lat', columns='MLT', values=stat_value)

		# creating a 2d array using the pivot table values and the meshgrid and filling the rest with zeros
		Z = np.zeros((len(latbin), len(mltbin)), dtype=float)
		for i, lat in enumerate(latbin):
			for mlt in mltbin:
				try:
					Z[i, mlt] = pivot.loc[lat, mlt]
				except:
					Z[i, mlt] = np.nan

		plotting_dict[name]['Z'] = Z
		plotting_dict[name]['R'] = R
		plotting_dict[name]['th'] = th
		return plotting_dict

	elif isinstance(combined_aggs, dict):
		plotting_dict = {key:{} for key in combined_aggs.keys()}
		for key in combined_aggs:
			# creating the pivot table
			pivot = combined_aggs[key].pivot_table(index='lat', columns='MLT', values=stat_value)

			# creating a 2d array using the pivot table values and the meshgrid and filling the rest with zeros
			Z = np.zeros((len(latbin), len(mltbin)), dtype=float)
			for i, lat in enumerate(latbin):
				for mlt in mltbin:
					try:
						Z[i, mlt] = pivot.loc[lat, mlt]
					except:
						Z[i, mlt] = np.nan

			plotting_dict[key]['Z'] = Z
			plotting_dict[key]['R'] = R
			plotting_dict[key]['th'] = th

	else:
		raise ValueError('combined_aggs must be a pandas dataframe or a dictionary')

	return plotting_dict



# plotting the heatmap using polar plot

rsd_plotting_dict = plot_formatting(rsd_bz_combined_aggs, 'perc95')
dbht_plotting_dict = plot_formatting(dbht_bz_combined_aggs, 'perc95')
fig, ax = plt.subplots(2, len(rsd_plotting_dict), figsize=(20,15), subplot_kw=dict(projection='polar'))
for i, key in enumerate(rsd_plotting_dict):
	Z = rsd_plotting_dict[key]['Z']
	R = rsd_plotting_dict[key]['R']
	th = rsd_plotting_dict[key]['th']
	c = ax[0,i].pcolormesh(th, R, Z.T, cmap='viridis', shading='auto')
	fig.colorbar(c, ax=ax[0,i], label='RSD')
	ax[0,i].set_rlim(bottom=85, top=50)
	ax[0,i].set_theta_zero_location('S')
	ax[0,i].set_thetagrids([theta * 45 for theta in range(360//45)], labels=[0, 3, 6, 9, 12, 15, 18, 21])
	ax[0,i].set_xlabel('MLT')
	ax[0,i].set_title(key)

for i, key in enumerate(dbht_plotting_dict):
	Z = dbht_plotting_dict[key]['Z']
	R = dbht_plotting_dict[key]['R']
	th = dbht_plotting_dict[key]['th']
	c = ax[1,i].pcolormesh(th, R, Z.T, cmap='magma', shading='auto')
	fig.colorbar(c, ax=ax[1,i], label='dB/dt')
	ax[1,i].set_rlim(bottom=85, top=50)
	ax[1,i].set_theta_zero_location('S')
	ax[1,i].set_thetagrids([theta * 45 for theta in range(360//45)], labels=[0, 3, 6, 9, 12, 15, 18, 21])
	ax[1,i].set_xlabel('MLT')
	ax[1,i].set_title(key)


plt.show()

# plotting the heatmap using polar plot

rsd_plotting_dict = plot_formatting(rsd_vx_combined_aggs, 'perc95')
dbht_plotting_dict = plot_formatting(dbht_vx_combined_aggs, 'perc95')
fig, ax = plt.subplots(2, len(rsd_plotting_dict), figsize=(20,15), subplot_kw=dict(projection='polar'))
for i, key in enumerate(rsd_plotting_dict):
	Z = rsd_plotting_dict[key]['Z']
	R = rsd_plotting_dict[key]['R']
	th = rsd_plotting_dict[key]['th']
	c = ax[0,i].pcolormesh(th, R, Z.T, cmap='viridis', shading='auto')
	fig.colorbar(c, ax=ax[0,i], label='RSD')
	ax[0,i].set_rlim(bottom=85, top=50)
	ax[0,i].set_theta_zero_location('S')
	ax[0,i].set_thetagrids([theta * 45 for theta in range(360//45)], labels=[0, 3, 6, 9, 12, 15, 18, 21])
	ax[0,i].set_xlabel('MLT')
	ax[0,i].set_title(key)

for i, key in enumerate(dbht_plotting_dict):
	Z = dbht_plotting_dict[key]['Z']
	R = dbht_plotting_dict[key]['R']
	th = dbht_plotting_dict[key]['th']
	c = ax[1,i].pcolormesh(th, R, Z.T, cmap='magma', shading='auto')
	fig.colorbar(c, ax=ax[1,i], label='dB/dt')
	ax[1,i].set_rlim(bottom=85, top=50)
	ax[1,i].set_theta_zero_location('S')
	ax[1,i].set_thetagrids([theta * 45 for theta in range(360//45)], labels=[0, 3, 6, 9, 12, 15, 18, 21])
	ax[1,i].set_xlabel('MLT')
	ax[1,i].set_title(key)

plt.show()

# plotting the heatmap using polar plot
rsd_hss_plotting_dict = plot_formatting(rsd_hss_combined_aggs, 'perc95', name='HSS')
rsd_cme_plotting_dict = plot_formatting(rsd_cme_combined_aggs, 'perc95', name='CME')
rsd_complex_plotting_dict = plot_formatting(rsd_complex_combined_aggs, 'perc95', name='Complex')
rsd_plotting_dict = {**rsd_hss_plotting_dict, **rsd_cme_plotting_dict, **rsd_complex_plotting_dict}

dbht_hss_plotting_dict = plot_formatting(dbht_hss_combined_aggs, 'perc95', name='HSS')
dbht_cme_plotting_dict = plot_formatting(dbht_cme_combined_aggs, 'perc95', name='CME')
dbht_complex_plotting_dict = plot_formatting(dbht_complex_combined_aggs, 'perc95', name='Complex')
dbht_plotting_dict = {**dbht_hss_plotting_dict, **dbht_cme_plotting_dict, **dbht_complex_plotting_dict}

fig, ax = plt.subplots(2, len(rsd_plotting_dict), figsize=(20,15), subplot_kw=dict(projection='polar'))
for i, key in enumerate(rsd_plotting_dict):
	Z = rsd_plotting_dict[key]['Z']
	R = rsd_plotting_dict[key]['R']
	th = rsd_plotting_dict[key]['th']
	c = ax[0,i].pcolormesh(th, R, Z.T, cmap='viridis', shading='auto')
	fig.colorbar(c, ax=ax[0,i], label='RSD')
	ax[0,i].set_rlim(bottom=85, top=50)
	ax[0,i].set_theta_zero_location('S')
	ax[0,i].set_thetagrids([theta * 45 for theta in range(360//45)], labels=[0, 3, 6, 9, 12, 15, 18, 21])
	ax[0,i].set_xlabel('MLT')
	ax[0,i].set_title(key)

for i, key in enumerate(dbht_plotting_dict):
	Z = dbht_plotting_dict[key]['Z']
	R = dbht_plotting_dict[key]['R']
	th = dbht_plotting_dict[key]['th']
	c = ax[1,i].pcolormesh(th, R, Z.T, cmap='magma', shading='auto')
	fig.colorbar(c, ax=ax[1,i], label='dB/dt')
	ax[1,i].set_rlim(bottom=85, top=50)
	ax[1,i].set_theta_zero_location('S')
	ax[1,i].set_thetagrids([theta * 45 for theta in range(360//45)], labels=[0, 3, 6, 9, 12, 15, 18, 21])
	ax[1,i].set_xlabel('MLT')
	ax[1,i].set_title(key)

plt.show()

# checking for common stations causing the max rsd in multiple regions
max_stations = pd.DataFrame(index=pd.date_range('2009-07-20', '2017-12-31', freq='min'))
for region in regions:
	max_stations = pd.concat([max_stations, stats[region]['max_rsd']['max_rsd_station']], axis=1, join='outer')

max_stations.columns = regions.keys()
max_stations.dropna(inplace=True, how='all')

# Getting the number of unique strings in each row of the dataframe
unique_stations = max_stations.apply(lambda row: len(row.dropna(inplace=False).unique()), axis=1)
total_stations_non_nan = max_stations.apply(lambda row: len(row.dropna(inplace=False)), axis=1)

# getting the number fo repeated stations by subtracting the number of unique stations from the total number of stations
repeated_stations = total_stations_non_nan - unique_stations

# adding the repeated stations to the max_stations dataframe
max_stations['repeated_stations'] = repeated_stations

perc_repeated_stations = repeated_stations/total_stations_non_nan*100

# taking the rolling mean of the percentage of repeated stations
perc_repeated_stations = perc_repeated_stations.rolling(window=1440, min_periods=1).mean()

fig, ax = plt.subplots(1,1, figsize=(20,7))
ax.plot(perc_repeated_stations, label='Percentage of repeated stations')
plt.show()

