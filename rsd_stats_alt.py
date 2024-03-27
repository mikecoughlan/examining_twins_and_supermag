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

# a bunch of functions

def creating_region_shape(stations):
	geolat, geolon = [], []
	# loading the station information and getting geographical coordinates
	station_df = pd.read_csv(station_info)
	for station in stations:
		geolat.append(station_df[station_df['IAGA'] == station]['GEOLAT'].values[0])
		geolon.append(station_df[station_df['IAGA'] == station]['GEOLON'].values[0])
	# transforming the coordiantas from 0-360 to -180-180
	geolon = [i if i < 180 else i-360 for i in geolon]
	# creating the region shape as a polygon for geopandas plotting
	region_shape = Polygon([(i, j) for i, j in zip(geolon, geolat)]).convex_hull
	return region_shape

def getting_dbdt_dataframe(region):

	dbdt_df = pd.DataFrame(index=pd.date_range(start='2009-07-20', end='2017-12-31', freq='min'))
	print(region)
	for station in region['stations']:
		# loading the station data
		station_df = pd.read_feather(supermag_dir + station + '.feather')
		station_df.set_index('Date_UTC', inplace=True)
		station_df.index = pd.to_datetime(station_df.index)
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

def finding_percentage_of_repeated_stations(cluster):

	if 'unique_stations' not in cluster.keys():

		max_stations = pd.DataFrame(index=pd.date_range(start='2009-07-20', end='2017-12-31', freq='min'))

		for region in cluster['regions'].keys():
			if 'rsd_df' not in cluster['regions'][region].keys():
				calculating_rsd(cluster['regions'][region])
			rsd_df = cluster['regions'][region]['rsd_df']
			max_stations = pd.concat([max_stations, rsd_df['max_station']], ignore_index=False, axis=1)

		max_stations.columns = cluster['regions'].keys()
		max_stations.dropna(inplace=True, how='all')

		# Getting the number of unique strings in each row of the dataframe
		unique_stations = max_stations.apply(lambda row: len(row.dropna(inplace=False).unique()), axis=1)
		total_stations_non_nan = max_stations.apply(lambda row: len(row.dropna(inplace=False)), axis=1)

		# getting the number fo repeated stations by subtracting the number of unique stations from the total number of stations
		repeated_stations = (unique_stations/total_stations_non_nan)

		# adding the repeated stations to the max_stations dataframe
		max_stations['perc_unique'] = repeated_stations

		cluster['unique_stations'] = max_stations

		return max_stations

def max_rsd_region_in_cluster(cluster):

	max_rsd = pd.DataFrame(index=pd.date_range(start='2009-07-20', end='2017-12-31', freq='min'))

	for region in cluster['regions'].keys():
		if 'rsd_df' not in cluster['regions'][region].keys():
			calculating_rsd(cluster['regions'][region])
		rsd_df = cluster['regions'][region]['rsd_df']
		max_rsd = pd.concat([max_rsd, rsd_df['max_rsd']], ignore_index=False, axis=1)

	max_rsd.columns = cluster['regions'].keys()
	max_rsd.dropna(inplace=True, how='all')
	maximum = max_rsd.max(axis=1)
	max_region = max_rsd.idxmax(axis=1)
	max_rsd['max_rsd'], max_rsd['max_reion'] = maximum, max_region

	cluster['max_rsd'] = max_rsd

	return max_rsd


# region_numbers = [387, 61, 202, 287, 207, 361, 137, 184, 36, 19, 9, 163, 16, 270, 194, 82,
# 								83, 143, 223, 44, 173, 321, 366, 383, 122, 279, 14, 95, 237, 26, 166, 86,
# 								62, 327, 293, 241, 107, 55, 111, 400, 401]

clusters = {
	'canadian_cluster':{
		'regions':{
		'CAN-0': {'stations':['NEW', 'T19', 'C10', 'LET', 'T03', 'T43']},
		'CAN-1': {'stations':['LET', 'T03', 'T43', 'RED', 'C06']},
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


for cluster in clusters:
	max_rsd_region_in_cluster(clusters[cluster])

for cluster in clusters:
	for region in clusters[cluster]['regions']:
		clusters[cluster]['regions'][region]['mean_lat'] = utils.getting_mean_lat(clusters[cluster]['regions'][region]['stations'])
		calculating_rsd(clusters[cluster]['regions'][region])

regions = {**clusters['greenland_cluster']['regions'], **clusters['canadian_cluster']['regions'],
			**clusters['fennoscandinavian_cluster']['regions'], **clusters['central_european_cluster']['regions'],
			**clusters['non_cluster_regions']['regions']}

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
	stations[station]['df'] = df[['MLAT', 'MLT', 'dbht']]


# df1 = clusters['fennoscandinavian_cluster']['regions']['FSC-2']['rsd_df'][['max_rsd', 'max_station']]
# df2 = clusters['fennoscandinavian_cluster']['regions']['FSC-4']['rsd_df'][['max_rsd', 'max_station']]
# df3 = clusters['fennoscandinavian_cluster']['regions']['FSC-6']['rsd_df'][['max_rsd', 'max_station']]

# temp = pd.concat([df1['max_station'], df2['max_station'], df3['max_station']], axis=1, ignore_index=False)

# temp.columns = ['FSC-2', 'FSC-4', 'FSC-6']

# # determining if any of the stations are the same for each time step
# temp['same'] = temp.apply(lambda row: len(row.dropna().unique()) < len(row.dropna()), axis=1)

# print(temp)

# dropped = temp[temp['same'] == True]
# dropped.drop('same', inplace=True, axis=1)

# # duplicate = dropped.T
# # duplicate.dropna(inplace=True)
# # dup = duplicate.duplicated(keep=False)
# # print(dup)

# # creating a mask for the stations that are different in each row
# dup = []
# i = 0
# for row in tqdm(dropped.iterrows()):
# 	duplicated = row[1].dropna().duplicated(keep=False)
# 	# dup = pd.concat([dup, duplicated], axis=1)
# 	dup.append(duplicated)
# 	i=+1
# 	if i % 10000 == 0:
# 		print(i/len(dropped))
# dup = pd.concat(dup, axis=1)

# to_mask = dropped.T

# dup_masked = to_mask.mask(~dup, np.nan)

# print(dup_masked)


# defining the latitude bins in degrees
lat_delimiter = 5

# setting a date range for the index
index_date_range = pd.date_range('2009-07-20', '2017-12-31', freq='min')

lat_dict = {}
if not os.path.exists('outputs/lat_bins/'):
	os.makedirs('outputs/lat_bins/')
for i in range(20, 90, lat_delimiter):
	cluster_flag = False
	# scandinavia, svalbard, greenland, central_europe, central_canadian = pd.DataFrame(index=index_date_range), \
	# 																	pd.DataFrame(index=index_date_range), \
	# 																	pd.DataFrame(index=index_date_range), \
	# 																	pd.DataFrame(index=index_date_range), \
	# 																	pd.DataFrame(index=index_date_range)
	lat_dict[f'{i}'] = {'rsd':{'df': pd.DataFrame()}, 'dbht':{'df': pd.DataFrame()}}
	rsd_df = pd.DataFrame()
	# try:
	for region in regions.values():
		temp_df = region['rsd_df']['max_rsd']
		if region['mean_lat'] >= i and region['mean_lat'] < i + lat_delimiter:
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
			rsd_df = pd.concat([rsd_df, temp_df], axis=0)
	if not rsd_df.empty:
		rsd_df.reset_index(inplace=True, drop=False)
		rsd_df.to_feather(f'outputs/lat_bins/{i}_rsd.feather')
	# except:
	# 	print('Found the error')
	# try:
	# 	lat_dict[f'{i}']['rsd']['df'] = rsd_df
	# except:
	# 	print('Error')
	# if cluster_flag:
	# 	# finding the non-empty cluster dataframe
	# 	clusters = [scandinavia, svalbard, greenland, central_europe, central_canadian]
	# 	non_empty_clusters = [cluster for cluster in clusters if not cluster.empty]
	# 	for cluster in non_empty_clusters:
	# 		cluster['max'] = cluster.max(axis=1)
	# 		cluster['max_region'] = cluster.idxmax(axis=1)
	# 		cluster.dropna(inplace=True)
	# 		print(cluster)
	# 		# using the idmax to get the data for that timestamp from that region and combining them all into a single dataframe
	# 		max_cluster_df = pd.DataFrame()
	# 		for index, row in cluster.iterrows():
	# 			max_cluster_df = pd.concat([max_cluster_df, stats[row['max_region']]['max_rsd'].loc[index]], axis=0)

	# 		print('max cluster df')
	# 		print(max_cluster_df)

	# 		# adding the omni data to the max_cluster_df
	# 		max_cluster_df = pd.concat([max_cluster_df, omni], axis=1, join='inner')
	# 		lat_dict[i]['rsd']['df'] = pd.concat([lat_dict[i]['rsd']['df'], max_cluster_df], axis=0)

for i in range(20, 90, lat_delimiter):

	# try:
	dbdt_df = pd.DataFrame()
	for station in stations:
		temp_df = stations[station]['df']
		segmented_df = temp_df[(temp_df['MLAT']>=i) & (temp_df['MLAT']<i+lat_delimiter)]
		if not segmented_df.empty:
			segmented_df = pd.concat([segmented_df, omni], axis=1, join='inner')
			dbdt_df = pd.concat([dbdt_df, segmented_df], axis=0)
	if not dbdt_df.empty:
		dbdt_df.reset_index(inplace=True, drop=False)
		dbdt_df.to_feather(f'outputs/lat_bins/{i}_dbht.feather')
	# except:
	# 	print('Error')
	# try:
	# 	lat_dict[f'{i}']['dbht']['df'] = dbdt_df
	# except:
	# 	print('later error')

print('Done')
