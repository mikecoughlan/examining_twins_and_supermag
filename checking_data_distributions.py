import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils

data_directory = '../../../../data/'
working_dir = data_directory+'mike_working_dir/twins_data_modeling/'
VERSION = 0

def loading_data():

	with open(f'outputs/dates_dict_version_{VERSION}.pkl', 'rb') as f:
		dates_dict = pickle.load(f)

	solarwind = utils.loading_solarwind(omni=True, limit_to_twins=True)

	return dates_dict, solarwind


def creating_combined_df_for_plotting(date_dict):

	# printint the percentage of each data split
	total_length = np.sum([len(date_dict[key]['Date_UTC']) for key in date_dict.keys()])
	for key, value in date_dict.items():
		print(f'{key} percentage: {len(value["Date_UTC"])/total_length}')

	start_date = pd.to_datetime('2009-07-19')
	end_date = pd.to_datetime('2017-12-31')
	date_range = pd.date_range(start_date, end_date, freq='min')

	# creating a dataframe with the dates as the index
	df = pd.DataFrame(index=date_range)

	# joining each of the dataframes to the main dataframe leaving nans where there aren't any values. Renaming the joined column to the dict key name
	for i, (key, value) in enumerate(date_dict.items()):
		temp_df = pd.DataFrame(index=value['Date_UTC'])
		temp_df[f'{key}_top'] = int(1)*(i+1)
		temp_df[f'{key}_bottom'] = int(1)*(i+0.1)
		df = df.join(temp_df, how='left')

	# filling the nans with 0s
	df.fillna(0, inplace=True)

	return df


def plotting_date_distributions(df):

	keys = ['train', 'val', 'test']

	colors = sns.color_palette('tab20', len(keys))

	fig = plt.figure(figsize=(20,10))

	plt.xlim(pd.to_datetime('2009-07-19'), pd.to_datetime('2017-12-31'))

	for col, key in zip(colors, keys):
		plt.fill_between(df.index, df[f'{key}_bottom'], df[f'{key}_top'], color=col, alpha=1, label=key,
							where=np.array(df[f'{key}_top'])>np.array(df[f'{key}_bottom']))
		plt.yticks([])

	plt.title('Data Splits')
	plt.margins(y=0)
	plt.yticks([])
	plt.legend()

	plt.savefig(f'plots/data_splits_by_month.png')




def checking_solarwind_distribution(date_dict, solarwind):

	params_to_examine = ['BZ_GSM', 'BY_GSM', 'B_Total', 'Vx', 'Vy', 'Vz', 'T', 'Pressure', 'proton_density', 'E_Field']

	solarwind = solarwind[params_to_examine]

	# scaling solar wind data so it is easy to plot
	solarwind = (solarwind - solarwind.min())/(solarwind.max() - solarwind.min())

	new_dfs = {key:pd.DataFrame(index=date_dict[key]['Date_UTC']) for key in date_dict.keys()}

	for key in date_dict.keys():
		new_dfs[key] = new_dfs[key].join(solarwind, how='left')

	stats_dict = {'mean':pd.DataFrame({key:new_dfs[key].mean() for key in new_dfs}),
					'std':pd.DataFrame({key:new_dfs[key].std() for key in new_dfs})}


	print(f'Printing the mean and std for each parameter for each data split')
	for stat, df in stats_dict.items():
		print(f'\n{stat}')
		print(df)


def main():

	date_dict, solarwind = loading_data()

	checking_solarwind_distribution(date_dict, solarwind)

	df = creating_combined_df_for_plotting(date_dict)

	plotting_date_distributions(df)


if __name__ == '__main__':

	main()
