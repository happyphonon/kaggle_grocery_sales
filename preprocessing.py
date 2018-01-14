import sys
from itertools import combinations
import datetime
import calendar
import time
import cPickle as pickle

import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv', usecols=[1,2,3,4,5],skiprows=range(1, 66458909), 
	                dtype={'onpromotion': bool, 'store_nbr' : np.int8, 'item_nbr' : np.int32})#, nrows=100)
test = pd.read_csv("../input/test.csv", usecols=[0, 1, 2, 3, 4],
				   dtype={'onpromotion': bool, 'store_nbr' : np.int8, 'item_nbr' : np.int32})#, nrows=100)
train['unit_sales'] = train['unit_sales'].map(lambda x : np.log1p(float(x)) if float(x)>0.0 else 0.0)
test['unit_sales'] = np.zeros((test.shape[0], 1), dtype=np.float32)

# create datetime cols
def week_of_month(tgtdate):
	tgtdate = tgtdate.to_datetime()

	days_this_month = calendar.mdays[tgtdate.month]
	for i in range(1, days_this_month):
		d = datetime.datetime(tgtdate.year, tgtdate.month, i)
		if d.day - d.weekday() > 0:
			startdate = d
			break
	# now we canuse the modulo 7 appraoch
	return (tgtdate - startdate).days //7 + 1

datetime_category = []

def get_datetime_features(data, datetime_col):
	data['date'] = pd.to_datetime(data[datetime_col])
	data['year'] = data[datetime_col].dt.year
	data['month'] = data[datetime_col].dt.month
	data['week'] = data[datetime_col].dt.weekofyear
	data['day'] = data[datetime_col].dt.day
	data['week_of_month'] = data[datetime_col].apply(week_of_month)
	data['quarter'] = data[datetime_col].dt.quarter
	datetime_category.extend(['year', 'month', 'week', 'day', 'week_of_month'])

	return data

n_train = train.shape[0]
data = pd.concat([train, test], axis=0)
del train, test

data = get_datetime_features(data, datetime_col='date')
print 'Datetime Category: {}'.format(datetime_category)

def create_rolling_average_feature(data, aggregate_cols, date_col, target_col, window = 3, min_periods = 3, shift = 16):
	data_agg = data.groupby(aggregate_cols + [date_col])[target_col].sum().reset_index()
	#data_agg.sort_values(aggregate_cols, inplace = True)
	col_name = '_'.join(aggregate_cols) + '_rolling_mean_{}'.format(window)
	dates = [i.date() for i in pd.date_range(min(data[date_col]), max(data[date_col]))]
	data_agg.set_index(pd.to_datetime(data_agg[date_col]), inplace = True) 
	temp = (data_agg.groupby(aggregate_cols)[target_col].apply(lambda x: x.reindex(dates).fillna(0).shift(shift)
														.rolling(window = window, min_periods = min_periods)
														.mean()).reset_index()
														.rename(columns={target_col : col_name, 
																'level_1' : date_col}))
	#if (temp[col_name].isnull().sum() * 1.0 / temp.shape[0] < 0.5):
		#Convert date_col to pandas datetime object, otherwise will cause problem in merging.
		#temp[date_col] = pd.to_datetime(temp[date_col])
		#data[date_col] = pd.to_datetime(data[date_col])
		#data = pd.merge(data, temp, on = aggregate_cols + [date_col], how = 'left')

	return temp, col_name, aggregate_cols + [date_col]

def create_rolling_std_feature(data, aggregate_cols, date_col, target_col, window = 3, min_periods = 3, shift = 16):
	data_agg = data.groupby(aggregate_cols + [date_col])[target_col].sum().reset_index()
	#data_agg.sort_values(aggregate_cols, inplace = True)
	col_name = '_'.join(aggregate_cols) + '_rolling_std_{}'.format(window)
	dates = [i.date() for i in pd.date_range(min(data[date_col]), max(data[date_col]))]
	data_agg.set_index(pd.to_datetime(data_agg[date_col]), inplace = True) 
	temp = (data_agg.groupby(aggregate_cols)[target_col].apply(lambda x: x.reindex(dates).fillna(0).shift(shift)
														.rolling(window = window, min_periods = min_periods)
														.std()).reset_index()
														.rename(columns={target_col : col_name, 
																'level_1' : date_col}))
	#if (temp[col_name].isnull().sum() * 1.0 / temp.shape[0] < 0.5):
		#Convert date_col to pandas datetime object, otherwise will cause problem in merging.
		#temp[date_col] = pd.to_datetime(temp[date_col])
		#data[date_col] = pd.to_datetime(data[date_col])
		##data = pd.merge(data, temp, on = aggregate_cols + [date_col], how = 'left')

	return temp, col_name, aggregate_cols + [date_col]

def create_rolling_average_feature_time(data, aggregate_cols, date_col, target_col, window = 3, min_periods = 3, shift = 16):
	data_agg = data.groupby(aggregate_cols + [date_col])[target_col].sum().reset_index()
	#data_agg.sort_values(aggregate_cols, inplace = True)
	col_name = '_'.join(aggregate_cols) + '_rolling_mean_time_{}'.format(window)
	temp = (data_agg.groupby(aggregate_cols)[target_col].apply(lambda x: x.shift(shift)
														.rolling(window = window, min_periods = min_periods)
														.mean()).reset_index()
														.rename(columns={target_col : col_name}))
	if (temp[col_name].isnull().sum() * 1.0 / temp.shape[0] < 0.5):
		#Convert date_col to pandas datetime object, otherwise will cause problem in merging.
		#data_agg[date_col] = pd.to_datetime(data_agg[date_col])
		#data[date_col] = pd.to_datetime(data[date_col])
		data_agg[col_name] = temp[col_name]
		data_agg.drop(target_col, axis = 1, inplace = True)
		#data = pd.merge(data, data_agg, on = aggregate_cols + [date_col], how = 'left')

	return data_agg, col_name, aggregate_cols + [date_col]

def create_rolling_std_feature_time(data, aggregate_cols, date_col, target_col, window = 3, min_periods = 3, shift = 16):
	data_agg = data.groupby(aggregate_cols + [date_col])[target_col].sum().reset_index()
	#data_agg.sort_values(aggregate_cols, inplace = True)
	col_name = '_'.join(aggregate_cols) + '_rolling_std_time_{}'.format(window)
	temp = (data_agg.groupby(aggregate_cols)[target_col].apply(lambda x: x.shift(shift)
														.rolling(window = window, min_periods = min_periods)
														.std()).reset_index()
														.rename(columns={target_col : col_name}))
	if (temp[col_name].isnull().sum() * 1.0 / temp.shape[0] < 0.5):
		#Convert date_col to pandas datetime object, otherwise will cause problem in merging.
		#data_agg[date_col] = pd.to_datetime(data_agg[date_col])
		#data[date_col] = pd.to_datetime(data[date_col])
		data_agg[col_name] = temp[col_name]
		data_agg.drop(target_col, axis = 1, inplace = True)
		#data = pd.merge(data, data_agg, on = aggregate_cols + [date_col], how = 'left')
		
	return data_agg, col_name, aggregate_cols + [date_col]

def generate_all_combinations(categorical_variables, max_len):
	res = []
	for i in range(1, min(max_len, len(categorical_variables)) + 1):
		for subset in combinations(categorical_variables, i):
			res.append(list(subset))

	return res

agg_funcs = [create_rolling_average_feature, create_rolling_std_feature]
time_agg_funcs = [create_rolling_average_feature_time, create_rolling_std_feature_time]

windows = [7, 30, 150, 365]
datetime_col = 'date'
target_col = 'unit_sales'
categorical_variables = ['store_nbr', 'onpromotion']
for window in windows:
	print window
	start_time = time.time()
	for combo in generate_all_combinations(categorical_variables, max_len=1):
		print combo
		aggregate_cols = ['item_nbr'] + combo
		for func in agg_funcs:
			agg_data, col_name, merge_cols = func(data, aggregate_cols, datetime_col, target_col, window=window)
			file_name = 'agg_data/agg_features_window_{}_combo_{}_func_{}.p'.format(window, '_'.join(combo), func.func_name)
			pickle.dump((agg_data, col_name, merge_cols), open(file_name, 'wb'))
		end_time = time.time()
		print 'Take %s to process %s' % (end_time-start_time, combo)
		start_time = end_time

for window in windows:
	print window
	start_time = time.time()
	for combo in generate_all_combinations(categorical_variables, max_len=1):
		print combo
		for c in datetime_category:
			print c
			aggregate_cols = ['item_nbr'] + combo + [c]
			for func in time_agg_funcs:
				agg_data, col_name, merge_cols = func(data, aggregate_cols, datetime_col, target_col, window=window)
				file_name = 'agg_data/agg_features_window_{}_combo_{}_func_{}.p'.format(window, '_'.join(combo), func.func_name)
			pickle.dump((agg_data, col_name, merge_cols), open(file_name, 'wb'))
			end_time = time.time()
			print 'Take %s to process %s, %s' % (end_time-start_time, combo, c)
			start_time = time.time()

pickle.dump((data, n_train), open('combine_train_test.p', 'wb'))