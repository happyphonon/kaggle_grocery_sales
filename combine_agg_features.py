import cPickle as pickle

import pandas as pd
import numpy as np

import glob

with open('combine_train_test.p') as file:
	data, n_train = pickle.load(file)

agg_files = glob.glob('agg_data/*.p')

numerical_variables = []
for f in agg_files:
	with open(f) as file:
		agg_data, col_name, merge_cols = pickle.load(file)
	print col_name
	if (agg_data[col_name].isnull().sum() * 1.0 / agg_data.shape[0] < 0.5):
		data = pd.merge(data, agg_data, on = merge_cols, how = 'left')
		numerical_variables.append(col_name)

#pickle.dump((data, n_train, numerical_variables), open('combine_agg_features_train_test.p', 'wb'))

train = data.iloc[:n_train, :]
test = data.iloc[n_train:, :]

train.to_csv('train_with_agg_features.csv', index=False)
test.to_csv('test_with_agg_features.csv', index=False)