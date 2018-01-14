import pandas as pd
import numpy as np
import glob

pred_files = glob.glob('ensemble/*.csv')

y_pred_a = None
y_pred_b = None
test_id = None
n_pred = 0
for f in pred_files:
	data = pd.read_csv(f)
	print data.head
	if test_id is None:
		test_id = data['id']
	y_pred_a = y_pred_a + data['unit_sales'].values if y_pred_a is not None else data['unit_sales'].values
	y_pred_b = y_pred_b + np.log1p(data['unit_sales'].values) if y_pred_b is not None else np.log1p(data['unit_sales'].values)
	n_pred += 1

y_pred_a = y_pred_a / n_pred
y_pred_b = np.expm1(y_pred_b / n_pred)

res_a = pd.DataFrame({'id' : test_id, 'unit_sales' : y_pred_a})
res_b = pd.DataFrame({'id' : test_id, 'unit_sales' : y_pred_b})

print res_a.head()
print res_b.head()

res_a.to_csv('ensemble_a_2017_01_09_20_42.csv', index=False)
res_b.to_csv('ensemble_b_2017_01_09_20_42.csv', index=False)