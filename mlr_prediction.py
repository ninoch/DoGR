import sys 
import os
import pickle
import numpy as np
import pandas as pd
from absl import flags
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

flags.DEFINE_string('dataset', 'Tract_Data_AFF_reduced_5', 'Name of the dataset located in Data/ folder.')
flags.DEFINE_string('y_var', 'meanV', 'Y variable of the dataset.')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if __name__ == '__main__':
	file_name =  "Data/" + FLAGS.dataset + ".csv"
	pickle_dir = 'Pickels/{}'.format(FLAGS.dataset)

	if not os.path.exists(pickle_dir):
		os.makedirs(pickle_dir)

	Y_var = FLAGS.y_var

	df = pd.read_csv(file_name)
	# df = df.drop(['BOROUGH'], axis=1)

	name_of_variables = set(df.columns.values)
	if Y_var not in name_of_variables:
		print ("Y_var is not in name of variables!")
		exit()
	name_of_variables.remove(Y_var)
	name_of_variables = list(sorted(name_of_variables))
	name_of_variables_including_y = []
	name_of_variables_including_y.extend(name_of_variables)
	name_of_variables_including_y.append(Y_var)

	# *******************************************
	# ********      Cross Validation     ********
	# *******************************************
	MLR_RMSE = []
	MLR_MAE = []
	MLR_R2 = []

	outer_corss_val = KFold(n_splits=5, shuffle=True)
	# for itr in range(5):
	for itr in range(5):
		print ("***** Iteration {} *****".format(itr))
		for train_index, test_index in outer_corss_val.split(df):
				train, test = df.iloc[train_index][name_of_variables_including_y], df.iloc[test_index][name_of_variables_including_y]

				reg = linear_model.LinearRegression()
				X = train[name_of_variables]
				y = train[Y_var]
				reg.fit(X, y)

				X_test = np.array(test[name_of_variables].values)
				Y_test = np.array(test[Y_var])

				y_pred = reg.predict(X_test)

				MLR_RMSE.append(np.sqrt(mean_squared_error(Y_test, y_pred)))
				MLR_MAE.append(mean_absolute_error(Y_test, y_pred))
				MLR_R2.append(r2_score(Y_test, y_pred))


				print ("Test RMSE={0:.3f}, MAE={1:.3f}, R2={2:.3f}".format(MLR_RMSE[-1], MLR_MAE[-1], MLR_R2[-1]))



	pickle.dump([MLR_RMSE, MLR_MAE, MLR_R2], open("{}/mlr_params.p".format(pickle_dir), "wb"))
	print ("")
	print ("MLR RMSE = {0:.3f} (+- {1:.4f})".format(np.mean(MLR_RMSE), np.std(MLR_RMSE)))
	print ("MLR MAE = {0:.3f} (+- {1:.4f})".format(np.mean(MLR_MAE), np.std(MLR_MAE)))
	print ("MLR R^2 score = {0:.3f} (+- {1:.4f})".format(np.mean(MLR_R2), np.std(MLR_R2)))


