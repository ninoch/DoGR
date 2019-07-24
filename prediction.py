import sys 
import os
import pickle
import time
import numpy as np
import pandas as pd
from absl import flags

from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from DogR import *

start_time = time.time()

flags.DEFINE_string('dataset', 'Tract_Data_AFF_reduced_5', 'Name of the dataset located in Data/ folder.')
flags.DEFINE_string('y_var', 'meanV', 'Y variable of the dataset.')
flags.DEFINE_string('min_comp', '1', 'Min number of components.')
flags.DEFINE_string('max_comp', '7', 'Max number of components.')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if __name__ == '__main__':
	file_name =  "Data/" + FLAGS.dataset + ".csv"
	pickle_dir = 'Pickels/{}'.format(FLAGS.dataset)

	if not os.path.exists(pickle_dir):
		os.makedirs(pickle_dir)

	Y_var = FLAGS.y_var

	df = pd.read_csv(file_name)

	name_of_variables = set(df.columns.values)
	if Y_var not in name_of_variables:
		print ("Y_var is not in name of variables!")
		exit()
	name_of_variables.remove(Y_var)
	name_of_variables = list(sorted(name_of_variables))
	name_of_variables_including_y = []
	name_of_variables_including_y.extend(name_of_variables)
	name_of_variables_including_y.append(Y_var)
	
	# df[name_of_variables] = (df[name_of_variables] - np.mean(df[name_of_variables], axis=0)) / np.std(df[name_of_variables], axis=0)

	# ***************************************************
	# ********       Nested Cross Validation     ********
	# ***************************************************
	num_com_hyper = range(int(FLAGS.min_comp), int(FLAGS.max_comp))
	validation_errors = []
	test_errors = []
	DogR_RMSE = []
	DogR_MAE = []

	outer_corss_val = KFold(n_splits=5, shuffle=True)
	for itr in range(5):
	# for itr in range(1):
		print ("***** Iteration {} *****".format(itr))
		for train_validation_index, test_index in outer_corss_val.split(df):
				train_df, test = df.iloc[train_validation_index][name_of_variables_including_y], df.iloc[test_index][name_of_variables_including_y]

				print ("\nSelecting hyper parameter")
				inner_cross_val = KFold(n_splits=5, shuffle=True)	
				RMSE_mean = dict()
				for number_of_components in num_com_hyper:
					RMSE = []
					for train_index, validation_index in inner_cross_val.split(train_df):
						train, validation = df.iloc[train_index][name_of_variables_including_y], df.iloc[validation_index][name_of_variables_including_y]

						X = np.array(train[name_of_variables_including_y].values)
						try:
							dogr = DogR(number_of_components)
							dogr.fit(X)
						except: 
							print (">>>>>> FAILED VALIDATION <<<<<<")
							continue

						X_val = np.array(validation[name_of_variables].values)
						Y_val = np.array(validation[Y_var])

						y_val_pred = dogr.predict(X_val)

						RMSE.append(np.sqrt(mean_squared_error(Y_val, y_val_pred)))

					RMSE_mean[number_of_components] = np.mean(RMSE)
					print ("\t RMSE for {0} = {1:.3f}".format(number_of_components, RMSE_mean[number_of_components]))

				validation_errors.append(RMSE_mean)
				best_num_of_components = min(RMSE_mean.keys(), key=lambda k: RMSE_mean[k])

				print ("Best number of components: {}".format(best_num_of_components))

				X = np.array(train_df[name_of_variables_including_y].values)
				try:
					dogr = DogR(best_num_of_components)
					dogr.fit(X)
				except:
					print (">>>>>> FAILED TEST <<<<<<")
					continue

				X_test = np.array(test[name_of_variables].values)
				Y_test = np.array(test[Y_var])

				y_pred = dogr.predict(X_test)

				DogR_RMSE.append(np.sqrt(mean_squared_error(Y_test, y_pred)))
				DogR_MAE.append(mean_absolute_error(Y_test, y_pred))


				print ("Test RMSE={0:.3f}, MAE={1:.3f}".format(DogR_RMSE[-1], DogR_MAE[-1]))

	pickle.dump([validation_errors, DogR_RMSE, DogR_MAE], open("{}/fdwr_prediction.p".format(pickle_dir), "wb"))
	print ("")
	print ("DogR RMSE = {0:.3f} (+- {1:.4f})".format(np.mean(DogR_RMSE), np.std(DogR_RMSE)))
	print ("DogR MAE = {0:.3f} (+- {1:.4f})".format(np.mean(DogR_MAE), np.std(DogR_MAE)))



import datetime
total_time = time.time() - start_time
print ("--- Time taken {} ---".format(str(datetime.timedelta(seconds=total_time))))

