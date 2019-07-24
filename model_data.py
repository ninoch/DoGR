import sys 
import os
import pickle
import numpy as np
import pandas as pd
from absl import flags
from sklearn import linear_model 
from DogR import *

flags.DEFINE_string('dataset', 'Tract_Data_AFF_reduced_5', 'Name of the dataset located in Data/ folder.')
flags.DEFINE_string('num_com', '5', 'Number of components.')
flags.DEFINE_string('y_var', 'meanV', 'Y variable of the dataset.')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if __name__ == '__main__':
	file_name =  "Data/" + FLAGS.dataset + ".csv"
	figs_dir = 'Figures/{}'.format(FLAGS.dataset)
	pickle_dir = 'Pickels/{}'.format(FLAGS.dataset)
	number_of_components = int(FLAGS.num_com)
	Y_var = FLAGS.y_var

	if not os.path.exists(figs_dir):
		os.makedirs(figs_dir)

	if not os.path.exists(pickle_dir):
		os.makedirs(pickle_dir)

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

	# ***********************************
	# ********         MLR       ********
	# ***********************************
	reg = linear_model.LinearRegression()
	X = df[name_of_variables]
	y = df[Y_var]
	reg.fit(X, y)
	mlr_coefs = [reg.intercept_]
	mlr_coefs.extend(reg.coef_)
	mlr_coefs = np.array(mlr_coefs)

	pickle.dump(mlr_coefs, open("{}/mlr_params.p".format(pickle_dir), "wb"))

	# ***********************************
	# ********         DogR       ********
	# ***********************************
	XY = np.array(df[name_of_variables_including_y].values)
	dogr = DogR(number_of_components)
	dogr.fit(XY)

	dogr_coefs = dogr.coefficients[:]
	groups = dogr.get_groups(XY)

	dogr.save("{}/fdwr_params.p".format(pickle_dir))


	# *************************************
	# ********       Show Info     ********
	# *************************************

	print ("***************************************")

	print ("Variables list:")
	for var in name_of_variables_including_y:
		print ("\t{}".format(var))

	print ("***************************************")

	print ("Mu of each component:")
	for ind, feature in enumerate(name_of_variables):
		print (feature)
		for k in range(number_of_components):
			print ("\t component #{0}: {1:.2f}".format(k, dogr.mu[k, ind]))


	print ("***************************************")

	print ("Regression of each component:")

	print ("Intercept -> \n\tMLR: {} \n\tDogR: {}".format(mlr_coefs[0], dogr_coefs[:, 0]))
	for ind, var in enumerate(name_of_variables):
		print ("{0} -> \n\tMLR: {1:.4f}".format(var, mlr_coefs[ind + 1]))
		for comp in range(number_of_components):
			print ("\tcomponent #{0}: {1:.4f}".format(comp, dogr_coefs[comp, ind + 1]))

	print ("***************************************")
