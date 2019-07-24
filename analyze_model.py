import sys 
import os 
import pickle
import numpy as np
import pandas as pd
from absl import flags
from sklearn import linear_model 

from DogR import *
from represent import *	

flags.DEFINE_string('dataset', 'Tract_Data_AFF_reduced_5', 'Name of the dataset located in Data/ folder.')
flags.DEFINE_string('y_var', 'meanV', 'Y variable of the dataset.')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if __name__ == '__main__':
	file_name =  "Data/" + FLAGS.dataset + ".csv"
	figs_dir = 'Figures/{}'.format(FLAGS.dataset)
	pickle_dir = 'Pickels/{}'.format(FLAGS.dataset)
	Y_var = FLAGS.y_var

	if not os.path.exists(figs_dir):
		os.makedirs(figs_dir)

	if not os.path.exists(pickle_dir):
		print ("Model file is not found!")
		exit()

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
	mlr_coefs = pickle.load(open("{}/mlr_params.p".format(pickle_dir), "r"))

	# ***********************************
	# ********         dogr       ********
	# ***********************************
	XY = np.array(df[name_of_variables_including_y].values)
	dogr = DogR(1) # 1 is a dummy number! 

	dogr.load("{}/fdwr_params.p".format(pickle_dir))

	number_of_components = dogr.coefficients.shape[0]

	dogr_coefs = dogr.coefficients[:]
	groups = dogr.get_groups(XY)

	# ***********************************
	# ********       COMPARE     ********
	# ***********************************

	def get_standard_error(comp_df, coefficients):
		X = np.array(df[name_of_variables].values)
		y = np.array(df[Y_var])
		n = len(df)

		y_pred = np.matmul(sm.add_constant(X), coefficients.T)

		C = np.mean((y_pred - y)**2) * np.diag(np.linalg.inv(np.matmul(X.T, X)))

		C = np.sqrt(C)

		return C, n

	# P-value, confidence interval, and hypothesis test
	def get_stat_tests(comp_df, coefficients, mlr_coefficients):
		from scipy.stats import t

		dogr_stds, n = get_standard_error(comp_df, coefficients)
		mlr_stds, n = get_standard_error(comp_df, mlr_coefficients)

		denom = np.sqrt(dogr_stds**2 + mlr_stds**2) 

		tt = (coefficients[1:] - mlr_coefficients[1:]) / denom 
		pval = t.sf(np.abs(tt), n - (len(coefficients)))*2

		import scipy.stats as st
		pval = 1.0 - st.norm.cdf(np.abs(tt))

		t_thresh = t.ppf(1 - (level_of_significance / 2),  n - (len(coefficients)))
		reject = np.abs(tt) > t_thresh

		return np.array(pval), t_thresh * dogr_stds, np.array(reject, dtype=int)

	level_of_significance = 0.05
	coef_pvalue = np.zeros((number_of_components, len(name_of_variables)))
	coef_confidence = np.zeros((number_of_components, len(name_of_variables)))
	coef_reject = np.zeros((number_of_components, len(name_of_variables)))
	for comp in range(number_of_components):
		# Bonferroni Correction 
		comp_df = df.iloc[np.where(groups == comp)[0], :]
		coef_pvalue[comp, :], coef_confidence[comp, :], coef_reject[comp, :] = get_stat_tests(comp_df, dogr_coefs[comp, :], mlr_coefs)

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
			print ("\tcomponent #{0}: {1:.4f} (p-value mlr = dogr: {2:.2f})".format(comp, dogr_coefs[comp, ind + 1], coef_pvalue[comp, ind]))


	print ("***************************************")
	# ****************************************
	# ********       Find Paradox     ********
	# ****************************************
	number_of_non_equal_components = np.sum(coef_reject, axis=0, dtype=int)

	print ("Find interesting variables: ")
	for ind in range(0, len(name_of_variables)):
		print ("\t# of non-equal components for {} = {}".format(name_of_variables[ind], number_of_non_equal_components[ind]))
		print ("groups: ", groups)
		if number_of_non_equal_components[ind] == number_of_components:
			print ("*** {} is important ***".format(name_of_variables[ind]))
			print ("\t Aggregated coefficient: {0:.6f} ".format(mlr_coefs[ind + 1]))
			print ("\t Disaggregated coefficients: {}".format(', '.join("{0:.6f} +- ({1:.6f})".format(dogr_coefs[c, ind + 1], coef_confidence[c, ind]) for c in range(number_of_components))))
		plot_paradoxical_feature(df[name_of_variables[ind]], df[Y_var], figs_dir, name_of_variables[ind], Y_var, groups, dogr_coefs[:, ind + 1], coef_confidence[:, ind], mlr_coefs[ind + 1])


	print ("***************************************")

	import scipy.stats

	def get_confidence_interval(std, n, confidence=0.95):
		h = std * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
		return h

	Y_mean = np.mean(XY[:, -1])
	y_con = get_confidence_interval(scipy.stats.sem(XY[:, -1]), len(XY))

	print ("All = {0}, meanV = {1:.2f}, st-deviation = {2:.3f}, st-error = {3:.4f}, 95-confidence = {4:.3f}".format(len(XY), Y_mean, np.sqrt(np.var(XY[:, -1])), np.sqrt(np.var(XY[:, -1])) / np.sqrt(len(XY)), y_con))
	
	Y_values = [np.nan for _ in range(number_of_components)]
	y_confidence = [np.nan for _ in range(number_of_components)]
	for com_num in range(number_of_components):
	    indices = [index for index in range(len(groups)) if groups[index] == com_num]
	    if len(indices) > 0:
	    	Y_values[com_num] = np.mean(XY[indices, -1])
	    	y_confidence[com_num] = get_confidence_interval(scipy.stats.sem(XY[indices, -1]), len(indices))

	    	print ("Group size = {0}, meanV = {1:.2f}, st-deviation = {2:.3f}, st-error = {3:.4f}, 95-confidence = {4:.3f}".format(len(indices), Y_values[com_num], np.sqrt(np.var(XY[indices, -1])), np.sqrt(np.var(XY[indices, -1])) / np.sqrt(len(indices)), y_confidence[com_num]))
	    	
	Y_pred = dogr.predict(dogr.mu)
	y_sigma = dogr.y_sigma
	var_dict = dict()
	for var in name_of_variables:
		var_dict[var] = var

	show_components_normalized(dogr.mu, name_of_variables, Y_var, Y_values, y_confidence, Y_pred, y_sigma, var_dict, "{}/_components_normalized.pdf".format(figs_dir), (1.2, -0.1))

	print ("***************************************")
