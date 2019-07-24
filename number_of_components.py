import sys 
import os
import pickle
import copy
import numpy as np
import pandas as pd
from absl import flags

from sklearn import linear_model 
from scipy.stats import f as fdist
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from DogR import *
from represent import *


flags.DEFINE_string('dataset', 'Tract_Data_AFF_reduced_5', 'Name of the dataset located in Data/ folder')
flags.DEFINE_string('y_var', 'meanV', 'Y variable of the dataset')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def get_AIC_BIC(df, name_of_variables, name_of_variables_including_y, components_range, pickle_dir):
  X = np.array(df[name_of_variables].values)
  Y = np.array(df[Y_var].values)
  XY = np.array(df[name_of_variables_including_y].values)
  D = len(name_of_variables)

  BIC = []
  AIC = []
  for number_of_components in components_range:
      X = np.array(df[name_of_variables_including_y].values)

      dogr = DogR(number_of_components)
      dogr.fit(X)
      dogr.save("{}/fdwr_params_{}.p".format(pickle_dir, number_of_components))
      number_of_params = number_of_components * (D + 1 + D + D * D + 1 + 1)
      BIC.append(np.log(len(df)) * number_of_params - 2 * dogr.ll)
      AIC.append(2 * number_of_params - 2 * dogr.ll)

      print ("# of comp = {0}: AIC={1:.2f}, BIC={2:.2f}".format(number_of_components, AIC[-1], BIC[-1]))

  return AIC, BIC

if __name__ == '__main__':
  file_name =  "Data/" + FLAGS.dataset + ".csv"
  figs_dir = 'Figures/{}'.format(FLAGS.dataset)
  pickle_dir = 'Pickels/{}'.format(FLAGS.dataset)

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
  components_range = range(1, 11)

  print ("Variables: \n", name_of_variables)
  # ***************************************
  # ********         AIC BIC       ********
  # ***************************************

  AIC, BIC = get_AIC_BIC(df, name_of_variables, name_of_variables_including_y, components_range, pickle_dir)
  pickle.dump([AIC, BIC], open("{}/AIC_BIC.p".format(pickle_dir), "wb"))

  # ***********************************
  # ********       Show        ********
  # ***********************************
  AIC_BIC_plot(components_range, AIC, BIC, figs_dir)
