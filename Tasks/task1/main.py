# TASK 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

# get directories of data
x_train_dir = 'X_train.csv'
y_train_dir = 'y_train.csv'
x_test_dir = 'X_test.csv'
y_test_dir = 'y_test.csv'

# get data
x_train = pd.read_csv(x_train_dir)
x_train = x_train.drop(columns=['id'])
y_train = pd.read_csv(y_train_dir)
y_train = y_train.drop(columns=['id'])
x_test = pd.read_csv(x_test_dir)
x_test = x_test.drop(columns=['id'])

n_samples_train = len(x_train.index)

# drop the 10% columns with most missing values
nan_train = x_train.isna().sum()
nan_test = x_test.isna().sum()
nan_tot = nan_train + nan_test
nan_tot.sort_values(ascending=False, inplace=True)
len_tot = len(nan_tot)
idx_worst_nan_tot = nan_tot[0:int(np.ceil(0.1*len_tot))].index
x_test.drop(columns=idx_worst_nan_tot, inplace=True)
x_train.drop(columns=idx_worst_nan_tot, inplace=True)

# impute the data using median imputation column-wise (simple imputer)
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
x_train = pd.DataFrame(imp_median.fit_transform(x_train))
x_test = pd.DataFrame(imp_median.fit_transform(x_test))

# normalize with min-max the training set
min_max_sc = preprocessing.MinMaxScaler()
x_train_norm = pd.DataFrame(min_max_sc.fit_transform(x_train))
x_test_norm = pd.DataFrame(min_max_sc.transform(x_test))

# do outlier detection using DBSCAN
outlier_detection = DBSCAN(eps=6.35, metric="euclidean", min_samples=20, n_jobs=-1)
clusters = outlier_detection.fit_predict(x_train_norm)
indices = np.where(clusters == -1)[0]
x_train.drop(x_train.index[indices], inplace=True)
y_train.drop(y_train.index[indices], inplace=True)
