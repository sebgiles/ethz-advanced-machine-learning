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

# plot scatter plots to understand distribution of outliers
# x_idx = [10, 20, 30, 40]
# for idx in x_idx:
#     plt.scatter(x_train[:, idx], x_train[:, idx+5])
#     plt.show()

# normalize with min-max the training set
min_max_sc = preprocessing.MinMaxScaler()
x_train_norm = pd.DataFrame(min_max_sc.fit_transform(x_train))
x_test_norm = pd.DataFrame(min_max_sc.transform(x_test))
df_tot_norm = pd.concat([x_train_norm, x_test_norm])

# do outlier detection using DBSCAN
eps_arr_man = [133.8, 133.9, 134.0, 134.1, 134.2, 134.3, 134.4, 134.5, 134.6]
for i in eps_arr_man:
    print('manhattan EPS: {}'.format(i))

    outlier_detection = DBSCAN(eps=i, metric="manhattan", min_samples=20, n_jobs=-1)
    clusters = outlier_detection.fit_predict(df_tot_norm)

    n_outliers_train = -np.sum(clusters[:n_samples_train])
    print('Train: {}'.format(n_outliers_train))

    n_outliers_test = -np.sum(clusters[n_samples_train:])
    print('Test: {}\n'.format(n_outliers_test))

eps_arr_euc = [6.3, 6.31, 6.32, 6.33, 6.34, 6.35]
for j in eps_arr_euc:
    print('euclidean EPS: {}'.format(j))

    outlier_detection = DBSCAN(eps=j, metric="euclidean", min_samples=20, n_jobs=-1)
    clusters = outlier_detection.fit_predict(df_tot_norm)

    n_outliers_train = -np.sum(clusters[:n_samples_train])
    print('Train: {}'.format(n_outliers_train))

    n_outliers_test = -np.sum(clusters[n_samples_train:])
    print('Test: {}\n'.format(n_outliers_test))



