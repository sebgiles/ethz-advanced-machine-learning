# TASK 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

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
labels_out = np.genfromtxt('X_test.csv', delimiter=",")[1:, 0]

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

# normalization on all data
scaler = StandardScaler()
x_tot = pd.concat([x_train, x_test])
x_tot_norm = scaler.fit_transform(x_tot)
x_train_norm = x_tot_norm[:x_train.shape[0], :]
x_test_norm = x_tot_norm[x_train.shape[0]:, :]
X_in = x_train_norm
X_out = x_test_norm
y_in = y_train

# feature selection
estimator = GradientBoostingRegressor(loss="ls", n_estimators=300, max_depth=4, subsample=0.7, random_state=666, max_features="auto")
rfe = RFE(estimator, n_features_to_select=200, step=20, verbose=1)
rfe.fit_transform(X_in, y_in)
indx = rfe.get_support(indices=True)
X_in = X_in[:, indx]
X_out = X_out[:, indx]

rfe = RFE(estimator, n_features_to_select=100, step=3, verbose=1)
rfe.fit_transform(X_in, y_in)
indx = rfe.get_support(indices=True)
X_in = X_in[:, indx]
X_out = X_out[:, indx]

rfe = RFE(estimator, n_features_to_select=50, step=1, verbose=1)
rfe.fit_transform(X_in, y_in)
indx = rfe.get_support(indices=True)
X_in = X_in[:, indx]
X_out = X_out[:, indx]

# use different regressors
grad_1 = GradientBoostingRegressor(n_estimators=300, max_depth=4, subsample=0.8, random_state=666, max_features="auto")
grad_2 = GradientBoostingRegressor(n_estimators=300, max_depth=4, subsample=0.8, random_state=667, max_features="auto")
grad_3 = GradientBoostingRegressor(n_estimators=300, max_depth=4, subsample=0.8, random_state=668, max_features="auto")
grad_4 = GradientBoostingRegressor(n_estimators=300, max_depth=4, subsample=0.8, random_state=669, max_features="auto")
grad_5 = GradientBoostingRegressor(n_estimators=300, max_depth=4, subsample=0.8, random_state=670, max_features="auto")

grad_6 = GradientBoostingRegressor(loss="ls", n_estimators=300, max_depth=4, subsample=0.7, random_state=671, max_features="auto")
grad_7 = GradientBoostingRegressor(loss="ls", n_estimators=300, max_depth=4, subsample=0.7, random_state=672, max_features="auto")
grad_8 = GradientBoostingRegressor(loss="ls", n_estimators=300, max_depth=4, subsample=0.7, random_state=673, max_features="auto")
grad_9 = GradientBoostingRegressor(loss="ls", n_estimators=300, max_depth=4, subsample=0.7, random_state=674, max_features="auto")
grad_10 = GradientBoostingRegressor(loss="ls", n_estimators=300, max_depth=4, subsample=0.7, random_state=675, max_features="auto")

grad_11 = GradientBoostingRegressor(loss="huber", n_estimators=300, max_depth=4, subsample=0.7, random_state=676, max_features="auto")
grad_12 = GradientBoostingRegressor(loss="huber", n_estimators=300, max_depth=4, subsample=0.7, random_state=677, max_features="auto")
grad_13 = GradientBoostingRegressor(loss="huber", n_estimators=300, max_depth=4, subsample=0.7, random_state=678, max_features="auto")
grad_14 = GradientBoostingRegressor(loss="huber", n_estimators=300, max_depth=4, subsample=0.7, random_state=679, max_features="auto")
grad_15 = GradientBoostingRegressor(loss="huber", n_estimators=300, max_depth=4, subsample=0.7, random_state=680, max_features="auto")


regs = [grad_1, grad_2, grad_3, grad_4, grad_5, grad_6, grad_7, grad_8, grad_9, grad_10, grad_11, grad_12,
        grad_13, grad_14, grad_15]
fits = np.zeros((X_in.shape[0], len(regs)))
preds = np.zeros((X_out.shape[0], len(regs)))
for i, reg in enumerate(regs):
    reg.fit(X_in, np.ravel(y_in))
    fits[:, i] = reg.predict(X_in)
    preds[:, i] = reg.predict(X_out)

train_pred = np.mean(fits, axis=1)
test_pred = np.mean(preds, axis=1)

print(r2_score(y_train, np.ravel(train_pred)))

test_pred = np.reshape(test_pred, (test_pred.shape[0], 1))
test_pred = np.round(test_pred)
labels_out = np.reshape(labels_out, (labels_out.shape[0], 1))

output = np.concatenate((labels_out, test_pred), axis=1)
np.savetxt(y_test_dir, output, delimiter=",", header="id,y", comments='')
