import numpy as np
from fancyimpute import simple_fill
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

"""
0. median imputation
1. normalise
2. feature selection using rfe to 250 ( -> IMPROVE!!!)
3. grad, knn regressors -> bagging -> average (1, 0.5)
4. => final prediction

"""

### 0. import -> impute
X_in = np.genfromtxt ('X_train.csv', delimiter=",")[1:,1:]
y_in = np.genfromtxt ('y_train.csv', delimiter=",")[1:,1:]
X_out = np.genfromtxt ('X_test.csv', delimiter=",")[1:,1:] #also contains NAs
print(X_in.shape)
print(X_out.shape)
labels_out = np.genfromtxt ('X_test.csv', delimiter=",")[1:,0]
X_tot = np.concatenate((X_in, X_out), axis=0)
print(X_tot.shape)

median_imputer = simple_fill.SimpleFill(fill_method="median")
X_tot = median_imputer.fit_transform(X_tot)


### 1. normalise
scaler = StandardScaler()
X_tot = scaler.fit_transform(X_tot)
X_in = X_tot[:X_in.shape[0],:]
X_out = X_tot[X_in.shape[0]:,:]
print(X_in.shape)
print(X_out.shape)

### 2. feature selection
estimator = LinearRegression()
rfe = RFE(estimator, n_features_to_select=250, step=2, verbose=1)
rfe.fit_transform(X_in, y_in)
indx = rfe.get_support(indices=True)
X_in = X_in[:,indx]
X_out = X_out[:,indx]

### 3. use different regressors
grad_1 = GradientBoostingRegressor(n_estimators=500, max_depth=4, subsample=0.8, random_state=666, max_features="auto")
grad_2 = GradientBoostingRegressor(n_estimators=500, max_depth=4, subsample=0.8, random_state=667, max_features="auto")
grad_3 = GradientBoostingRegressor(n_estimators=500, max_depth=4, subsample=0.8, random_state=668, max_features="auto")
grad_4 = GradientBoostingRegressor(n_estimators=500, max_depth=4, subsample=0.8, random_state=669, max_features="auto")
grad_5 = GradientBoostingRegressor(n_estimators=500, max_depth=4, subsample=0.8, random_state=670, max_features="auto")

knn = KNeighborsRegressor(n_neighbors=9, weights="distance", p=1)

regs = [grad_1, grad_2, grad_3, grad_4, grad_5, knn]
fits= np.zeros((X_in.shape[0], len(regs)))
preds = np.zeros((X_out.shape[0], len(regs)))
for i, reg in enumerate(regs):
    reg.fit(X_in, np.ravel(y_in))
    fits[:, i] = reg.predict(X_in)
    preds[:, i] = reg.predict(X_out)


mask = [.2, .2, .2, .2, .2, .5]
mask = mask / np.sum(mask)
print(mask)
train_pred = np.sum(fits*mask, axis=1)
test_pred = np.sum(preds*mask, axis=1)

print(r2_score(y_in, np.ravel(train_pred))) # training accuracy

test_pred = np.reshape(test_pred, (test_pred.shape[0],1))
labels_out = np.reshape(labels_out, (labels_out.shape[0],1))

out = np.concatenate((labels_out, test_pred), axis=1)
np.savetxt("out.csv", out, delimiter=",", header="id,y") # NEED TO MANUALLY REMOVE # FROM HEADER