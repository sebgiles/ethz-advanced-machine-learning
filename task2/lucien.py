import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
from collections import Counter

from sklearn.metrics import r2_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE

"""
1. import data, normalise
2. train different models
    - 3 different NN structures, 5 reps each
    - ?? GradBoostClf ?? on ROS data??
=> ensemble 
?? 3. semi-supervised learning
"""


######################################################################################################################
### IMPORT DATA
X_in = np.genfromtxt ('X_train.csv', delimiter=",")[1:,1:]
y_in = np.genfromtxt ('y_train.csv', delimiter=",")[1:,1:]
X_out = np.genfromtxt ('X_test.csv', delimiter=",")[1:,1:]
print(X_out.shape)
y_in_hot = to_categorical(y_in)
y_out = np.genfromtxt ('sample.csv', delimiter=",")
print(y_out.shape)
### NORMALISE
sts = StandardScaler()
X_in = sts.fit_transform(X_in)
X_out = sts.fit_transform(X_out)

sm = SVMSMOTE(random_state=42)
X_in, y_in_hot = sm.fit_resample(X_in, y_in_hot)


######################################################################################################################
### FFNNs

# callback
cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto', restore_best_weights=True)

### MODELS
model_probs = np.zeros((4100,3), dtype=float)

# NN FLAT
for rep in range(5):
    model0 = Sequential()
    model0.add(Dense(100, activation='relu', input_dim=1000))
    model0.add(Dropout(0.5))
    model0.add(Dense(3, activation='softmax'))

    model0.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    model0.fit(X_in, y_in_hot, epochs=100, batch_size=128, callbacks=[cb], validation_split=0.3, verbose=0)

    model_probs = model_probs+model0.predict_proba(X_out)

# NN MEDIUM
for rep in range(5):
    model1 = Sequential()
    model1.add(Dense(40, activation='relu', input_dim=1000))
    model1.add(Dropout(0.5))
    model1.add(BatchNormalization())
    model1.add(Dense(40, activation='relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(3, activation='softmax'))

    model1.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    model1.fit(X_in, y_in_hot, epochs=100, batch_size=128, callbacks=[cb], validation_split=0.3, verbose=0)

    model_probs = model_probs+model1.predict_proba(X_out)

# NN LARGE
for rep in range(5):
    model2 = Sequential()
    model2.add(Dense(200, activation='relu', input_dim=1000))
    model2.add(Dropout(0.5))
    model2.add(BatchNormalization())
    model2.add(Dense(200, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(200, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(3, activation='softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    model2.fit(X_in, y_in_hot, epochs=100, batch_size=128, callbacks=[cb], validation_split=0.3, verbose=0)

    model_probs = model_probs+model2.predict_proba(X_out)

y_out[1:,1] = np.argmax(model_probs, axis=1)
np.savetxt("y_out.csv", y_out, delimiter=",", header="id,y",  comments='')



