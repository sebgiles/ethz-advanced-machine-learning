# using code in part from https://github.com/ismorphism/DeepECG/blob/master/Conv1D_ECG.py
# (accompanying material to Deep Learning for ECG Classification(Pyakillya et al, 2017)

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, GlobalMaxPool1D
from keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
import numpy as np
from sklearn.metrics import f1_score

#import h5py


#### IMPORT DATA
maxlen = 17800 # length for input. 17800 -> _rep17k; 10100 -> _rep10k
epochs = 120


###############################################################################################
############## IMPORT DATA -> NORMALISE -> TRAIN-TEST SPLIT
X_in = np.genfromtxt('X_train_rep17k.csv')
X_out = np.genfromtxt('X_test_rep17k.csv')

y_in = np.genfromtxt('y_train.csv', delimiter=",")[1:,1]
print(y_in.shape)
print(X_in.shape)
print(X_out.shape)

#### NORMALISE
sts = StandardScaler()
X_in = np.expand_dims(sts.fit_transform(X_in), axis=2)
X_out = np.expand_dims(sts.fit_transform(X_out), axis=2)
print(X_out.shape)
# #### TRAIN-TEST SPLIT
# X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.3, random_state=666)


### HOT ENCODING
#y_train_hot = to_categorical(y_train)
y_in_hot = to_categorical(y_in)

# ############ RUN NN
model = Sequential()
model.add(Conv1D(128, 5, activation='relu', input_shape=(maxlen, 1)))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))

model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
#checkpointer = keras.callbacks.callbacks.ModelCheckpoint(filepath='Best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
cb = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=epochs, verbose=1, mode='auto', restore_best_weights=True)
# bc patience = epochs: just restores best weights
print("start training...")

hist = model.fit(X_in, y_in_hot,  batch_size=275, epochs=epochs, verbose=2, shuffle=True, callbacks=[cb],
                 validation_split=0.20, class_weight={0:1/0.44, 1:1/.16, 2:1/.27, 3:1/.13})#, #validation_data=(X_val, y_val_hot),
#class_weight={0:1/0.39, 1:1/.22, 2:1/.26, 3:1/.21})
#checkpointer # counts 1000: 0.4420314 0.1582757 0.2713612 0.1283317; counts: {0:1/.59, 1:1/.09, 2:1/.29, 3:1/.03}
# counts 5000: 0.3197038 0.2167058 0.2577537 0.2058367
# counts 10000: 0.2888047 0.2314649 0.2543166 0.2254139
#pd.DataFrame(hist.history).to_csv(path_or_buf='HistoryClassWeights1000.csv')

print("------------------------------")
pred_proba = model.predict(X_out)
#y_predT = np.argmax(pred_proba, axis=1)
#scoreT = f1_score(y_test, y_predT, average="micro")
#print('Best epoch\'s test score is ', scoreT)
np.save("probas_npy/OUTmain0_17k_CW1000_probas_rand"+str(np.random.randint(1000))+".npy", pred_proba)
#np.save("probas_npy/OUTmain0_17k_OTHERSTRUC&CW1000_probas_rand"+str(np.random.randint(1000))+".npy", pred_proba)
#
#

