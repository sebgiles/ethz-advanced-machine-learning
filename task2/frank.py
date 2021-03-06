import pandas as pd
import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from imblearn.over_sampling import SVMSMOTE
from sklearn.svm import SVC

# ---------------------------------------------------------------------------

# Set to 1 if you intend to have a validation set
val = 0
# Set to 1 if you intend to create a cvs with the result
create_csv = 1

# ---------------------------------------------------------------------------

# Directories of data files
x_train_dir = 'X_train.csv'
y_train_dir = 'y_train.csv'
x_test_dir = 'X_test.csv'

# Get the data
x_train = pd.read_csv(x_train_dir)
y_train = pd.read_csv(y_train_dir)
x_test = pd.read_csv(x_test_dir)

# Drop the id labels
x_train.drop(columns='id', inplace=True)
y_train.drop(columns='id', inplace=True)
x_test.drop(columns='id', inplace=True)

# To arrays
x_train = x_train.values
y_train = y_train.values
x_test = x_test.values

# Dimensions
train_datap = x_train.shape[0]
test_datap = x_test.shape[0]
n_features = x_train.shape[1]

# To standardize data
scaler = StandardScaler()

# Splitting set into training and validation
if val == 1:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)
    val_datap = x_val.shape[0]
    # Standardize validation
    x_val = scaler.fit_transform(x_val)

# Standardize training
x_train = scaler.fit_transform(x_train)

# Standardize test
x_test = scaler.fit_transform(x_test)

# --------------------------------------------------------------------------

#  Oversampling with SVM SMOTE

sm = SVMSMOTE(random_state=42)

x_train, y_train = sm.fit_resample(x_train, y_train)

# --------------------------------------------------------------------------

# # Neural Network
#
# # Epochs during training
# epochs = 200
# batch_size = 512
#
# net = keras.models.Sequential()
#
# # Fully connected layer
# net.add(keras.layers.Dense(512))
# net.add(BatchNormalization())
# net.add(Activation('relu'))
# net.add(keras.layers.Dropout(0.4))
#
# net.add(keras.layers.Dense(512))
# net.add(BatchNormalization())
# net.add(Activation('relu'))
# net.add(keras.layers.Dropout(0.4))
#
# net.add(keras.layers.Dense(256))
# net.add(BatchNormalization())
# net.add(Activation('relu'))
# net.add(keras.layers.Dropout(0.4))
#
# # Output layer with 3 possible classes
# net.add(keras.layers.Dense(3, activation=tf.nn.softmax))
#
# net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Early stopping function
# early_stop = [keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=15, verbose=1, mode='auto',
#                                             restore_best_weights=True)]
#
# # It is using weights
# net.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=early_stop)

svc = SVC(kernel='poly', gamma=7e-5, C=1, coef0=1,).fit(x_train, y_train)
svm_model = svc.fit(x_train, y_train)

# ---------------------------------------------------------------------------

# Results on validation set and prediction on test set

# if val == 1:
#     # Keras function
#     print('\nValidation Set:\n')
#     val_loss, val_accuracy = net.evaluate(x_val, y_val)
#     # Our accuracy
#     y_val_probabilities = net.predict(x_val)
#     y_val_test = np.arange(0, val_datap, 1)
#     for i in range(0, val_datap):
#         y_val_test[i] = np.argmax(y_val_probabilities[i])
#     acc = accuracy_score(y_val, y_val_test)
#     BMAC = balanced_accuracy_score(y_val, y_val_test)
#     print('\nAccuracy score:', acc)
#     print('\nBMAC score:', BMAC, '\n')
#
# y_test_probabilities = net.predict(x_test)
#
# y_test = np.arange(0, test_datap, 1)
# for i in range(0, test_datap):
#     y_test[i] = np.argmax(y_test_probabilities[i])

if val == 1:
    y_val_test = svc.predict(x_val)
    BMAC = balanced_accuracy_score(y_val, y_val_test)
    print('\nBMAC score:', BMAC, '\n')

y_test = svc.predict(x_test)

# ---------------------------------------------------------------------------

# Create cvs file

if create_csv == 1:
    Id_y_test = np.arange(0, test_datap, 1)
    df = pd.DataFrame({'id': Id_y_test[:], 'y': y_test[:]})
    pd.DataFrame(df).to_csv('y_pred.csv', index=False, header=True)
