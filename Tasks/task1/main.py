import pandas as pd
import keras
from tensorflow.python.keras import optimizers
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense

# Set to 1 if you intend to have a validation set
val = 0
# Set to 1 if you intend to create a cvs with the result
create_cvs = 1

# ---------------------------------------------------------------------------
# Get the data
# train = pd.read_csv('train.h5')
# test = pd.read_csv('test.h5')

# Divide into datapoints and classes
# x_train = train.drop(columns = ['y'])
# y_train = train['y']

x_train_dir = 'X_train.csv'
y_train_dir = 'y_train.csv'
x_test_dir = 'X_test.csv'
y_test_dir = 'y_test.csv'

x_train = pd.read_csv(x_train_dir)
x_train = x_train.drop(columns=['id'])
y_train = pd.read_csv(y_train_dir)
y_train = y_train.drop(columns=['id'])
x_test = pd.read_csv(x_test_dir)
x_test = x_test.drop(columns=['id'])

# To standardize data
scaler = StandardScaler()

# To arrays
x_train = x_train.values
y_train = y_train.values
x_test = x_test.values

# Dimensions
train_datap = x_train.shape[0]
test_datap = x_test.shape[0]
n_features = x_train.shape[1]

for i in range(n_features):
    mn = np.nanmedian(x_train[:, i])
    x_train[:, i] = np.nan_to_num(x_train[:, i], nan=mn)
    x_test[:, i] = np.nan_to_num(x_test[:, i], nan=mn)

# Splitting set into training and validation
if val == 1:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)
    val_datap = x_val.shape[0]
    # Standardize validation
    x_val = scaler.fit_transform(x_val)

# Standardize training
x_train = scaler.fit_transform(x_train)

# Standardize test
x_test = scaler.fit_transform(x_test)

# ---------------------------------------------------------------------------
# Balancing the classes
# class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# --------------------------------------------------------------------------
# Neural Network

# Epochs during training
epochs = 200;
batch_size = 256;

net = keras.models.Sequential()

# Fully connected layer
net.add(keras.layers.Dense(1024))
net.add(BatchNormalization())
net.add(Activation('relu'))
# Dropout on first hidden layer
net.add(keras.layers.Dropout(0.5))

net.add(keras.layers.Dense(512))
net.add(BatchNormalization())
net.add(Activation('relu'))
# Dropout on first hidden layer
net.add(keras.layers.Dropout(0.5))

net.add(keras.layers.Dense(256))
net.add(BatchNormalization())
net.add(Activation('relu'))
net.add(keras.layers.Dropout(0.4))

# Output layer with 5 possible classes
net.add(keras.layers.Dense(100, activation=tf.nn.softmax))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
net.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Early stopping function
early_stop = [keras.callbacks.EarlyStopping(monitor='loss',
                                            min_delta=0,
                                            patience=10,
                                            verbose=0, mode='auto')]

# It is using weights
# net.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=epochs, batch_size=batch_size, callbacks=early_stop, class_weight = class_weights)
net.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=early_stop)

# ---------------------------------------------------------------------------
# Results on validation set and prediction on test set

if val == 1:
    # Keras function
    print('\nValidation Set:\n')
    val_loss, val_accuracy = net.evaluate(x_val, y_val)
    # Our accuracy
    y_val_probabilities = net.predict(x_val)
    y_val_test = np.arange(0, val_datap, 1)
    for i in range(0, val_datap):
        y_val_test[i] = np.argmax(y_val_probabilities[i])
    acc = accuracy_score(y_val, y_val_test)
    print('\nAccuracy score:', acc, '\n')

y_test_probabilities = net.predict(x_test)

y_test = np.arange(0, test_datap, 1)
for i in range(0, test_datap):
    y_test[i] = np.argmax(y_test_probabilities[i])

# ---------------------------------------------------------------------------
# Create cvs file

if create_cvs == 1:
    Id_y_test = np.arange(train_datap, test_datap + train_datap, 1)
    df = pd.DataFrame({'Id': Id_y_test[:], 'y': y_test[:]})
    pd.DataFrame(df).to_csv('ypred.csv', index=False, header=True)
