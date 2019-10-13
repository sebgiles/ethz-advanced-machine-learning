# TASK 1

# import all necessary libraries
import numpy as np
import pandas as pd

# locate the .csv files in the same directory as the main.py
x_train_dir = 'X_train.csv'
y_train_dir = 'y_train.csv'
x_test_dir = 'X_test.csv'

# import the data from the .csv files
x_train = pd.read_csv(x_train_dir)
y_train = pd.read_csv(y_train_dir)
x_test = pd.read_csv(x_test_dir)

