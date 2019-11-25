import numpy as np

f = open("filename.txt")
f.readline()  # skip the header
data = np.loadtxt(f)