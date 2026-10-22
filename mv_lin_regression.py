# multiple variable linear regression lab
# October 22, 2026

import copy, math
import numpy as np
import matplotlib.pyplot as plt

# set the graph style
plt.style.use('./deeplearning.mplstyle')

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

# ---------- HOUSING PRICE PREDICTION -------------
# four features: size, bedrooms, floors, age

# create numpy arrays with out dataset
# example - house 1: 2104 sqft, 5 bedrooms, 1 floor, 45 years old, cost: 460,000 USD
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178]) # end result – the prices of the houses

print(f"X shape: {X_train.shape}, X Type: {type(X_train)}")
print(X_train)
print(f"y shape: {y_train.shape}, y Type:{type(y_train)}")
print(y_train)

# Parameter vector w, b

# establish the initial parameter values near optimal
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")
