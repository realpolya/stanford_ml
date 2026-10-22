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

# single prediction via loop function
def predict_single_loop(x, w, b):
    """
    single predict using linear regression using loop

    Args:
    x – ndarray of shape (n,) with multiple features
    w – ndarray of shape (n,) model parameters
    b – scalar model parameter

    Returns:
    p – scalar prediction
    """

    # first element of the shape yields number of rows
    n = x.shape[0]

    # p is initiated to 0
    p = 0

    for i in range(n):
        # multiply w * x for this index
        p_i = x[i] * w[i]

        # add the result to the p
        p += p_i

    # add the scalar model parameter
    p += b

    return p

# make a prediction for the first row of the training data
f_wb = predict_single_loop(X_train[0,:], w_init, b_init)
print(f"loop: f_wb shape {f_wb.shape}, prediction: {f_wb}")

# a dot product vector function
def predict(x, w, b):
    """
    single predict using linear regression using numpy dot product

    Args:
    x – ndarray of shape (n,) with multiple features
    w – ndarray of shape (n,) model parameters
    b – scalar model parameter

    Returns:
    p – scalar prediction
    """

    p = np.dot(x, w) + b
    return p

f_wb = predict(X_train[0,:], w_init, b_init)
print(f"dot product: f_wb shape {f_wb.shape}, prediction: {f_wb}")