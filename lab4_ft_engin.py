'''
Lab 4: Feature Engineering and Polynomial Regression
'''

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng

#reduce display precision on numpy arrays
np.set_printoptions(precision=2) 

# no amount of adjusting of w and b will achieve a fit 
# to a nonlinear curve

# ------- polynomial features ----------

# create target data
x = np.arange(0, 20, 1) # arange creates a range of evenly spaced numbers
y = x**2 + 1
X = x.reshape(-1,1) # reshape rearranges the array into a 2D column
iterations = 1000
alpha = 0.01

# run gradient descent
model_w, model_b = run_gradient_descent_feng(X, y, iterations, alpha)

# see the plotted results without feature engineering
# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
# plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()

# v2: create target data
x_v2 = np.arange(0, 20, 1)

# add an engineered feature: x squared
X_v2 = x_v2**2

# create a 2D matrix
X_v2 = X_v2.reshape(-1,1)
iterations = 10000
alpha = 1e-5
model_w, model_b = run_gradient_descent_feng(X_v2, y, iterations, alpha)

# plot the engineered feature
plt.scatter(x_v2, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x_v2, np.dot(X_v2,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()