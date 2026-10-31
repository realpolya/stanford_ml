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
x = np.arange(0, 20, 1)

# add an engineered feature: x squared
X = x**2

# create a 2D matrix
X = X.reshape(-1,1)
iterations = 10000
alpha = 1e-5
model_w, model_b = run_gradient_descent_feng(X, y, iterations, alpha)

# plot the engineered feature
# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
# plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# ----------- selecting features -----------
# trying squared + cubed x
# seeing if it fits our data better

x = np.arange(0, 20, 1)
y = x**2

# engineer new features [x, x**2, x**3]
# np.c_ is a NumPy tool that combines arrays column by column
X = np.c_[x, x**2, x**3]
iterations = 10000
alpha = 1e-7

model_w,model_b = run_gradient_descent_feng(X, y, iterations, alpha)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
# plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# 0.08𝑥+0.54𝑥2+0.03𝑥3+0.0106
# Gradient descent is picking the 'correct' features for us 
# by emphasizing its associated parameter
# the x2 is the closest term to the actual relationship 
# so more weight is assigned to it

# when weight is too close to 0 or reaches 0
# it means the feature has little or no impact on fitting the model to data

# plot each x feature separately
X_features = ['x','x^2','x^3']
# fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X[:,i],y)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("y")
# plt.show()

# plot result:
# 𝑥2 feature mapped against the target value 𝑦 is linear
# linear regression can then generate a model using that feature

# -------- scaling features ---------
# see the range
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)
# see the range after normalization
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

# increase alpha due to normalization
X = zscore_normalize_features(X)
iterations = 100000
alpha = 1e-1

model_w, model_b = run_gradient_descent_feng(X, y, iterations, alpha)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# resulting weights
# w = [0.0000527, 113, 0.0000843]
# 113 emphasizes x squared as the dominant relationship

