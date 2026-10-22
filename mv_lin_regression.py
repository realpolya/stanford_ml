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



