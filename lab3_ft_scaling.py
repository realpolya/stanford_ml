'''

Feature Scaling and Learning Rate lab
multi-variable

Oct 27, 2026

'''

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import load_house_data, run_gradient_descent, norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

# 4 features – size, bedrooms, floors, and age
# prediction – housing price

# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# plot the dataset (each feature vs price)
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
# TODO: enable plt.show() below to see the first plot
#plt.show()

# experiment with learning rate
# it takes some testing to pick the most efficient learning rate
# v1: set alpha to 9.9e-7
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)

# plot the result of this alpha rate – the overshooting is visible
#plot_cost_i_w(X_train, y_train, hist)

# v2: set alpha to 9e-7 (smaller value)
_,_,hist = run_gradient_descent(X_train, y_train, 50, alpha = 9e-7)

# plot the v2 alpha rate – the cost is decreasing and is arriving at the global minimum
# but it is still oscillating back and forth
# the solution is converging
# plot_cost_i_w(X_train, y_train, hist)

# v3: set alpha to 1e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)
# this v3 solution is converging without oscillations
# dj_w0 is negative throughout this run
plot_cost_i_w(X_train,y_train,hist)

# feature scaling

# implement z-score normalization first
def zscore_normalize_features(X):
    '''
    computes X, zscore normalized by column

    Args:
        X – ndarray of shape (m,n) – input data with m examples and n features

    Returns:
        X_norm – ndarray of shape (m,n) – input data normalized by column
        mu – ndarray of shape (n,) – mean of each feature
        sigma – ndarray of shape (n,) – standard deviation of each feature
    '''

    # find the mean of each column/feature
    mu = np.mean(X, axis=0)

    # find the standard deviation of each column / feature
    sigma = np.std(X, axis=0)

    # element-wise, subtract mu for that column from each example
    # divide by std for that column according to the
    # z score normalization formula
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)


