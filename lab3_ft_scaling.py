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
# TODO: enable lines below to see the scatterplots
# fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:,i],y_train)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("Price (1000's)")
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
# TODO: enable below to see the plot
# plot_cost_i_w(X_train,y_train,hist)

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

    # step one: remove the mean from each x
    # step two: divide by standard deviation
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)

# plot the steps involved in z score normalization
# TODO: uncomment to see the plots

# mu     = np.mean(X_train,axis=0)   
# sigma  = np.std(X_train,axis=0) 
# X_mean = (X_train - mu)
# X_norm = (X_train - mu)/sigma      

# fig,ax=plt.subplots(1, 3, figsize=(12, 3))
# ax[0].scatter(X_train[:,0], X_train[:,3])
# ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
# ax[0].set_title("unnormalized")
# ax[0].axis('equal')

# ax[1].scatter(X_mean[:,0], X_mean[:,3])
# ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
# ax[1].set_title(r"X - $\mu$")
# ax[1].axis('equal')

# ax[2].scatter(X_norm[:,0], X_norm[:,3])
# ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
# ax[2].set_title(r"Z-score normalized")
# ax[2].axis('equal')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.suptitle("distribution of features before, during, after normalization")
# plt.show()

# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# plot the distribution before normalization
fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()

# plot the distribution after normalization
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")
plt.show()

# normalized data for x-axis is centered around 0 now

# re-run gradient descent with normalized data
# choose a much larger alpha
alpha = 0.1
iterations = 1000
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, iterations, alpha)


