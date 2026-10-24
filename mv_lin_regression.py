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

# a dot product vector function to make prediction
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

# make a prediction for the first row using dot product
f_wb = predict(X_train[0,:], w_init, b_init)
print(f"dot product: f_wb shape {f_wb.shape}, prediction: {f_wb}")

# compute cost with multiple variables
def compute_cost(X, y, w, b):
    """
    compute cost via cost function J summing up the squared 
    differences between predictions and actual results and 
    dividing the sum by 2m where m is number of examples

    args:
    X – ndarray of shape (m,n) – m examples with n features
    y – ndarray of shape (m,) – target values
    w – ndarray of shape (n,) – model parameters
    b – scalar model parameter

    returns:
    cost – scalar
    """

    # number of training examples extracted from the np array
    m = X.shape[0]

    # initialize cost to its minimum
    cost = 0.0

    # loop through each row in X sample adding to the cost 
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2

    # divide the cost sum by 2m
    cost /= 2*m
    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost J at optimal w: {cost}')

# gradient descent with multiple variables
def compute_gradient(X, y, w, b):
    """
    computes gradient for linear regression
    args:
    X – ndarray of shape (m,n) – m examples with n features
    y – ndarray of shape (m,) – target values
    w – ndarray of shape (n,) – model parameters
    b – scalar model parameter

    returns:
    dj_dw – ndarray of shape (n,) the gradient of the cost with respect to w
    dj_db – scalar, the gradient of the cost with respect to b
    """

    # get number of examples, number of features
    m,n = X.shape

    # create an array with number of zeros equal to n
    dj_dw = np.zeros((n,))
    dj_db = 0.

    # outer loop over m examples
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]

        # inner loop over n features
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    # divide by number of examples
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

# compute and display gradient
tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

# gradient descent with convergence and learning rate alpha
def gradient_descent(X, y, w_in, b_in, cost_fn, gradient_fn, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b
    by taking num_iters gradient steps with learning rate alpha.
    Args:
    X – ndarray of shape (m,n) – m examples with n features
    y – ndarray of shape (m,) – target values
    w_in – ndarray of shape (n,) – initial model parameters
    b_in – scalar initial model parameter
    cost_fn – function to compute cost
    gradient_fn – function to compute the gradient
    alpha – float – learning rate
    num_iters – integer – number of iterations to run gradient descent

    returns:
    w – ndarray of shape (n,) – updated values of parameters
    b – scalar – updated value of parameter
    """

    # define an array for storing history to create a graph later
    J_history = []

    # avoid mofidying global w within function
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):

        # calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_fn(X, y, w, b)

        # update parameters using w, b, alpha and gradient
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # save cost J at each iteration
        if i < 100000: # prevent resource exhaustion
            J_history.append(cost_fn(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0

# gradient descent settings
iterations = 1000
alpha = 5.0e-7

# run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")


# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()