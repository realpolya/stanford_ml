"""
Week 3 Lab 02 – Sigmoid Function & Logistic Regression
"""

import numpy as np
np.set_printoptions(suppress=True, precision=6)

import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh

plt.style.use('/Users/polinastepanova/Desktop/comp_design_coding/stanford_ml/deeplearning.mplstyle')

# for sigmoid function – what is z?
# z is the output of the linear regression model wx + b
# z can be scalar, vector – sigmoid function should cover both of these possibilities

# vector is a list
# matrix is a two-dimensional list
# matrix is a collection of training examples each containing multiple features

# numpy has exp() function
# calculates the exponential on all elements in the input array
# raise the Euler's number to the power of the input

# input is an array
input_array = np.array([1, 2, 3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# input is a single number
input_val = 1
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

# write sigmoid function for both scalar and vector z
def sigmoid_fn(z):
    '''
    Compute the sigmoid of z

    args: z – ndarray – a scalar or numpy array of any size
    returns: g – ndarray – sigmoid(z) with the same shape as z
    '''

    g = 1/(1+np.exp(-z))

    return g

# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid_fn(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)

plt.show()

# sigmoid function remaps the result we get from linear regression
# into probability between 0 and 1

# tumor classification
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

plt.close('all') 
# addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)