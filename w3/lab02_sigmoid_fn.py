"""
Week 3 Lab 02 – Sigmoid Function & Logistic Regression
"""

import numpy as np
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

