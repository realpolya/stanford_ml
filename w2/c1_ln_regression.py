'''
Practice Lab: Linear Regression
'''

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
# %matplotlib inline

# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    
    # we have only one w, so one loop should be sufficient
    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        total_cost += (f_wb - y[i])**2
    
    total_cost /= 2*m
    
    ### END CODE HERE ### 

    return total_cost


# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ###
    
    # only one feature w
    
    # loop over m examples
    for i in range(m):
        
        err = (np.dot(x[i], w) + b) - y[i]
        
        dj_dw += err * x[i]
        dj_db += err # no need to multiply by x
    
    # divide by number of examples
    dj_dw /= m
    dj_db /= m
    
    ### END CODE HERE ### 
        
    return dj_dw, dj_db