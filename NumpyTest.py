#!/usr/bin/env python
# coding: utf-8

import sys
import math
import numpy as np





# Python Basics with Numpy (optional assignment)
# Any time you need more info on a numpy function, we encourage you to look at the official documentation
# https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.exp.html.





# ### sigmoid function ###
# returns the sigmoid of a real number x. Using math.exp(x) for the exponential function.
"""
Compute sigmoid of x.
Arguments:  x -- A scalar
Return:     s -- sigmoid(x)
"""
def basic_sigmoid(x):
    assert(type(x)==int)
    s = 1/(1 + math.exp(-x))
    return s










# example of np.exp returns (exp(1), exp(2), exp(3))
def npExp(x):
    return(np.exp(x))


# Vector Add operation
def npAdd(x):
    return(x + 3)



"""
Compute the sigmoid of x
Arguments: x -- A scalar or numpy array of any size
Return: s -- sigmoid(x)
"""
# **Expected Output**:
# sigmoid([1,2,3])= array([ 0.73105858,  0.88079708,  0.95257413])
def npSigmoid(x):
    s =  1/(1 + np.exp(-x))
    return s



# Sigmoid derivative
"""
Compute the gradient of the sigmoid function with respect to its input x.
Arguments: x -- A scalar or numpy array
Return: ds -- Your computed gradient.
"""
# **Expected Output**:
# sigmoid_derivative([1,2,3]) = [ 0.19661193  0.10499359  0.04517666]
def npSigmoid_derv(x):
    s = npSigmoid(x)
    ds = s * (1 - s)
    return ds



# Reshaping arrays
# 
# Two common numpy functions used in deep learning are
#   np.shape get the shape (dimension) of a matrix/vector X - https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
#   np.reshape() to reshape X into some other dimension - https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

# image2vector
"""
Argument: image -- a numpy array of shape (length, height, depth)
Returns: v -- a vector of shape (length*height*depth, 1)
"""


def image2vector(image):
    size = 1
    for i in image.shape:
        size *= i
        print("size = " + str(i))
    v = image.reshape(size,1)
    return v

# ### 1.4 - Normalizing rows
# 
# Another common technique we use in Machine Learning and Deep Learning is to normalize our data.
# It often leads to a better performance because gradient descent converges faster after normalization.
# Here, by normalization we mean changing x to dividing each row vector of x by its norm.

# normalizeRows
"""
Implement a function that normalizes each row of the matrix x (to have unit length).
Argument:   x -- A numpy matrix of shape (n, m)
Returns:    x -- The normalized (by row) numpy matrix.
"""
def normalizeRows(x):
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    # Divide x by its norm.
    x = x/x_norm
    return x


# **Note**:
# In normalizeRows(), you can try to print the shapes of x_norm and x, and then rerun the assessment. You'll find out that they have different shapes. This is normal given that x_norm takes the norm of each row of x. So x_norm has the same number of rows but only 1 column. So how did it work when you divided x by x_norm? This is called broadcasting and we'll talk about it now! 

# Broadcasting and the softmax
# #### Note
# Softmax should be performed for all features of each training example
# so softmax would be performed on the columns (once we switch to that representation later in this course).
# 
# However, in this coding practice, we're just focusing on getting familiar with Python, so we're using the common math notation $m \times n$  
# where $m$ is the number of rows and $n$ is the number of columns.

# GRADED FUNCTION: softmax

"""Calculates the softmax for each row of the input x.
The code work for a row vector and also for matrices of shape (m,n).
Argument: x -- A numpy matrix of shape (m,n)
Returns: s -- A numpy matrix equal to the softmax of x, of shape (m,n)
"""
# **Expected Output**:  [   [  9.80897665e-01   8.94462891e-04   1.79657674e-02   1.21052389e-04   1.21052389e-04]
#                           [  8.78679856e-01   1.18916387e-01   8.01252314e-04   8.01252314e-04   8.01252314e-04]]

def npSoftmax(x):

    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum
    return s





# **Note**:
# - If you print the shapes of x_exp, x_sum and s above and rerun the assessment cell, you will see that x_sum is of shape (2,1) while x_exp and s are of shape (2,5). **x_exp/x_sum** works due to python broadcasting.
# ## 2) Vectorization
# In deep learning, you deal with very large datasets. Hence, a non-computationally-optimal function can become a huge bottleneck in your algorithm and can result in a model that takes ages to run. To make sure that your code is  computationally efficient, you will use vectorization. For example, try to tell the difference between the following implementations of the dot/outer/elementwise product.
# In[42]:


import time


### VECTORIZED DOT PRODUCT OF VECTORS ###
def VecDotProd(x1,x2):
    tic = time.process_time()
    dot = np.dot(x1,x2)
    toc = time.process_time()
    print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
def VecDotProdLoop(x1,x2):
    tic = time.process_time()
    dot = 0
    for i in range(len(x1)):
        dot+= x1[i]*x2[i]
    toc = time.process_time()
    print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")




### VECTORIZED OUTER PRODUCT ###
def VecOuterProd(x1,x2):
    tic = time.process_time()
    outer = np.outer(x1,x2)
    toc = time.process_time()
    print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
def VecOuterProdLoop(x1,x2):
    tic = time.process_time()
    outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
    for i in range(len(x1)):
        for j in range(len(x2)):
            outer[i,j] = x1[i]*x2[j]
    toc = time.process_time()
    print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")



### VECTORIZED ELEMENTWISE MULTIPLICATION ###
def VecMult(x1,x2):
    tic = time.process_time()
    mul = np.multiply(x1,x2)
    toc = time.process_time()
    print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
def VecMultLoop(x1,x2):
    tic = time.process_time()
    mul = np.zeros(len(x1))
    for i in range(len(x1)):
        mul[i] = x1[i]*x2[i]
    toc = time.process_time()
    print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")




### VECTORIZED GENERAL DOT PRODUCT ###
def VecDotProd1(x1):
    W = np.random.rand(3, len(x1))  # Random 3*len(x1) numpy array
    tic = time.process_time()
    dot = np.dot(W,x1)
    toc = time.process_time()
    print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
def VecDotProd1Loop(x1):
    W = np.random.rand(3, len(x1))  # Random 3*len(x1) numpy array
    tic = time.process_time()
    gdot = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        for j in range(len(x1)):
            gdot[i] += W[i,j]*x1[j]
    toc = time.process_time()
    print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")





# L1 - numpy vectorized version of the L1 loss.
"""
Arguments: yhat -- vector of size m (predicted labels)
            y -- vector of size m (true labels)
Returns:    loss -- the value of the L1 loss function defined above
"""
def LossL1(yhat, y):
    loss = np.sum(np.abs(y-yhat))
    return loss

# L2 - numpy vectorized version of the L2 loss.
"""
Arguments:  yhat -- vector of size m (predicted labels)
            y -- vector of size m (true labels)
Returns:    loss -- the value of the L2 loss function defined above
"""
def LossL2(yhat, y):
    diff = y-yhat
    loss = np.sum(np.dot(diff,diff))
    return loss

# Congratulations on completing this assignment. We hope that this little warm-up exercise helps you in the future assignments, which will be more exciting and interesting!

# <font color='blue'>
# **What to remember:**
# - Vectorization is very important in deep learning. It provides computational efficiency and clarity.
# - You have reviewed the L1 and L2 loss.
# - You are familiar with many numpy functions such as np.sum, np.dot, np.multiply, np.maximum, etc...

def main():
    basic_sigmoid(3)
    try:
        # Actually, we rarely use the "math" library in deep learning because the inputs of the functions are real numbers. In deep learning we mostly use matrices and vectors. This is why numpy is more useful.
        ### One reason why we use "numpy" instead of "math" in Deep Learning ###
        x = [1, 2, 3]
        basic_sigmoid(x)  # you will see this give an error when you run it, because x is a vector.
    except Exception:
        pass

    x = np.array([1, 2, 3])
    print(npExp(x))
    print(npAdd(x))
    print(npSigmoid(x))
    print("sigmoid_derivative(x) = " + str(npSigmoid_derv(x)))

    # This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
    image = np.array([[[0.67826139, 0.29380381],
                       [0.90714982, 0.52835647],
                       [0.4215251, 0.45017551]],

                      [[0.92814219, 0.96677647],
                       [0.85304703, 0.52351845],
                       [0.19981397, 0.27417313]],

                      [[0.60659855, 0.00533165],
                       [0.10820313, 0.49978937],
                       [0.34144279, 0.94630077]]])
    print("image2vector(image) = " + str(image2vector(image)))

    x = np.array([ [0, 3, 4], [1, 6, 4]])
    print("normalizeRows(x) = " + str(normalizeRows(x)))

    x = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])
    print("softmax(x) = " + str(npSoftmax(x)))

    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
    VecDotProd(x1, x2)
    VecDotProdLoop(x1, x2)

    VecOuterProd(x1,x2)
    VecOuterProdLoop(x1, x2)

    VecMult(x1, x2)
    VecMultLoop(x1, x2)

    VecDotProd1(x1)
    VecDotProd1Loop(x1)

    #Loss functions.
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(LossL1(yhat, y)))   # Expected Output: 1.1
    print("L2 = " + str(LossL2(yhat, y)))   # Expected Output: 0.43
    return



if __name__ == "__main__":
    argsv = sys.argv[1:]

    if len(argsv) > 1:
        pass
    else:
        pass
    main()

