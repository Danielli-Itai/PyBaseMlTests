#!/usr/bin/env python
# coding: utf-8

# ## 1 - Packages
# - [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
# - [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
# - [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
# - [PIL](http://www.pythonware.com/products/pil/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.
import os
import sys
import h5py
import numpy as np
from PIL import Image

from PyBase import Test
from PyBase import Files
from PyBaseMl import GraphPlot
from PyBaseMl import NeuralNetLx





#####################################################
#                                                   #
#                   Constants.                      #
#                                                   #
#####################################################
TEST_NAME = 'NeuralNetLxTest'
DATA_DIR = Test.DataDir(TEST_NAME)
OUT_DIR = Test.OutDir(TEST_NAME)
np.random.seed(1)





#####################################################
#                                                   #
#              Parmeters Initialization             #
#                                                   #
#####################################################
def initialize_parameters_test():
    np.random.seed(1)
    parameters = NeuralNetLx.initialize_parameters(3,2,1, seed=1)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    parameters = NeuralNetLx.initialize_parameters_deep([5,4,3],seed=3)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))





"""
X = np.array([[-1.02387576, 1.12397796],
[-1.62328545, 0.64667545],
[-1.74314104, -0.59664964]])
W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
b = np.array([[1]])
"""
def linear_forward_test_case():
    np.random.seed(1)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    return A, W, b

def linear_forward_test():
    A, W, b = linear_forward_test_case()
    Z, linear_cache = NeuralNetLx.linear_forward(A, W, b)
    print("Z = " + str(Z))





"""
X = np.array([[-1.02387576, 1.12397796],
[-1.62328545, 0.64667545],
[-1.74314104, -0.59664964]])
W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
b = 5
"""
def linear_activation_forward_test_case():
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def linear_activation_forward_test():
    A_prev, W, b = linear_activation_forward_test_case()
    A, linear_activation_cache = NeuralNetLx.linear_activation_forward(A_prev, W, b, activation = "sigmoid")
    print("With sigmoid: A = " + str(A))

    A, linear_activation_cache = NeuralNetLx.linear_activation_forward(A_prev, W, b, activation = "relu")
    print("With ReLU: A = " + str(A))
    return





def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return X, parameters

def L_model_forward_test_2hidden():
    X, parameters = L_model_forward_test_case_2hidden()
    AL, caches = NeuralNetLx.L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("Length of caches list = " + str(len(caches)))





def compute_cost_test_case():
    Y = np.asarray([[1, 1, 0]])
    aL = np.array([[.8, .9, 0.4]])
    return Y, aL

def compute_cost_test():
    Y, AL = compute_cost_test_case()
    print("cost = " + str(NeuralNetLx.compute_cost(AL, Y)))





"""
z, linear_cache = (np.array([[-0.8019545 ,  3.85763489]]), (np.array([[-1.02387576,  1.12397796],
   [-1.62328545,  0.64667545],
   [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), np.array([[1]]))
"""
def linear_backward_test_case():
    np.random.seed(1)
    dZ = np.random.randn(3,4)
    A = np.random.randn(5,4)
    W = np.random.randn(3,5)
    b = np.random.randn(3,1)

    linear_cache = (A, W, b)
    return dZ, linear_cache

def linear_backward_test():
    # Set up some test inputs
    dZ, linear_cache = linear_backward_test_case()
    dA_prev, dW, db = NeuralNetLx.linear_backward(dZ, linear_cache)
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))





"""
aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
"""
def linear_activation_backward_test_case():
    np.random.seed(2)
    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache


def linear_activation_backward_test():
    dAL, linear_activation_cache = linear_activation_backward_test_case()
    dA_prev, dW, db = NeuralNetLx.linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")

    dA_prev, dW, db = NeuralNetLx.linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))

    return





"""
X = np.random.rand(3,2)
Y = np.array([[1, 1]])
parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}

aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
       [ 0.02738759,  0.67046751], [ 0.4173048 ,  0.55868983]]),
np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
np.array([[ 0.]])),
np.array([[ 0.41791293,  1.91720367]]))])
"""
def L_model_backward_test_case():

    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)
    return AL, Y, caches


def L_model_backward_test():
    AL, Y_assess, caches = L_model_backward_test_case()
    grads = NeuralNetLx.L_model_backward(AL, Y_assess, caches)

    print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dA1 = " + str(grads["dA1"]))

    return





"""
 parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
     [-1.8634927 , -0.2773882 , -0.35475898],
     [-0.08274148, -0.62700068, -0.04381817],
     [-0.47721803, -1.31386475,  0.88462238]]),
'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
     [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
     [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
     [-0.16051336, -0.76883635, -0.23003072]]),
'b1': np.array([[ 0.], [ 0.], [ 0.], [ 0.]]),
'b2': np.array([[ 0.], [ 0.], [ 0.]]),
'b3': np.array([[ 0.], [ 0.]])}
 grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
     [ 0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ]]),
'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ]]),
'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
'da1': np.array([[ 0.70760786,  0.65063504], [ 0.17268975,  0.15878569],  [ 0.03817582,  0.03510211]]),
'da2': np.array([[ 0.39561478,  0.36376198], [ 0.7674101 ,  0.70562233], [ 0.0224596 ,  0.02065127], [-0.18165561, -0.16702967]]),
'da3': np.array([[ 0.44888991,  0.41274769], [ 0.31261975,  0.28744927], [-0.27414557, -0.25207283]]),
'db1': 0.75937676204411464,
'db2': 0.86163759922811056,
'db3': -0.84161956022334572}
 """
def update_parameters_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    np.random.seed(3)
    dW1 = np.random.randn(3, 4)
    db1 = np.random.randn(3, 1)
    dW2 = np.random.randn(1, 3)
    db2 = np.random.randn(1, 1)
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return parameters, grads

def update_parameters_test():
    parameters, grads = update_parameters_test_case()
    parameters = NeuralNetLx.update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))
    return


"""
Plots images where predictions and truth were different.
X -- dataset
y -- true labels
p -- predictions
"""



def load_data():
    train_dataset = h5py.File(os.path.join(DATA_DIR,'datasets/train_catvnoncat.h5'), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(os.path.join(DATA_DIR,'datasets/test_catvnoncat.h5'), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def TestLoadData() -> tuple:
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # The following code will show you an image in the dataset. Feel free to change the index and re-run the cell multiple times to see other images.
    index = 10
#    plt.imshow(train_x_orig[index])
    GraphPlot.ImageShow(train_x_orig[index], os.path.join(OUT_DIR, 'Data image'+ str(index)))
    print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

    # Explore your dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    num_py = train_x_orig.shape[2]
    num_pc = train_x_orig.shape[3]

    m_test = test_x_orig.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_py) + ", " + str(num_pc))
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))

    return (train_x_orig, train_y, test_x_orig, test_y, classes, num_px, num_py, num_pc)


def TestNormData(train_x_orig, test_x_orig) -> tuple:
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    return (train_x, test_x)


def TestImageClassification2L(train_x, train_y, test_x, test_y, n_x, n_h, n_y):
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    # Train your parameters (The cost should be decreasing)
    # It may take up to 5 minutes to run 2500 iterations.
    parameters, costs = NeuralNetLx.two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), learning_rate=0.0075, num_iterations=2500, print_cost=True)
    GraphPlot.CostsPlot(np.squeeze(costs), learning_rate=0.0075, name="Two layers Costs",  out_dir=OUT_DIR)

    # Use the trained parameters to classify images from the dataset.
    predictions_train = NeuralNetLx.predict(train_x, train_y, parameters)
    print('Accuracy 2L train ' + str(NeuralNetLx.Accuracy(train_x,train_y, predictions_train)))

    predictions_test = NeuralNetLx.predict(test_x, test_y, parameters)
    print('Accuracy 2L test ' + str(NeuralNetLx.Accuracy(test_x, test_y, predictions_test)))
    return


def TestImageClassification5L(train_x, train_y, test_x, test_y, classes, num_px, num_py, num_pc):
    layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model

    # You will now train the model as a 4-layer neural network.
    parameters, costs = NeuralNetLx.L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
    GraphPlot.CostsPlot(np.squeeze(costs), learning_rate=0.0075, name="Two layers Costs",  out_dir=OUT_DIR)

    pred_train = NeuralNetLx.predict(train_x, train_y, parameters)
    print('Accuracy 5L train ' + str(NeuralNetLx.Accuracy(train_x, train_y, pred_train)))

    pred_test = NeuralNetLx.predict(test_x, test_y, parameters)
    print('Accuracy 5L test ' + str(NeuralNetLx.Accuracy(test_x, test_y, pred_test)))

    GraphPlot.print_mislabeled_images(classes, test_x, test_y, pred_test)

    my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
    fname = os.path.join(DATA_DIR,"images", "my_image.jpg")
    image = np.array(Image.open(fname))
    my_image = np.array(Image.open(fname).resize(size=(num_px, num_py))).reshape((num_px * num_py * num_pc, 1))

    my_image = my_image / 255.
    my_predicted_image = NeuralNetLx.predict(my_image, my_label_y, parameters)

    GraphPlot.ImageShow(image,os.path.join(OUT_DIR, "my_image.jpg"))

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
    return




def SimpleTest():
    linear_forward_test()
    initialize_parameters_test()
    linear_activation_forward_test()
    L_model_forward_test_2hidden()
    compute_cost_test()
    linear_backward_test()
    linear_activation_backward_test()
    L_model_backward_test()
    update_parameters_test()
    return


def ImageCalssTest():
    train_x_orig, train_y, test_x_orig, test_y, classes, num_px, num_py, num_pc = TestLoadData()

    train_x, test_x = TestNormData(train_x_orig, test_x_orig)

    TestImageClassification2L(train_x, train_y, test_x, test_y, n_x=12288, n_h=7, n_y=1)

    TestImageClassification5L(train_x, train_y, test_x, test_y, classes, num_px, num_py, num_pc)
    return





#####################################################
#                                                   #
#                 Main entry point                  #
#                                                   #
#####################################################
if __name__ == "__main__":
    argsv = sys.argv[1:]

    if len(argsv) > 1:
        pass
    else:
        pass

    Files.DirClean(OUT_DIR)

    SimpleTest()

    ImageCalssTest()

    print('Test completed')