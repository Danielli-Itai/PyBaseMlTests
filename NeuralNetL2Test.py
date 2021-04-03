#!/usr/bin/env python
# coding: utf-8





#Packages.
# - [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
# - [sklearn](http://scikit-learn.org/stable/) provides simple and efficient tools for data mining and data analysis.
# - [matplotlib](http://matplotlib.org) is a library for plotting graphs in Python.
import sys
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

from PyBase import Test
from PyBase import Files
from PyBaseMl import NeuralNetL2
from PyBaseMl import GraphPlot





#####################################################
#                                                   #
#               Constants                           #
#                                                   #
#####################################################
TEST_NAME='NeuralNetL2Test'
OUT_DIR = Test.OutDir(TEST_NAME)





#####################################################
#                                                   #
#               Calculate Layer Sizes               #
#                                                   #
#####################################################
def layer_sizes_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(5, 3)
    Y_assess = np.random.randn(2, 3)
    return X_assess, Y_assess

def testLayerSizes():
    X_assess, Y_assess = layer_sizes_test_case()
    (n_x, n_h, n_y) = NeuralNetL2.layer_sizes(X_assess, Y_assess)
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))
    return





#####################################################
#                                                   #
#               Initialize parameters               #
#                                                   #
#####################################################
def initialize_parameters_test_case():
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y

def testInitializeParameters():
    n_x, n_h, n_y = initialize_parameters_test_case()
    parameters = NeuralNetL2.initialize_parameters(n_x, n_h, n_y)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))





#####################################################
#                                                   #
#               Forward Propagate                   #
#                                                   #
#####################################################

def forward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    b1 = np.random.randn(4, 1)
    b2 = np.array([[-1.3]])

    parameters = {'W1': np.array([[-0.00416758, -0.00056267], [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747], [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': b1,
                  'b2': b2}

    return X_assess, parameters

def testForwaredPropagate():
    X_assess, parameters = forward_propagation_test_case()
    A2, cache = NeuralNetL2.forward_propagation(X_assess, parameters)

    # Note: we use the mean here just to make sure that your output matches ours.
    print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
    return





#####################################################
#                                                   #
#               Calculate cost                      #
#                                                   #
#####################################################
def compute_cost_test_case():
    np.random.seed(1)
    Y_assess = (np.random.randn(1, 3) > 0)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267], [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747], [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.], [0.], [0.], [0.]]),
                  'b2': np.array([[0.]])}

    a2 = (np.array([[0.5002307, 0.49985831, 0.50023963]]))

    return a2, Y_assess, parameters

def testCost():
    A2, Y_assess, parameters = compute_cost_test_case()
    print("cost = " + str(NeuralNetL2.compute_cost(A2, Y_assess, parameters)))
    return





#####################################################
#                                                   #
#               Backwords propagation               #
#                                                   #
#####################################################

def backward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = (np.random.randn(1, 3) > 0)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267], [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747], [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.], [0.], [0.], [0.]]),
                  'b2': np.array([[0.]])}

    cache = {'A1': np.array([[-0.00616578, 0.0020626, 0.00349619], [-0.05225116, 0.02725659, -0.02646251],
                             [-0.02009721, 0.0036869, 0.02883756], [0.02152675, -0.01385234, 0.02599885]]),
             'A2': np.array([[0.5002307, 0.49985831, 0.50023963]]),
             'Z1': np.array([[-0.00616586, 0.0020626, 0.0034962],  [-0.05229879, 0.02726335, -0.02646869],
                             [-0.02009991, 0.00368692, 0.02884556],[0.02153007, -0.01385322, 0.02600471]]),
             'Z2': np.array([[0.00092281, -0.00056678, 0.00095853]])}
    return parameters, cache, X_assess, Y_assess

def testBackPropagate():
    parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
    grads = NeuralNetL2.backward_propagation(parameters, cache, X_assess, Y_assess)
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dW2 = "+ str(grads["dW2"]))
    print ("db2 = "+ str(grads["db2"]))
    return





#####################################################
#                                                   #
#               Update parameters                   #
#                                                   #
#####################################################
def update_parameters_test_case():
    parameters = {'W1': np.array([[-0.00615039, 0.0169021], [-0.02311792, 0.03137121],
                                  [-0.0169217, -0.01752545],[0.00935436, -0.05018221]]),
                  'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
                  'b1': np.array([[-8.97523455e-07], [8.15562092e-06], [6.04810633e-07], [-2.54560700e-06]]),
                  'b2': np.array([[9.14954378e-05]])}

    grads = {'dW1': np.array([[0.00023322, -0.00205423], [0.00082222, -0.00700776], [-0.00031831, 0.0028636],
                              [-0.00092857, 0.00809933]]),
             'dW2': np.array([[-1.75740039e-05, 3.70231337e-03, -1.25683095e-03, -2.55715317e-03]]),
             'db1': np.array([[1.05570087e-07], [-3.81814487e-06], [-1.90155145e-07], [5.46467802e-07]]),
             'db2': np.array([[-1.08923140e-05]])}
    return parameters, grads

def testUpdateParamaters():
    parameters, grads = update_parameters_test_case()
    parameters = NeuralNetL2.update_parameters(parameters, grads)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))





#####################################################
#                                                   #
#               Neural netwok model                 #
#                                                   #
#####################################################
def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = (np.random.randn(1, 3) > 0)
    return X_assess, Y_assess

def testNnModel():
    X_assess, Y_assess = nn_model_test_case()
    parameters = NeuralNetL2.nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    return





#####################################################
#                                                   #
#               Prediction                          #
#                                                   #
#####################################################
# FUNCTION: predict
def predict_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039, 0.0169021], [-0.02311792, 0.03137121], [-0.0169217, -0.01752545],
                                  [0.00935436, -0.05018221]]),
                  'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
                  'b1': np.array([[-8.97523455e-07], [8.15562092e-06], [6.04810633e-07], [-2.54560700e-06]]),
                  'b2': np.array([[9.14954378e-05]])}
    return parameters, X_assess

def testPredict():
    parameters, X_assess = predict_test_case()
    predictions = NeuralNetL2.predict(parameters, X_assess)
    print("predictions mean = " + str(np.mean(predictions)))
    return





#####################################################
#                                                   #
#                       Datasets                    #
#                                                   #
#####################################################
def load_planar_dataset():
    np.random.seed(1)
    m = 400                 # number of examples
    N = int(m / 2)          # number of points per class
    D = 2                   # dimensionality

    X = np.zeros((m, D))                    # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')     # labels vector (0 for red, 1 for blue)
    a = 4                                   # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2 # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2                        # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure





#####################################################
#                                                   #
#               Logistic Regression                 #
#                                                   #
#####################################################

from PyBaseMl import BaseMl
def LogRegression(X, Y):
    # You have:
    #     - a numpy-array (matrix) X that contains your features (x1, x2)
    #     - a numpy-array (vector) Y that contains your labels (red:0, blue:1).
    shape_X = X.shape
    shape_Y = Y.shape
    m = X.shape[1]  # training set size
    print('The shape of X is: ' + str(shape_X))
    print('The shape of Y is: ' + str(shape_Y))
    print('I have m = %d training examples!' % (m))

    # Simple Logistic Regression
    # Train the logistic regression classifier
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T.ravel())

    # Plot the decision boundary for logistic regression
    GraphPlot.plot_decision_boundary(lambda x: clf.predict(x), X, Y, OUT_DIR, "Logistic Regression")

    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % BaseMl.Accuracy(Y, LR_predictions) + '%')
    return





#####################################################
#                                                   #
#               Neural Network                      #
#                                                   #
#####################################################
def NeuralNetwork(X,Y):
    # Build a model with a n_h-dimensional hidden layer
    parameters = NeuralNetL2.nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

    # Plot the decision boundary
    GraphPlot.plot_decision_boundary(lambda x: NeuralNetL2.predict(parameters, x.T), X, Y, OUT_DIR, "Decision Boundary for hidden layer size " + str(4))

    # Print accuracy
    predictions = NeuralNetL2.predict(parameters, X)
    print ('Accuracy: %d' % BaseMl.Accuracy(Y,predictions.T) + '%')





#####################################################
#                                                   #
#               Neural Network                      #
#                                                   #
# Tuning hidden layer size                          #
# This may take about 2 minutes to run              #
#####################################################
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
def NeuralNetworkHlayers(X,Y, hidden_layer_sizes):
    #plt.figure(figsize=(16, 32))
    for i, n_h in enumerate(hidden_layer_sizes):
        #plt.subplot(5, 2, i+1)
        #plt.title('Hidden Layer of size %d' % n_h)
        parameters = NeuralNetL2.nn_model(X, Y, n_h, num_iterations = 5000)
        GraphPlot.plot_decision_boundary(lambda x: NeuralNetL2.predict(parameters, x.T), X, Y, OUT_DIR, "Decision Boundary for hidden layers " + str(n_h))
        predictions = NeuralNetL2.predict(parameters, X)
        accuracy = BaseMl.Accuracy(Y, predictions.T)
        print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    return





#####################################################
#                                                   #
#                 Run all tests                     #
#                                                   #
#####################################################
def RunAllTests():
    testLayerSizes()
    testInitializeParameters()
    testForwaredPropagate()
    testCost()
    testBackPropagate()
    testUpdateParamaters()
    testNnModel()
    testPredict()
    return





#####################################################
#                                                   #
#              Run all classifications              #
#                                                   #
#####################################################
def RunAllNets(dataset = None):
    if not dataset:
        X, Y = load_planar_dataset()
        dataset = "plannar"
    else:
        # Datasets
        noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
        datasets = {"noisy_circles": noisy_circles, "noisy_moons": noisy_moons, "blobs": blobs, "gaussian_quantiles": gaussian_quantiles}
        X, Y = datasets[dataset]
        X, Y = X.T, Y.reshape(1, Y.shape[0])
        # make blobs binary
        if dataset == "blobs":
            Y = Y % 2
    # Visualize the data
    GraphPlot.plot_dataset(X_h =X[0, :], X_v=X[1, :], Y_c=Y[0, :], scatter =40, path=OUT_DIR, title=dataset)

    LogRegression(X, Y)

    NeuralNetwork(X, Y)

    NeuralNetworkHlayers(X, Y, hidden_layer_sizes)
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

    RunAllTests()

    np.random.seed(1)  # set a seed so that the results are consistent

    RunAllNets()
#    RunAllNets("noisy_circles")
#    RunAllNets("noisy_moons")
#    RunAllNets("blobs")
#    RunAllNets("gaussian_quantiles")

