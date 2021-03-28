#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with a Neural Network mindset
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image

from PyBase import Test
from PyBase import Files
from PyBaseMl import BaseMl
from PyBaseMl import LogReg





#Constants.
TEST_NAME = 'LogRegTest'
DATA_DIR = Test.DataDir(TEST_NAME)
OUT_DIR = Test.OutDir(TEST_NAME)
PREDICT_PROB = 0.5





#####################################################
#                                                   #
#   Logistic regression simple test.                #
#                                                   #
#####################################################

def ParamsCompare(params1,params2):
    assert (params1['w'].shape == params2['w'].shape)
    assert (params1['w'].any() == params2['w'].any())
    assert (params1['b'].shape == params2['b'].shape)
    assert (params1['b'].any() == params2['b'].any())
    return

def SimpleTest():
    print("sigmoid([0, 2]) = " + str(BaseMl.sigmoid(np.array([0, 2]))))

    dim = 2
    w, b = LogReg.ParamsInitialize(dim, 1)
    print("w = " + str(w))
    print("b = " + str(b))

    w, b, X, Y = np.array([[1.], [2.]]), np.array([2.]), np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    grads, cost = LogReg.propagate(w, b, X, Y)
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print("cost = " + str(cost))

    params, grads, costs = LogReg.optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
    LogReg.ParamsSave(os.path.join(OUT_DIR,"SimpleTestParams"), params)
    params1 = LogReg.ParamsRead(os.path.join(OUT_DIR,"SimpleTestParams"))
    ParamsCompare(params, params1)

    print("w = " + str(params["w"]))
    print("b = " + str(params["b"]))
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))

    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    print("predictions = " + str(LogReg.predict(w, b, X, PREDICT_PROB)))
    return





#####################################################
#                                                   #
#   Logistic Regression image classification.       #
#                                                   #
#####################################################
# Show an image and save it in the plot directory.
def ImageShow(image, name):
    plt.imshow(image)
    plt.savefig(os.path.join(OUT_DIR, name))
    plt.close()
    return



# Loading the data (cat/non-cat)
def DatasetLoad():
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



# Many software bugs in deep learning come from having matrix/vector dimensions that don't fit.
# If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating many bugs.
#     - m_train (number of training examples)
#     - m_test (number of test examples)
#     - num_px (= height = width of a training image)
#train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3).
# Expected Output: m_train 209, m_test 50, num_px 64.
def DatasetSizes(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes):
    m_train = train_set_x_orig.shape[0]
    print("Number of training examples: m_train = " + str(m_train))

    m_test = test_set_x_orig.shape[0]
    print ("Number of testing examples: m_test = " + str(m_test))

    num_pxy = train_set_x_orig.shape[1]
    print ("Height of each image: num_px = " + str(num_pxy))

    num_py = train_set_x_orig.shape[2]
    print ("Width of each image: num_py = " + str(num_py))

    num_pc = train_set_x_orig.shape[3]
    print ("Color of each image: num_pc = " + str(num_pc))

    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    return m_train, m_test, num_pxy, num_py, num_pc



# For convenience, reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px * num_px * 3, 1).
# Reshape the training and test examples
def DatasetFlatten(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y):
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
    return train_set_x_flatten,test_set_x_flatten



# Common preprocessing step is to center and standardize your dataset,
# But for picture datasets, it works almost as well to just divide every row of the dataset by 255.
def DatasetStandardise(train_set_x_flatten, test_set_x_flatten):
    train_set_x = (train_set_x_flatten - train_set_x_flatten.mean())/train_set_x_flatten.std()  #train_set_x = train_set_x_flatten / 255.
    assert(train_set_x.shape == train_set_x_flatten.shape)

    test_set_x = (test_set_x_flatten-test_set_x_flatten.mean())/test_set_x_flatten.std()        #test_set_x_flatten / 255.
    assert(test_set_x.shape == test_set_x_flatten.shape)
    return train_set_x, test_set_x



def ImagePredict(classes, fname:str, num_py, num_px, num_pc, model):
    # We preprocess the image to fit your algorithm.
    image = np.array(Image.open(fname))
    my_image = np.array(Image.open(fname).resize(size=(num_py, num_px))).reshape((1, num_py*num_px*num_pc)).T
    my_image = LogReg.Standardise(my_image)

    # Predict the image class.
    prediction = LogReg.predict(model[LogReg.MODEL_PARAMS]["w"], model[LogReg.MODEL_PARAMS]["b"], my_image, PREDICT_PROB)
    predic_class = classes[int(np.squeeze(prediction)),].decode("utf-8")

    return image, prediction, predic_class


def GetClass(classes, lables_set_y, pic_index):
    class_lable = classes[int(lables_set_y[0, pic_index])].decode("utf-8")
    return(class_lable)




#####################################################
#                                                   #
#       Logistic Regression image Modeling.         #
#                                                   #
#####################################################
def LogRegModel():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = DatasetLoad()

    # Example of a picture visualization:
    pic_index = 30
    ImageShow(train_set_x_orig[pic_index],'example-pic.jpg')
    print("Train picture:" + str(pic_index) + " is a " + GetClass(classes,train_set_y,pic_index))

    # Preparing the images data for logistic regression.
    m_train, m_test, num_py, num_px, num_pc = DatasetSizes(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)
    train_set_x_flatten, test_set_x_flatten = DatasetFlatten(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y)
    train_set_x, test_set_x = DatasetStandardise(train_set_x_flatten, test_set_x_flatten)

    # Run the model.
    model = LogReg.regression_rmodel(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
    LogReg.ParamsSave(os.path.join(OUT_DIR,"LogRegModelParams"), model[LogReg.MODEL_PARAMS])
    params = LogReg.ParamsRead(os.path.join(OUT_DIR,"LogRegModelParams"))
    ParamsCompare(params, model[LogReg.MODEL_PARAMS])

    LogReg.CostsPlot(np.squeeze(model['costs']), model["learning_rate"], 'costs.jpg', OUT_DIR)

    # Example of a picture that is wrongly classified.
    ImageShow(test_set_x[:, 1].reshape((num_py, num_px, num_pc)), "not-cat-eror.jpg")
    print("Test picture "+str(pic_index)+": is a \"" + GetClass(classes,test_set_y, pic_index) + "\", you predicted that it is a \"" + GetClass(classes,model["Y_prediction_test"],pic_index)+"\"")

    # Example of your picture prediction.
    image, prediction, predic_class = ImagePredict(classes, os.path.join(DATA_DIR,"images/my_cat.jpg"), num_py, num_px, num_pc, model)
    print("my_cat.jpg y=" + str(np.squeeze(prediction)) + ", your algorithm predicts a \"" + predic_class + "\" picture.")
    ImageShow(image,'my_cat.jpg')

    # Example of your picture prediction.
    image, prediction, predic_class = ImagePredict(classes, os.path.join(DATA_DIR,"images/my_none_cat.jpg"), num_py, num_px, num_pc, model)
    print("my_none_cat.jpg y=" + str(np.squeeze(prediction)) + ", your algorithm predicts a \"" + predic_class + "\" picture.")
    ImageShow(image,'my_none_cat.jpg')
    return




#####################################################
#                                                   #
#   Logistic Regression image classification.       #
#                                                   #
#####################################################
def LogRegLearningRates():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = DatasetLoad()

    # Preparing the images data for logistic regression.
    train_set_x_flatten, test_set_x_flatten = DatasetFlatten(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y)
    train_set_x, test_set_x = DatasetStandardise(train_set_x_flatten, test_set_x_flatten)


    # Example of multiple learning rates.
    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    models = LogReg.LearningRates(train_set_x, train_set_y, test_set_x, test_set_y, learning_rates, PREDICT_PROB)
    LogReg.LearningRatesCostsPlot(models, learning_rates, 'learning rates', OUT_DIR)

    return





if __name__ == "__main__":
    argsv = sys.argv[1:]

    if len(argsv) > 1:
        pass
    else:
        pass

    Files.DirClean(OUT_DIR)

    SimpleTest()

    LogRegModel()

    LogRegLearningRates()




