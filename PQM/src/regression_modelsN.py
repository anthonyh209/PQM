import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


def ml_weightsN(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    #print("ml_weights function")
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1)) #making it into a vector
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets #design matrix
    dim, N=inputmtx.shape
    #print("computed weights")
    #print("np.array(weights).flatten()")
    return np.array(weights).flatten()

def regularised_ml_weightsN(
    inputmtx, targets, reg_param):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (reg_param)
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(reg_param*I + Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def linear_model_predictN(designmtx, weights):
    #print("hello")
    #print(designmtx)
    #Note placing of 1 as first value in the design matrix
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()

def kNN_construct(train_inputs, train_targets, k):

    #data_inputs = inputs[0:6, 0:5]
    #print(test_inputs)
    #train_inputs = inputs[1:6,0:5]
    trainN, trainD = train_inputs.shape

    def prediction_kNN(test_inputs):
        inputs = test_inputs.reshape((test_inputs.size, 1))
        N, dim = test_inputs.shape
        data_inputs = test_inputs
        #Create all euclidean distances for each data point
        distance_matrix = np.zeros(shape=(N, trainN))
        for i in range(N):
            test_inputs = data_inputs[i,:]
            # print("testinputs")
            # print(test_inputs)
            # print("train_inputs")
            # print(train_inputs)
            feature_distance = (train_inputs-test_inputs)**2 #correctly calculated
            #print(feature_distance)
            euclidean_distance = np.sum(feature_distance, axis=1)
            #append these distances to matrix
            x = euclidean_distance.transpose()
            distance_matrix[i,:] = euclidean_distance
            counter = 1
        z=np.argpartition(distance_matrix, k, axis=1)[:, :k]

        predicts = np.empty(N)
        counter = 0
        for i, neighbourhood in enumerate(z):
            #print("neighbourhood")
            #print(neighbourhood)
            # the neighbourhood is the indices of the closest inputs to xs[i]
            xx = np.mean(train_targets[neighbourhood])
            xaa = 1
            # the prediction is the mean of the targets for this neighbourhood
            predicts[i] = np.mean(train_targets[neighbourhood])
            #print(len(predicts))
            counter += 1
        x = 1
        return predicts
    return prediction_kNN

