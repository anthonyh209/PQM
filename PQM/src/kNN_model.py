import numpy as np

from regression_plotN import plot_kNN_errors
from regression_train_testN import cv_evaluation_kNN_model
from regression_multipoly import import_data
from regression_multipoly import new_conversion
from regression_modelsN import kNN_construct


def error_features_kNN(inputs, targets, folds, features_combination):
    """
    Function which goes through all possible features combinations and saves the resulting
    error test matrix into a csv file
    :param inputs:
    :param targets:
    :param folds:
    :param features_combination:
    :return:
    """
    N = 100

    error_matrix = np.zeros(shape=(len(features_combination)+1, N+1))
    counterx = 1
    for j in range(2,100):
        error_matrix[0][j] = counterx
        counterx += 1

    counter = 0
    for i, selection in enumerate(features_combination):
        print(counter)
        print(selection)
        x = input_construction_kNN(inputs, selection)
        print("hello")
        print(x.shape)
        error_matrix[i+1][0] = counter
        error_matrix[i+1][1] = magic(selection)
        error_mean = evaluate_for_various_k(x, targets, folds)
        error_matrix[i+1, 2:] = error_mean
        counter += 1

    np.savetxt('testkNN.csv', error_matrix, fmt='%.6f', delimiter=',',)

def evaluate_kNN(data_inputs, data_targets, folds, k):


    x = cv_evaluation_kNN_model(data_inputs, data_targets, folds, k)
    print("kNN error")
    print(x)



    # the rbf feature mapping performance

def input_construction_kNN(data_inputs, features=None):
    #print("hi")
    dim = len(features)
    N, dim2 = data_inputs.shape
    inputs = np.zeros(shape=(N, dim))
    for i, column_selection in enumerate(features):
        inputs[:, i] = data_inputs[:, column_selection]
    x = inputs
    return inputs

def evaluate_for_various_k(
        data_inputs, data_targets, folds):
    """
    """

    # the rbf feature mapping performance
    k_values = range(1,100)

    kNN_errors_list = []
    kNN_stdv_list = []


    for i,k in enumerate(k_values):
        kNN_errors = cv_evaluation_kNN_model(data_inputs, data_targets, folds, k)
        kNN_mean_error = np.mean(kNN_errors)
        kNN_stdv_error = np.std(kNN_errors)
        kNN_errors_list.append(kNN_mean_error)
        kNN_stdv_list.append(kNN_stdv_error)

    fig, ax = plot_kNN_errors(
        "$k$", k_values, kNN_errors_list)
    # we also want to plot a straight line showing the linear performance
    xlim = ax.get_xlim()
    fig.savefig("kNN_best.pdf", fmt="pdf")
    return kNN_errors_list, kNN_stdv_list

def kNN_apply_validation_set(data_pool, pool_targets, feature_combin, k):
    val_set, field_names = import_data("testing_data_split.csv", ";", True, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    inputs, targets = new_conversion(val_set)
    kNN_test_inputs = input_construction_kNN(inputs, feature_combin)
    kNN_prediction = kNN_construct(data_pool, pool_targets, k)
    train_predicts = kNN_prediction(kNN_test_inputs)
    pred_ys = train_predicts
    pred_targets = []
    for i in range(pred_ys.size):
        pred_targets.append(round(pred_ys[i],0))
    counter = 0
    for i in range(len(pred_targets)):
        if pred_targets[i] == targets[i]:
            counter +=1
    print("CORRECT: ",counter)
    print("WRONG: ", len(pred_targets)-counter)