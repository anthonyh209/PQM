import csv
import numpy as np
from regression_modelsN import ml_weightsN
from regression_plotN import plot_train_test_errors
from regression_modelsN import linear_model_predictN

# for design matrix
from models_polynomial import poly_designmtx

# for cv evaluation
from regression_train_testN import cv_evaluation_linear_modelN

def error_features_poly(inputs, targets, folds, features_combination):
    """
    Function which goes through all possible features combinations and saves the resulting
    error test matrix into a csv file
    :param inputs:
    :param targets:
    :param folds:
    :param features_combination:
    :return:
    """


    error_matrix = np.zeros(shape=(len(features_combination), 9))
    counter = 0
    for i, selection in enumerate(features_combination):
        print(counter)
        print(selection)
        error_matrix[i][0] = counter
        error_matrix[i][1] = magic(selection)
        error_mean = poly_evaluate(inputs, targets, selection, folds)
        error_matrix[i, 2:] = error_mean
        counter += 1

    np.savetxt('fErrorSet1.csv', error_matrix, fmt='%.6f', delimiter=',',
               header="#index, #combination, #0, #1,  #2,  #3,  #4, #5, #6")

def magic(numList):
    """
    converts the combination into a number for storing in the csv file
    :param numList:
    :return:
    """
    s = map(str, numList)
    s = ''.join(s)
    s = int(s)
    return s

def poly_evaluate(inputs, targets, features, folds):
    """

    :param inputs:
    :param targets:
    :param features:
    :param folds:
    :return:
    """
    # specifiy the inputs possibilities
    features_selection = features
    #degrees = [0, 1, 2, 3, 4, 5, 6]
    degrees = [0, 1, 2, 3, 4]

    # choose a range of regularisation parameters
    num_degrees = len(degrees)
    num_folds = len(folds)

    # create some arrays to store results
    train_mean_errors = np.zeros(num_degrees)
    test_mean_errors = np.zeros(num_degrees)
    train_stdev_errors = np.zeros(num_degrees)
    test_stdev_errors = np.zeros(num_degrees)
    #
    for count, deg in enumerate(degrees):
        # count is the index of degrees, deg is the degree of the polynomial
        # cross validate with this degree

        designmtx = poly_designmtx(inputs, features_selection, deg)

        train_errors, test_errors = cv_evaluation_linear_modelN(
            designmtx, targets, folds)


        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        train_mean_errors[count] = train_mean_error
        test_mean_errors[count] = test_mean_error
        train_stdev_errors[count] = train_stdev_error
        test_stdev_errors[count] = test_stdev_error

    return train_mean_errors,test_mean_errors, train_stdev_errors, test_stdev_errors

def single_poly_evaluate(inputs,targets, features, folds, degree):
    features_selection = features

    # choose a range of regularisation parameters
    num_folds = len(folds)


    # cross validate with this degree

    designmtx = poly_designmtx(inputs, features_selection, degree)

    train_errors, test_errors = cv_evaluation_linear_modelN(
            designmtx, targets, folds)

    # we're interested in the average (mean) training and testing errors
    train_mean_error = np.mean(train_errors)
    test_mean_error = np.mean(test_errors)
    train_stdev_error = np.std(train_errors)
    test_stdev_error = np.std(test_errors)
    train_mean_errors = train_mean_error
    test_mean_errors = test_mean_error
    train_stdev_errors = train_stdev_error
    test_stdev_errors = test_stdev_error

    return train_mean_errors, test_mean_errors, train_stdev_errors, test_stdev_errors

def external_weights(train_inputs, features, train_targets, degree):
    designmtx = poly_designmtx(train_inputs, features, degree)
    weights = ml_weightsN(designmtx, train_targets)
    return weights

def poly_apply_validation_set(feature_combin, weights, degree):
    # y=feature_combin
    # col=feature_combin.tolist()
    # col.append(11)
    val_set, field_names = import_data("testing_data_split.csv",";",True,columns=[0,1,2,3,4,5,6,7,8,9,10,11])
    inputs, targets = new_conversion(val_set)
    designmtx = poly_designmtx(inputs, feature_combin, degree)
    train_predicts = linear_model_predictN(designmtx, weights)
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

def evaluate_poly_for_various_reg_params(
        inputs, targets, features_selection, folds, degree):
    """
    """
    # for rbf feature mappings
    # for the centres of the basis functions choose 10% of the data
    N = inputs.shape[0]
    designmtx= poly_designmtx(inputs, features_selection, degree)
    num_folds = len(folds)

    train_error_linear_set, test_error_linear_set = cv_evaluation_linear_modelN(designmtx, targets, folds)
    test_error_linear = np.mean(test_error_linear_set)

    # the rbf feature mapping performance
    reg_params = np.logspace(-3,2, 11)

    train_errors_list = []
    test_errors_list = []
    test_stdev_errors = []
    train_stdev_errors = []

    for reg_param in reg_params:
        print(reg_param)
        print("Evaluating reg_para " + str(reg_param))

        train_errors, test_errors = cv_evaluation_linear_modelN(
            designmtx, targets, folds, reg_param)

        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)

        train_errors_list.append(train_mean_error)
        test_errors_list.append(test_mean_error)
        train_stdev_errors.append(train_stdev_error)
        test_stdev_errors.append(test_stdev_error)

    fig , ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors_list, test_errors_list)
    # we also want to plot a straight line showing the linear performance
    xlim = ax.get_xlim()
    ax.plot(xlim, test_error_linear*np.ones(2), 'g:')
    ax.set_xscale('log')
    fig.savefig("regparamfinal", fmt="pdf")

    return train_errors_list, test_errors_list, train_stdev_errors, test_stdev_errors, reg_params

def import_data(ifname, delimiter=None, has_header=False, columns=None):
    """
    Imports a tab/comma/semi-colon/... separated data file as an array of
    floating point numbers. If the import file has a header then this should
    be specified, and the field names will be returned as the second argument.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)

    returns
    -------
    data_as_array -- the data as a numpy.array object
    field_names -- if file has header, then this is a list of strings of the
      the field names imported. Otherwise, it is a None object.
    """
    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        # instead we'll print it to screen
        if has_header:
            field_names = next(datareader)
            print("Importing data with field_names:\n\t" + ",".join(field_names))
        else:
            # if there is no header then the field names is a dummy variable
            field_names = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            #print("row = %r" % (row,))
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data list
            data.append(row)
        print("There are %d entries" % len(data))
        print("Each row has %d elements" % len(data[0]))
    # convert the data (list object) into a numpy array.
    data_as_array = np.array(data).astype(float)
    if not columns is None and not field_names is None:
        # thin the associated field names if needed
        field_names = [field_names[c] for c in columns]
    # return this array to caller (and field_names if provided)
    return data_as_array, field_names

def new_conversion(data):
    """
    This function converts the imported data (from csv) to the respective
    inputs and targets data matrices
    :param data: imported data from csv file
    :return: inputs and respective targets matrices
    """
    inputs = data[:,:11]
    targets = data[:,11]
    return inputs, targets