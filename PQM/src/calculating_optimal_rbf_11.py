import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
from regression_models import ml_weights
from regression_models import regularised_ml_weights
# for evaluating fit
from regression_train_test import create_cv_folds
from regression_train_test import cv_evaluation_linear_model


def main(ifname=None, delimiter=None, columns=None, normalise=None, features=None):
    """
    To be called when the script is run. This function fits and plots imported data (if a filename is
    provided). Data is 2 dimensional real valued data and is fit
    with maximum likelihood 2d gaussian.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)
    """
    # if no file name is provided then use synthetic data
    if ifname is None:
        print("You need to ingest the CSV file")
    else:
        data, field_names = import_data(ifname, delimiter=delimiter, has_header=True, columns=columns)

        # DATA PREPARATION-----------------------------------------------
        N = data.shape[0]
        target = data[:, 11:]

        # Ask user to confirm whether to normalise or not
        if normalise == None:
            normalise_response = input("Do you want to normalise the data? (Y/N)")
            normalise = normalise_response.upper()
            normalise_label = ""

        if normalise == "Y":
            normalise_label = "_normalised"
            # Normalise input data
            fixed_acidity = data[:, 0]
            volatility_acidity = data[:, 1]
            citric_acid = data[:, 2]
            residual_sugar = data[:, 3]
            chlorides = data[:, 4]
            free_sulfur_dioxide = data[:, 5]
            total_sulfur_dioxide = data[:, 6]
            density = data[:, 7]
            pH = data[:, 8]
            sulphates = data[:, 9]
            alcohol = data[:, 10]

            data[:, 0] = (fixed_acidity - np.mean(fixed_acidity)) / np.std(fixed_acidity)
            data[:, 1] = (volatility_acidity - np.mean(volatility_acidity)) / np.std(volatility_acidity)
            data[:, 2] = (citric_acid - np.mean(citric_acid)) / np.std(citric_acid)
            data[:, 3] = (residual_sugar - np.mean(residual_sugar)) / np.std(residual_sugar)
            data[:, 4] = (chlorides - np.mean(chlorides)) / np.std(chlorides)
            data[:, 5] = (free_sulfur_dioxide - np.mean(free_sulfur_dioxide)) / np.std(free_sulfur_dioxide)
            data[:, 6] = (total_sulfur_dioxide - np.mean(total_sulfur_dioxide)) / np.std(total_sulfur_dioxide)
            data[:, 7] = (density - np.mean(density)) / np.std(density)
            data[:, 8] = (pH - np.mean(pH)) / np.std(pH)
            data[:, 9] = (sulphates - np.mean(sulphates)) / np.std(sulphates)
            data[:, 10] = (alcohol - np.mean(alcohol)) / np.std(alcohol)
        elif normalise != "N":
            sys.exit("Please enter valid reponse of Y or N")

        if features == None:
            feature_response = input("Please specify which feature combination you want (e.g.1,2,5,7)")
            feature_response = feature_response.split(",")
            # need to convert list of strings into list of integer
            feature_combin = []
            for i in range(len(feature_response)):
                print(feature_response[i])
                feature_combin.append(int(feature_response[i]))
        else:
            feature_combin = features

        inputs = np.array([])
        for j in range(len(feature_combin)):
            inputs = np.append(inputs, data[:, feature_combin[j]])
        inputs = inputs.reshape(len(feature_combin), data.shape[0])
        inputs = (np.rot90(inputs, 3))[:, ::-1]
        #print("INPUT: ", inputs)

        # Plotting RBF Model ----------------------------------------------------------
        # specify the centres of the rbf basis functions
        centres = np.asarray([0.35,0.4,0.45,0.459090909,0.468181818,0.477272727,0.486363636,0.495454545,0.504545455,0.513636364,0.522727273,0.531818182,0.540909091,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.7,0.75,0.8])
        # the width (analogous to standard deviation) of the basis functions
        scale = 450
        reg_param = 7.906043210907701e-11

        print("centres = %r" % (centres,))
        print("scale = %r" % (scale,))
        print("reg param = %r" % (reg_param,))

        # create the feature mapping
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        # plot the basis functions themselves for reference
        #display_basis_functions(feature_mapping)
        # now construct the design matrix for the inputs
        designmtx = feature_mapping(inputs)
        # the number of features is the widht of this matrix
        print("DESIGN MATRIX: ", designmtx)

        if reg_param is None:
            # use simple least squares approach
            weights = ml_weights(
                designmtx, target)
        else:
            # use regularised least squares approach
            weights = regularised_ml_weights(
                designmtx, target, reg_param)

        # get the cross-validation folds
        num_folds = 4
        folds = create_cv_folds(N, num_folds)

        train_errors, test_errors = cv_evaluation_linear_model(designmtx, target, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        print("TRAIN MEAN ERROR: ", train_mean_error)
        print("TEST MEAN ERROR: ", test_mean_error)
        print("TRAIN STDEV ERROR: ", train_stdev_error)
        print("TEST STDEV ERROR: ", test_stdev_error)
        print("ML WEIGHTS: ", weights)
        apply_validation_set(feature_combin,feature_mapping,weights)



def apply_validation_set(feature_combin,feature_mapping,weights):
    feature_combin.append(11)
    val_set, field_names = import_data("testing_data_split.csv",";",True,columns=feature_combin)
    inputs = val_set[:, 0:val_set.shape[1] - 1]
    targets = val_set[:, val_set.shape[1] - 1:]
    pred_func = construct_feature_mapping_approx(feature_mapping, weights)
    pred_ys = pred_func(inputs)
    pred_targets = []
    for i in range(pred_ys.size):
        pred_targets.append(round(pred_ys[i],0))
    targets = np.rot90(targets,3)
    counter = 0
    for i in range(len(pred_targets)):
        if pred_targets[i] == targets[:,i]:
            counter +=1
    print("CORRECT: ",counter)
    print("WRONG: ", len(pred_targets)-counter)


def magic(numList):
    s = map(str, numList)
    s = ''.join(s)
    s = int(s)
    return s


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
            #print("Importing data with field_names:\n\t" + ",".join(field_names))
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


def display_basis_functions(feature_mapping):
    datamtx = np.linspace(0, 1, 51)
    designmtx = feature_mapping(datamtx)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for colid in range(designmtx.shape[1]):
        ax.plot(datamtx, designmtx[:, colid])
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])


def create_combination():
    features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    combination_array = ([])
    counter = 0

    for L in range(3, 12):  # specifies the range with 3 features up to 11 features
        for subset in itertools.combinations(features, L):
            combination_array.append(subset)
            counter += 1

    return combination_array


if __name__ == '__main__':
    """
    To run this script on just synthetic data use:

        python multivariate_data.py

    You can pass the data-file name as the first argument when you call
    your script from the command line. E.g. use:

        python multivariate_data.py datafile.tsv

    If you pass a second argument it will be taken as the delimiter, e.g.
    for comma separated values:

        python multivariate_data.py comma_separated_data.csv ","

    for semi-colon separated values:

        python multivariate_data.py comma_separated_data.csv ";"

    If your data has more than 2 columns you must specify which columns
    you wish to plot as a comma separated pair of values, e.g.

        python multivariate_data.py comma_separated_data.csv ";" 8,9

    For the wine quality data you will need to specify which columns to pass.
    """
    import sys

    if len(sys.argv) == 1:
        main()  # calls the main function with no arguments
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(ifname=sys.argv[1])
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        main(ifname=sys.argv[1], delimiter=sys.argv[2])
    elif len(sys.argv) == 4:
        # assumes that the third argument is the list of columns to import
        columns = list(map(int, sys.argv[3].split(",")))
        main(ifname=sys.argv[1], delimiter=sys.argv[2], columns=columns)



