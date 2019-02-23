import csv
import pandas as pd
import numpy as np
import itertools #part of python
import matplotlib.pyplot as plt

from regression_multipoly import magic
from regression_modelsN import linear_model_predictN

# for plotting results
from regression_plotN import plot_train_test_errors
from data_analysis import sigma_data
from data_analysis import cov_plot

# for creating the folds and evaluation
from regression_train_testN import create_cv_foldsN
from regression_train_testN import cv_evaluation_linear_modelN

# for calculating the different errors
from regression_multipoly import error_features_poly
from regression_multipoly import poly_evaluate
from regression_multipoly import poly_designmtx
from regression_multipoly import single_poly_evaluate
from regression_train_testN import ml_weightsN

# for polynomial
from regression_multipoly import evaluate_poly_for_various_reg_params
from regression_multipoly import external_weights
from regression_multipoly import poly_apply_validation_set

# for RBF
from crossvalidation_rbf import main as crossvalidation_rbf
from identify_parameters_rbf import main as identify_parameters_rbf
from evaluate_rbf import main as evaluate_rbf
from calculating_optimal_rbf_coursework import main as assess_optimal_rbf
from calculating_optimal_rbf_11 import main as assess_optimal_rbf_11
from split_train_and_test import main as split_train_and_test

# for kNN
from regression_train_testN import cv_evaluation_kNN_model
from regression_plotN import plot_kNN_errors
from regression_modelsN import kNN_construct
from kNN_model import input_construction_kNN
from kNN_model import evaluate_for_various_k
from kNN_model import kNN_apply_validation_set


#Final code2
def main(ifname=None, delimiter=None, columns=[0,1,2,3,4,5,6,7,8,9,10,11]):
    """
    This function contains the code to process the data and output the plots which
    shows use of the basis functions with a polynomial use
    """
    analyse_data()

    # for importing the data
    data, field_names = import_data(
        ifname, delimiter=';', has_header=True, columns=columns)
    inputs, targets = new_conversion(data)
    N = targets.shape[0]

    # for creating the folds
    num_folds = 4
    folds = create_cv_foldsN(N, num_folds)
    features_selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    sigma_matrix = sigma_data(data, field_names)
    #cov_plot(sigma_matrix)

##Split data as training and testing samples
    # Changed to produce a split with different csv file name in order to preserve the original train and validation test samples
    split_train_and_test("winequality-red.csv", ";")

##polynomial section

    "for creating all features combinations (optional to use in later section of code)"
    features_combination = create_combination()


    "for evaluating all features combination from degrees 0 to 6 (computationally very expensive)" \
    "uncomment following code line to exectute this computation (be aware that this might take hours)"""
    # error_features_poly(inputs, targets, folds, features_combination)



    "Linear model: Best features included"
    degree = 1 #for linear
    feature_sel_lin = 1971
    train_errors_s, test_errors_s, train_stdev_errors_s, test_stdev_errors_s = single_poly_evaluate(inputs, targets, features_combination[feature_sel_lin], folds, degree)
    print("linear model: best feature selection")
    print("test errors: ", test_errors_s)
    print("train errors: ",train_errors_s)
    print("test-std: ", test_stdev_errors_s)
    print("train-std: ", train_stdev_errors_s)

    "Linear model: All features included"
    degree = 1  # for linear
    feature_sel_lin = 1980
    train_errors_s, test_errors_s, train_stdev_errors_s, test_stdev_errors_s = single_poly_evaluate(inputs, targets,features_combination[feature_sel_lin],folds, degree)
    print("linear model: All feature selection")
    print("test errors: ", test_errors_s)
    print("train errors: ", train_errors_s)
    print("test-std: ", test_stdev_errors_s)
    print("train-std: ", train_stdev_errors_s)

    "Polynomial: best features for degrees 0 to 4"
    degrees = [0,1,2,3,4]
    feature_sel_poly = 1703 #1980 to select all features
    print(features_combination[feature_sel_poly])
    train_errors, test_errors, train_stdev_errors, test_stdev_errors = poly_evaluate(inputs, targets, features_combination[feature_sel_poly], folds)
    fig,ax=plot_train_test_errors("degree", [0,1,2,3,4], train_errors[:5], test_errors[:5])
    ax.set_ylim([0, 1])
    plt.xticks([0, 1, 2, 3, 4])
    #fig.savefig("BestPoly.pdf", fmt="pdf")
    print("test_errors: ", test_errors)
    print("train_errors: ",train_errors)
    print("test_std: ",test_stdev_errors)
    print ("train_std: ",train_stdev_errors)


    "Polynomial: all features for degrees 0 to 4"
    degrees = [0,1,2,3,4]
    feature_sel_poly = 1980 #1980 to select all features
    print(features_combination[feature_sel_poly])
    train_errors, test_errors, train_stdev_errors, test_stdev_errors = poly_evaluate(inputs, targets, features_combination[feature_sel_poly], folds)
    fig,ax=plot_train_test_errors("degree", [0,1,2,3,4], train_errors[:5], test_errors[:5])

    ax.set_ylim([0, 1])
    plt.xticks([0, 1, 2, 3, 4])
    #plot_train_test_errors("degree", [0, 1, 2, 3], train_errors[:4], test_errors[:4])
    #fig.savefig("BestPoly.pdf", fmt="pdf")
    print("test_errors: ", test_errors)
    print("train_errors: ",train_errors)
    print("test_std: ",test_stdev_errors)
    print ("train_std: ",train_stdev_errors)


    "If want to get error values for specific configuration including reg.param"
    "section which plots the error values over a range of reg_params values (sd 8)"
    train_errors, test_errors, train_stdev_errors, test_stdev_errors, regparams = evaluate_poly_for_various_reg_params(inputs, targets, features_combination[1703], folds, 3) # 3 is the degree
    print("regularisation degree 3 for best features")
    index = np.argmin(test_errors)
    print(train_errors[index])
    print(test_errors[index])
    print(train_stdev_errors[index])
    print(test_stdev_errors[index])
    print("lambda")
    print(regparams[index])


##polynomial external validation (on final data set)
    "validation for linear model"
    x = 1971
    features_selection = features_combination[x]
    degree = 1
    weights = external_weights(inputs, features_selection, targets, degree)
    poly_apply_validation_set(features_selection, weights, degree)
    print("validation with linear")

    "validation for polynomial"
    x = 1703
    features_selection = features_combination[x]
    degree = 2
    weights = external_weights(inputs, features_selection, targets, degree)
    poly_apply_validation_set(features_selection, weights, degree)
    print("validation with polynomial")

##Radial Basis Function section
    "Be aware below cross validaitons are computationally expensive because it runs every possible feature combination. Each run may take a couple hours"
    # # crossvalidation_rbf("training_data_split.csv",";", normalise="N")
    # # crossvalidation_rbf("training_data_split.csv", ";", normalise="Y")
    # # Loops through csv to find optimum parameters and feature combinations with least error
    identify_parameters_rbf("reg_param_errors.csv", ",")
    identify_parameters_rbf("reg_param_errors_normalised.csv", ",")
    identify_parameters_rbf("scale_errors.csv", ",")
    identify_parameters_rbf("scale_errors_normalised.csv", ",")
    #
    # # Finds mean test error and std. Also attempts to predict the validation testing set
    print("NOT NORMALISED OPTIMAL RBF : ")
    # assess_optimal_rbf("training_data_split.csv", ";", normalise="N", features=[2, 7, 9, 10])
    print("NORMALISED OPTIMAL RBF : ")
    # assess_optimal_rbf("training_data_split.csv", ";", normalise="Y", features=[2, 7, 9, 10])
    print("OPTIMAL RBF FOR 11 FEATURES: ")
    # assess_optimal_rbf_11("training_data_split.csv", ";", normalise="N", features=[0,1,2,3,4,5,6, 7,8, 9, 10])
    #
    # # Plots against varying reg param and scale
    evaluate_rbf("training_data_split.csv", ";", columns=[2, 7, 9, 10, 11])

##kNN section
    "for evaluating all features combination for k values from 1 to 99 (computationally very expensive)"
    "uncomment following code line to exectute this computation (be aware that this might take hours)"""
    # error_features_kNN(inputs, targets, folds, features_combination)

    "section to evaluate and plot the error values for kNN regression ranging from 1 to 100 points"
    selected_features_kNN = features_combination[830] # for selecting specific numbers of features
    kNN_inputs = input_construction_kNN(inputs, selected_features_kNN) #create the correct input form of the kNN
    x,y = evaluate_for_various_k(kNN_inputs, targets, folds)
    index= np.argmin(x)
    print(x[index])
    print(y[index])

##kNN external validation (on final data set)

    "final evaluation data set kNN: for best found parameters, this method will predict new quality values for given " \
    "wine parameters "
    selected_features_kNN = features_combination[830]
    k = 19
    kNN_data_pool = input_construction_kNN(inputs, selected_features_kNN) #create the correct input form of the kNN
    kNN_apply_validation_set(kNN_data_pool, targets, selected_features_kNN, k)
    print("Validation with kNN")
    plt.show()


# Method creates summary statistics and histogram
def analyse_data():
    # Load the Red Wines dataset
    data_set = pd.read_csv("winequality-red.csv", sep=';')
    data_array = np.array(data_set)
    quality_data = data_array[:, 11]

    correlation = data_set.corr()
    plt.figure(figsize=(14, 12))

    # Histogram of distribution of wine quality
    plt.hist(quality_data, bins=12)
    plt.title("Histogram of wine quality")
    #plt.show


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

def create_combination():
    """
    this function creates all possible combinations with 3 features up to all 11 features
    :return:
    """
    features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    combination_array = ([])
    counter = 0

    for L in range(3, 12): #specifies the range with 3 features up to 11 features
         for subset in itertools.combinations(features, L):
             combination_array.append(subset)
             counter+=1

    features_combation = []
    for i in range(0, 1981):  # second set should be from (1920, 1980)
        # (1901, 1981): from 1901 to 1980
        features_combation.append(np.asarray(combination_array[i]))

    return features_combation

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
        main() # calls the main function with no arguments
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