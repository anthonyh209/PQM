
import numpy as np
import matplotlib.pyplot as plt


# for creating synthetic data
from regression_samples import arbitrary_function_2
from regression_samples import sample_data
# for performing regression
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
# for plotting results
from regression_plot import plot_train_test_errors
# for evaluating fit
from regression_train_test import train_and_test
# two new functions for cross validation
from regression_train_test import create_cv_folds
from regression_train_test import cv_evaluation_linear_model

def evaluate_reg_param(inputs, targets, folds, centres, scale, reg_params=None):
    """
      Evaluate then plot the performance of different regularisation parameters
    """
    # create the feature mappoing and then the design matrix 
    feature_mapping = construct_rbf_feature_mapping(centres,scale) 
    designmtx = feature_mapping(inputs) 
    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-15,0)
    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #    
    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[r] = train_mean_error
        test_mean_errors[r] = test_mean_error
        train_stdev_errors[r] = train_stdev_error
        test_stdev_errors[r] = test_stdev_error

    #Get test error without reg param
    blank ,test_errors_without_reg = cv_evaluation_linear_model(designmtx,targets,folds,reg_param=None)
    test_mean_error_without_reg_param = np.mean(test_errors_without_reg)

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='r')
    #plot green line to represent no reg params
    xlim = ax.get_xlim()
    ax.plot(xlim, test_mean_error_without_reg_param * np.ones(2), 'g:')
    ax.set_xscale('log')


def evaluate_scale(inputs, targets, folds, centres, reg_param, scales=None):
    """
    evaluate then plot the performance of different basis function scales
    """
    # choose a range of scales
    if scales is None:
        scales = np.logspace(-10,10)
    #
    num_values = scales.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #    
    for s, scale in enumerate(scales):
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        designmtx = feature_mapping(inputs) 
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[s] = train_mean_error
        test_mean_errors[s] = test_mean_error
        train_stdev_errors[s] = train_stdev_error
        test_stdev_errors[s] = test_stdev_error

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "scale", scales, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(scales, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(scales, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')





def evaluate_num_centres(
        inputs, targets, folds, scale, reg_param, num_centres_sequence=None):
    """
      Evaluate then plot the performance of different numbers of basis
      function centres.
    """
    # fix the reg_param
    reg_param = 0.01
    # fix the scale
    scale = 100
    # choose a range of numbers of centres
    if num_centres_sequence is None:
        num_centres_sequence = np.arange(5,200)
    num_values = num_centres_sequence.size
    print(num_values)
    num_folds = len(folds)
    #
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #    
    # run the experiments
    for c, num_centres in enumerate(num_centres_sequence):
        centres = np.linspace(0,1,num_centres)
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        designmtx = feature_mapping(inputs)
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[c] = train_mean_error
        test_mean_errors[c] = test_mean_error
        train_stdev_errors[c] = train_stdev_error
        test_stdev_errors[c] = test_stdev_error
    #
    # Now plot the results
    fig, ax = plot_train_test_errors(
        "Num. Centres", num_centres_sequence, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(num_centres_sequence, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(num_centres_sequence, lower, upper, alpha=0.2, color='r')

    return train_mean_errors

def main():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """

    # choose number of data-points and sample a pair of vectors: the input
    # values and the corresponding target values
    N = 500
    inputs, targets = sample_data(N, arbitrary_function_2, seed=1)

    # specify the centres and scale of some rbf basis functions
    default_centres = np.linspace(0,1,21)
    default_scale = 0.03
    default_reg_param = 0.08

    # get the cross-validation folds
    num_folds = 4
    folds = create_cv_folds(N, num_folds)

    # evaluate then plot the performance of different reg params
    evaluate_reg_param(inputs, targets, folds, default_centres, default_scale)
    # evaluate then plot the performance of different scales
    evaluate_scale(inputs, targets, folds, default_centres, default_reg_param)
    # evaluate then plot the performance of different numbers of basis
    # function centres.
    evaluate_num_centres(
        inputs, targets, folds, default_scale, default_reg_param)

    plt.show()






def evaluate_errors_num_centres(
        inputs, targets, folds, scale, reg_param, num_centres_sequence=None):
    """
      Evaluate then plot the performance of different numbers of basis
      function centres.
    """
    # fix the reg_param
    reg_param = 0.01
    # fix the scale
    scale = 100
    # choose a range of numbers of centres
    if num_centres_sequence is None:
        num_centres_sequence = np.arange(200, 250)
    num_values = num_centres_sequence.size
    num_folds = len(folds)
    #
    # create array to store results
    test_mean_errors = np.zeros(num_values)

    #
    # run the experiments
    for c, num_centres in enumerate(num_centres_sequence):
        centres = np.linspace(0, 1, num_centres)
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        designmtx = feature_mapping(inputs)
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        test_mean_error = np.mean(test_errors)
        # store the results
        test_mean_errors[c] = test_mean_error

    return test_mean_errors

def evaluate_errors_scale(inputs, targets, folds, centres, reg_param, scales=None):
    """
    evaluate then plot the performance of different basis function scales
    """
    # choose a range of scales
    if scales is None:
        scales = np.logspace(-10,10)
    #
    num_values = scales.size
    num_folds = len(folds)
    # create some arrays to store results
    test_mean_errors = np.zeros(num_values)
    #
    for s, scale in enumerate(scales):
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        designmtx = feature_mapping(inputs)
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        test_mean_error = np.mean(test_errors)
        # store the results
        test_mean_errors[s] = test_mean_error

    return test_mean_errors

def evaluate_errors_reg_param(inputs, targets, folds, centres, scale, reg_params=None):
    """
      Evaluate then plot the performance of different regularisation parameters
    """
    # create the feature mappoing and then the design matrix
    feature_mapping = construct_rbf_feature_mapping(centres,scale)
    designmtx = feature_mapping(inputs)
    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-11,0)
    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    test_mean_errors = np.zeros(num_values)
    #
    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        test_mean_error = np.mean(test_errors)
        # store the results
        test_mean_errors[r] = test_mean_error

    return test_mean_errors

if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()