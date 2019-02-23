import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats
import random
from regression_samples import arbitrary_function_1
from regression_models import calculate_weights_posterior
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
from regression_models import predictive_distribution
# for plotting results
from regression_plot import plot_function_and_data
from regression_plot import plot_function_data_and_approximation
from regression_plot import plot_train_test_errors
# for evaluating fit
from regression_train_test import train_and_test
from regression_train_test import create_cv_folds
from regression_train_test import cv_evaluation_linear_model
from regression_cross_validation import evaluate_num_centres
from regression_cross_validation import evaluate_errors_num_centres
from regression_cross_validation import evaluate_errors_scale
from regression_cross_validation import evaluate_errors_reg_param
from regression_cross_validation import evaluate_reg_param
from regression_cross_validation import evaluate_scale

def main(ifname=None, delimiter=None, columns=None):
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

        N = data.shape[0]
        split = int(N*0.8)
        np.random.shuffle(data)
        training,testing = data[:split,:], data[split:,:]
        print(training)

        np.savetxt('training_data_split2.csv', training, fmt='%.6f', delimiter=';',
                   header="fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality")

        np.savetxt('testing_data_split2.csv', testing, fmt='%.6f', delimiter=';',
                   header="fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality")


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