import csv
import numpy as np
import matplotlib.pyplot as plt
from regression_models import construct_rbf_feature_mapping


# for evaluating fit
from regression_train_test import create_cv_folds
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

       #DATA PREPARATION-----------------------------------------------
        counter = 0
        N = data.shape[0]
        input = data[:, 0:data.shape[1]-1]
        target = data[:, data.shape[1]-1:]
        #print("FEATURES : ",columns)
        #print("INPUT :", input)

        #declare number of centre to explore and create matrix for storing testing mean errors
        num_centres_sequence = np.arange(5,100)
        scales = np.logspace(-10, 10)
        reg_params = np.logspace(-15, 10)

        # specify the centres and scale of some rbf basis functions
        default_centres = np.asarray([0.35,0.4,0.45,0.459090909,0.468181818,0.477272727,0.486363636,0.495454545,0.504545455,0.513636364,0.522727273,0.531818182,0.540909091,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.7,0.75,0.8])
        default_scale = 26.8
        default_reg_param = 7.906043210907701e-11

        # get the cross-validation folds
        num_folds = 4
        folds = create_cv_folds(N, num_folds)

        # evaluate then plot the performance of different reg params
        evaluate_reg_param(input, target, folds, default_centres, default_scale,reg_params)
        # evaluate then plot the performance of different scales
        evaluate_scale(input, target, folds, default_centres, default_reg_param)
        # evaluate then plot the performance of different numbers of basis
        # function centres.
        #test_mean_errors_for_centre = evaluate_num_centres(input, target, folds, default_scale, default_reg_param,num_centres_sequence)
        #steep_centre,optimum_centre = point_of_steepest_gradient(test_mean_errors_for_centre,num_centres_sequence)
        #print("Centre with steepest drop of test mean errors: ",steep_centre)
        #print("Optimum number of centres which within tolerance: ",optimum_centre)


        # the width (analogous to standard deviation) of the basis functions
        scale = 0.1
        feature_mapping = construct_rbf_feature_mapping(default_centres, scale)
        datamtx = np.linspace(0, 1, 51)
        designmtx = feature_mapping(datamtx)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for colid in range(designmtx.shape[1]):
            ax.plot(datamtx, designmtx[:,colid])
            ax.set_xlim([0,1])
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])


        #plt.show()

def point_of_steepest_gradient(test_mean_error,num_centres_sequence):
    gradient = np.gradient(test_mean_error)
    change_of_gradient = np.gradient(gradient)
    steep_index = np.argmax(abs(change_of_gradient))

    #assume tolerance of change to gradient less than 2.5e-06
    for i in range(change_of_gradient.size):
        if abs(change_of_gradient[i]) > 2.5e-06:
            intolerate_index = i
        else:
            intolerate_index = steep_index

    return num_centres_sequence[steep_index],num_centres_sequence[intolerate_index]




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



