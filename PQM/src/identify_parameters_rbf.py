import csv
import numpy as np
import itertools


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

        if ifname == "centre_errors.csv":
            parameter = "Centres with least test errors: "
        if ifname == "reg_param_errors.csv":
            parameter = "Reg param with least test errors: "
        if ifname == "reg_param_errors_normalised.csv":
            parameter = "Normalised - Reg param with least test errors: "
        if ifname == "scale_errors.csv":
            parameter = "Scale with least test errors: "
        if ifname == "scale_errors_normalised.csv":
            parameter = "Normalised - Scale with least test errors: "

        feature_combin = create_combination()
        N = data.shape[0]
        errors = data[:, 1:]
        features = data[:,:1]
        table_least_errors = np.zeros(shape=(len(features), 3))

        for i in range(len(features)):
            least_error_index = np.argmin(errors[i])
            least_error_value = np.min(errors[i])
            table_least_errors[i][0] = magic(feature_combin[i])
            table_least_errors[i][1] = (field_names[least_error_index+1])
            table_least_errors[i][2] = least_error_value


        absolute_least_error_index = np.argmin(table_least_errors[:,2])
        print("Feature combination with least errors: ",features[absolute_least_error_index])
        print("INDEX: ", absolute_least_error_index)
        print(parameter, table_least_errors[absolute_least_error_index][1], " with test errors of " ,np.min(table_least_errors[:,2]))



def create_combination():
    features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    combination_array = ([])
    counter = 0

    for L in range(3, 12): #specifies the range with 3 features up to 11 features
         for subset in itertools.combinations(features, L):
             combination_array.append(subset)
             counter+=1

    return combination_array

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