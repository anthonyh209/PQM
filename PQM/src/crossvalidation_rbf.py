import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools
from regression_train_test import create_cv_folds
from regression_cross_validation import evaluate_errors_scale
from regression_cross_validation import evaluate_errors_reg_param


def main(ifname=None, delimiter=None, columns=None, normalise=None):
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
        if normalise == None:
            # Ask user to confirm whether to normalise or not
            normalise_response = input("Do you want to normalise the data? (Y/N)")
            normalise = normalise_response.upper()
            normalise_label = ""

        if normalise == "Y":
            normalise_label = "_normalised"
            #Normalise input data
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
            data[:, 2] = (citric_acid- np.mean(citric_acid)) / np.std(citric_acid)
            data[:, 3] = (residual_sugar - np.mean(residual_sugar)) / np.std(residual_sugar)
            data[:, 4] = (chlorides - np.mean(chlorides)) / np.std(chlorides)
            data[:, 5] = (free_sulfur_dioxide - np.mean(free_sulfur_dioxide)) / np.std(free_sulfur_dioxide)
            data[:, 6] = (total_sulfur_dioxide - np.mean(total_sulfur_dioxide)) / np.std(total_sulfur_dioxide)
            data[:, 7] = (density- np.mean(density)) / np.std(density)
            data[:, 8] = (pH - np.mean(pH)) / np.std(pH)
            data[:, 9] = (sulphates - np.mean(sulphates)) / np.std(sulphates)
            data[:, 10] = (alcohol - np.mean(alcohol)) / np.std(alcohol)
        elif normalise != "N":
            sys.exit("Please enter valid reponse of Y or N")

        counter = 0
        N = data.shape[0]
        target = data[:, 11:]

        # get the cross-validation folds
        num_folds = 4
        folds = create_cv_folds(N, num_folds)

        feature_combin = create_combination()
        #declare number of centre to explore and create matrix for storing testing mean errors
        #num_centres_sequence = np.arange(5,80)
        #num_centre = num_centres_sequence.size
        #matrix_centre_errors = np.zeros(shape=(len(feature_combin), num_centre+1))
        #declare scales to explore and create matrix for storing testing mean errors
        scales = np.logspace(-10, 10)
        num_scales = scales.size
        matrix_scale_errors = np.zeros(shape=(len(feature_combin), num_scales+1))
        #declare reg_param to explore and create matrix for storing testing mean errors
        reg_params = np.logspace(-15, 5)
        num_reg_params = reg_params.size
        matrix_reg_param_errors = np.zeros(shape=(len(feature_combin), num_reg_params + 1))

        for i in range(len(feature_combin)):
            inputs = np.array([])
            for j in range(len(feature_combin[i])):
                inputs = np.append(inputs,data[:,feature_combin[i][j]])
            inputs = inputs.reshape(len(feature_combin[i]),data.shape[0])
            inputs = (np.rot90(inputs,3))[:, ::-1]
            counter += 1
            print("COUNTER: ",counter)
            #CROSS VALIDATION----------------------------------------------
            default_centres = np.asarray([0.35,0.4,0.45,0.459090909,0.468181818,0.477272727,0.486363636,0.495454545,0.504545455,0.513636364,0.522727273,0.531818182,0.540909091,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.7,0.75,0.8])
            default_scale = 27
            default_reg_param = 7.906043210907701e-11

            #matrix_centre_errors[i][0] = magic(feature_combin[i])
            #test_mean_errors_centres = evaluate_errors_num_centres(inputs, target, folds, default_scale, default_reg_param,num_centres_sequence)
            #matrix_centre_errors[i][1:] = test_mean_errors_centres

            matrix_scale_errors[i][0] = magic(feature_combin[i])
            test_mean_errors_scales = evaluate_errors_scale(inputs, target, folds, default_centres, default_reg_param,scales=scales)
            matrix_scale_errors[i][1:] = test_mean_errors_scales

            matrix_reg_param_errors[i][0] = magic(feature_combin[i])
            test_mean_errors_reg_param = evaluate_errors_reg_param(inputs, target, folds, default_centres, default_scale,reg_params=reg_params)
            matrix_reg_param_errors[i][1:] = test_mean_errors_reg_param


        #np.savetxt('centre_errors'+normalise_label+'.csv', matrix_centre_errors, fmt='%.6f', delimiter=',',
         #          header="#combination, #5,#6,#7,#8,#9,#10,#11,#12,#13,#14,#15,#16,#17,#18,#19,#20,#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#31,#32,#33,#34,#35,#36,#37,#38,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#49,#50,#51,#52,#53,#54,#55,#56,#57,#58,#59,#60,#61,#62,#63,#64,#65,#66,#67,#68,#69,#70,#71,#72,#73,#74,#75,#76,#77,#78,#79,#80")
        np.savetxt('scale_errors'+normalise_label+'.csv', matrix_scale_errors, fmt='%.6f', delimiter=',',
                   header="#combination, 1e-10, 2.5595479226995335e-10, 6.551285568595495e-10, 1.67683293681101e-09, 4.291934260128778e-09, 1.0985411419875573e-08, 2.8117686979742307e-08, 7.196856730011529e-08, 1.8420699693267165e-07, 4.7148663634573897e-07, 1.2067926406393288e-06, 3.088843596477485e-06, 7.906043210907702e-06, 2.0235896477251556e-05, 5.1794746792312125e-05, 0.0001325711365590111, 0.000339322177189533, 0.000868511373751352, 0.0022229964825261957, 0.005689866029018305, 0.014563484775012445, 0.03727593720314938, 0.09540954763499963, 0.2442053094548655, 0.6250551925273976, 1.5998587196060574, 4.094915062380419, 10.481131341546874, 26.826957952797272, 68.66488450042998, 175.75106248547965, 449.8432668969453, 1151.3953993264481, 2947.0517025518097, 7543.120063354608, 19306.977288832535, 49417.13361323838, 126485.52168552957, 323745.7542817653, 828642.7728546859, 2120950.8879201924, 5428675.439323859, 13894954.94373136, 35564803.06223121, 91029817.79915264, 232995181.05153814, 596362331.6594661, 1526417967.1752365, 3906939937.0546207, 10000000000.0")
        np.savetxt('reg_param_errors'+normalise_label+'.csv', matrix_reg_param_errors, fmt='%.6f', delimiter=',',
                   header="#combination, 1e-15, 2.5595479226995332e-15, 6.5512855685954955e-15, 1.67683293681101e-14, 4.291934260128778e-14, 1.0985411419875572e-13, 2.8117686979742364e-13, 7.196856730011528e-13, 1.8420699693267164e-12, 4.71486636345739e-12, 1.2067926406393265e-11, 3.088843596477485e-11, 7.906043210907701e-11, 2.0235896477251556e-10, 5.179474679231223e-10, 1.3257113655901108e-09, 3.3932217718953295e-09, 8.68511373751352e-09, 2.2229964825261957e-08, 5.689866029018305e-08, 1.4563484775012443e-07, 3.727593720314938e-07, 9.540954763499963e-07, 2.4420530945486548e-06, 6.250551925273976e-06, 1.5998587196060572e-05, 4.094915062380419e-05, 0.00010481131341546875, 0.0002682695795279727, 0.0006866488450042998, 0.0017575106248547965, 0.004498432668969453, 0.011513953993264481, 0.029470517025518096, 0.07543120063354607, 0.19306977288832536, 0.49417133613238384, 1.2648552168552958, 3.237457542817653, 8.28642772854686, 21.209508879201927, 54.286754393238596, 138.9495494373136, 355.64803062231215, 910.2981779915265, 2329.9518105153816, 5963.6233165946605, 15264.179671752365, 39069.39937054621, 100000.0")






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
            print("Importing data with field_names:\n\t" + ",".join(field_names))
        else:
            # if there is no header then the field names is a dummy variable
            field_names = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            print("row = %r" % (row,))
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
    datamtx = np.linspace(0,1, 51)
    designmtx = feature_mapping(datamtx)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for colid in range(designmtx.shape[1]):
      ax.plot(datamtx, designmtx[:,colid])
    ax.set_xlim([0,1])
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

def create_combination():
    features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    combination_array = ([])
    counter = 0

    for L in range(3, 12): #specifies the range with 3 features up to 11 features
         for subset in itertools.combinations(features, L):
             combination_array.append(subset)
             counter+=1

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



