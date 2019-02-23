import matplotlib.pyplot as plt

from crossvalidation_rbf import main as crossvalidation_rbf
from identify_parameters_rbf import main as identify_parameters_rbf
from evaluate_rbf import main as evaluate_rbf
from calculating_optimal_rbf_coursework import main as assess_optimal_rbf
from calculating_optimal_rbf_11 import main as assess_optimal_rbf_11
from split_train_and_test import main as split_train_and_test

def main():
    # Changed to produce a split with different csv file name in order to preserve the original train and validation test samples
    split_train_and_test("winequality-red.csv", ";")
    # Be aware below cross validaitons are computationally expensive because it runs every possible feature combination. Each run may take a couple hours
    #crossvalidation_rbf("training_data_split.csv",";", normalise="N")
    #crossvalidation_rbf("training_data_split.csv", ";", normalise="Y")
    # Loops through csv to find optimum parameters and feature combinations with least error
    identify_parameters_rbf("reg_param_errors.csv", ",")
    identify_parameters_rbf("reg_param_errors_normalised.csv", ",")
    identify_parameters_rbf("scale_errors.csv", ",")
    identify_parameters_rbf("scale_errors_normalised.csv", ",")

    # Finds mean test error and std. Also attempts to predict the validation testing set
    assess_optimal_rbf("training_data_split.csv", ";",normalise = "N",features=[2,7,9,10])
    assess_optimal_rbf("training_data_split.csv", ";",normalise = "Y",features=[2,7,9,10])
    assess_optimal_rbf_11("training_data_split.csv", ";", normalise="N", features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Plots against varying reg param and scale
    evaluate_rbf("training_data_split.csv", ";", columns=[2, 7, 9, 10, 11])







if __name__ == '__main__':
        # this bit only runs when this script is called from the command line
        main()
