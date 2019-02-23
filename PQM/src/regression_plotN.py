import numpy as np
import matplotlib.pyplot as plt

def plot_train_test_errors(
        control_var, experiment_sequence, train_errors, test_errors):
    """
    Plot the train and test errors for a sequence of experiments.

    parameters
    ----------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    train_line, = ax.plot(experiment_sequence, train_errors,'b-')
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-')
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")
    ax.legend([train_line, test_line], ["train", "test"])

    return fig, ax

def plot_kNN_errors(
        control_var, experiment_sequence, train_errors):
    """
    Plot the train and test errors for a sequence of experiments.

    parameters
    ----------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    train_line, = ax.plot(experiment_sequence, train_errors,'b-')
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")
    ax.legend([train_line], ["kNN"])
    return fig, ax

def plot_functionN(true_func, linewidth=3, xlim=None):
    """
    Plot a function in a given range

    parameters
    ----------
    true_func - the function to plot
    xlim (optional) - the range of values to plot for. A pair of values lower,
        upper. If not specified, the default will be (0,1)
    linewidth (optional) - the width of the plotted line

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the (one) line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xs = np.linspace(xlim[0], xlim[1], 101)
    true_ys = true_func(xs)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.5, 1.5)
    return fig, ax

def exploratory_plotsN(data, field_names=None):
    # the number of dimensions in the data
    dim = data.shape[1]
    # create an empty figure object
    fig = plt.figure()
    # create a grid of four axes
    plot_id = 1
    for i in range(dim):
        for j in range(dim):
            ax = fig.add_subplot(dim,dim,plot_id)
            # if it is a plot on the diagonal we histogram the data
            if i == j:
                ax.hist(data[:,i]) #all
            # otherwise we scatter plot the data
            else:
                ax.plot(data[:,i],data[:,j], 'o', markersize=1)
            # we're only interested in the patterns in the data, so there is no
            # need for numeric values at this stage
            #ax.set_xticks([])
            #ax.set_yticks([])
            # if we have field names, then label the axes
            if not field_names is None:
                if i == (dim-1):
                    ax.set_xlabel(field_names[j])
                if j == 0:
                    ax.set_ylabel(field_names[i])
            # increment the plot_id
            plot_id += 1
    plt.tight_layout()
  
