import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

def cov_plot(sigma_matrix):
    sns.set(style="white")
    x = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    # Generate a large random dataset
    #rs = np.random.RandomState(33)
    d = pd.DataFrame(data=sigma_matrix,
                     columns=x)


    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    counter = 0
    for i in range(12):
        mask[i][counter] = True
        counter += 1

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 12))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    sns.heatmap(corr, mask=mask, annot=False, cmap="Blues", vmax=.3, center=0,
                square=True, linewidths=.1, cbar_kws={"shrink": .5})
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.75, 0, 0.25])
    colorbar.set_ticklabels(['Tendency negative correlation', 'Tendency no correlation', 'Tendency positive correlation'])

    #f.savefig("correlation matrix", fmt="pdf")

    plt.show()

def sigma_data(data, field_names=None):
    # the number of dimensions in the data
    dim = data.shape[1]
    # create an empty figure object

    # create a grid of four axes
    plot_id = 1
    x = [0,1,2]
    y = [0,1,2]
    N, dim = data.shape;
    counter = 0
    internal_data = np.zeros((N,2))
    sigma_matrix = np.zeros(shape=(12,12))
    print(sigma_matrix)
    for i in range(dim):
        for j in range(dim):
            # if it is a plot on the diagonal we histogram the data
            if i == j:
                counter += 1
                sigma_matrix[i][j] = 0
            # otherwise we scatter plot the data
            else:
                internal_data[:,0] = data[:,i]
                internal_data[:,1] = data[:,j]
                mu, Sigma = max_lik_mv_gaussian_approx(internal_data)
                #print(mu)
                sigma_matrix[i][j] = Sigma[0][1]
                print("Sigma: "+field_names[i]+" vs "+field_names[j])
                print(Sigma)
                counter += 1
            # we're only interested in the patterns in the data, so there is no
            # need for numeric values at this stage
            #ax.set_xticks([])
            #ax.set_yticks([])
            # if we have field names, then label the axes
            # increment the plot_id
            #fig.savefig(field_names[i], fmt="pdf")
    print(sigma_matrix)

    return sigma_matrix

def exploratory_plots(data, field_names=None):
    # the number of dimensions in the data
    dim = data.shape[1]
    # create an empty figure object

    # create a grid of four axes
    plot_id = 1
    x = [0,1,2]
    y = [0,1,2]
    N, dim = data.shape;
    counter = 0
    internal_data = np.zeros((N,2))
    sigma_matrix = np.zeros(shape=(12,12))
    print(np.zeros)
    for i in range(dim):
        fig = plt.figure(figsize=(20,10))
        for j in range(dim):
            ax = fig.add_subplot(3,4,plot_id)
            # if it is a plot on the diagonal we histogram the data
            if i == j:
                ax.hist(data[:,i]) #all
                counter += 1
            # otherwise we scatter plot the data
            else:
                ax.plot(data[:,i],data[:,j], 'o', markersize=1)
                internal_data[:,0] = data[:,i]
                internal_data[:,1] = data[:,j]
                mu, Sigma = max_lik_mv_gaussian_approx(internal_data)
                #print(mu)
                print("Sigma: "+field_names[i]+" vs "+field_names[j])
                print(Sigma)
                counter += 1
                overlay_2d_gaussian_contour(ax, mu, Sigma)
            # we're only interested in the patterns in the data, so there is no
            # need for numeric values at this stage
            #ax.set_xticks([])
            #ax.set_yticks([])
            # if we have field names, then label the axes
            if not field_names is None:
                ax.set_xlabel(field_names[i])
                ax.set_ylabel(field_names[j])
            # increment the plot_id
            plot_id += 1
            #fig.savefig(field_names[i], fmt="pdf")
        plot_id = 1
    plt.tight_layout()

def overlay_2d_gaussian_contour(ax, mu, Sigma, num_grid_points=60):
    """
    Overlays the contours of a 2d-gaussian with mean, mu, and covariance matrix
    Sigma onto an existing set of axes.

    parameters
    ----------
    ax -- a matplotlib.axes.Axes object on which to plot the contours
    mu -- a 2-vector mean of the distribution
    Sigma -- the (2x2)-covariance matrix of the distribution.
    num_grid_points (optional) -- the number of grid_points along each dimension
      at which to evaluate the pdf
    """
    # generate num_grid_points grid-points in each dimension
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xpoints = np.linspace(xmin, xmax, num_grid_points)
    ypoints = np.linspace(ymin, ymax, num_grid_points)
    # meshgrid produces two 2d arrays of the x and y coordinates
    xgrid, ygrid = np.meshgrid(xpoints, ypoints)
    # Pack xgrid and ygrid into a single 3-dimensional array
    pos = np.empty(xgrid.shape + (2,)) #changed this one
    N, dim, x = pos.shape
    pos[:, :, 0] = xgrid
    pos[:, :, 1] = ygrid
    # create a distribution over the random variable
    rv = stats.multivariate_normal(mu, Sigma)
    # evaluate the rv probability density at every point on the grid
    prob_density = rv.pdf(pos)
    ax.contour(xgrid, ygrid, prob_density)

def max_lik_mv_gaussian_approx(X):
    """
    Finds the maximum likelihood mean and variance for gaussian data samples (X)

    parameters
    ----------
    X - data array, 2d array of samples, each row is assumed to be an
      independent sample from a multi-variate gaussian

    returns
    -------
    mu - mean vector
    Sigma - 2d array corresponding to the covariance matrix
    """
    # the mean sample is the mean of the rows of X
    N, dim = X.shape
    mu = np.mean(X,0)
    Sigma = np.zeros((dim,dim))
    # the covariance matrix requires us to sum the dyadic product of
    # each sample minus the mean.
    for x in X:
        # subtract mean from data point, and reshape to column vector
        # note that numpy.matrix is being used so that the * operator
        # in the next line performs the outer-product v * v.T
        x_minus_mu = np.matrix(x - mu).reshape((dim,1))
        # the outer-product v * v.T of a k-dimentional vector v gives
        # a (k x k)-matrix as output. This is added to the running total.
        Sigma += x_minus_mu * x_minus_mu.T
    # Sigma is unnormalised, so we divide by the number of datapoints
    Sigma /= N
    # we convert Sigma matrix back to an array to avoid confusion later
    return mu, np.asarray(Sigma)



