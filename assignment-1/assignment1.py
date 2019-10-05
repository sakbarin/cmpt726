"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0, include_bias=True, mu=[], s=1):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    phi = design_matrix(x, basis, poly_degree=degree, miu=mu, s=s, include_bias=include_bias)

    # Learning Coefficients
    if reg_lambda > 0:
        w = np.linalg.pinv(reg_lambda*np.identity(67) + phi.transpose()*phi) * phi.transpose() * t
    else:
        # no regularization
        w = np.linalg.pinv(phi) * t

    y_train = phi * w

    # Measure root mean squared error on training data.
    train_err_E = np.square(y_train - t)
    train_err_RMS =  np.sqrt(np.sum(train_err_E) / len(x))

    return (w, y_train, train_err_RMS)


def design_matrix(x, basis, poly_degree=1, miu=[], s=0, include_bias=True):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        x is training input
        basis is basis function name
        degree is polynomial degree

    Returns:
      phi design matrix
    """

    if basis == 'polynomial':

        phi = np.ones((len(x), 1))

        for i in range(1, poly_degree + 1):
            phi_power_i = np.power(x, i)
            phi = np.concatenate((phi, phi_power_i), axis=1)

        if (include_bias == False):
            phi = phi[:, 1:]

        return phi

    elif basis == 'sigmoid':

        phi = np.ones((len(x), 1))

        for i in range(len(miu)):
            phi_sig_i = 1 / (1 + np.exp(-(x - miu[i]) / s))
            phi = np.concatenate((phi, phi_sig_i), axis=1)

        if (include_bias == True):
            return phi
        else:
            return  phi[:,1:]

    else: 
        assert(False), 'Unknown basis %s' % basis


def evaluate_regression(x, t, w, basis, degree=0, mu=[], s=0, include_bias=True):
    """Evaluate linear regression on a dataset.

    Args:
      x is test set input
      t is target value for test set
      w is coefficients
      degree is polynomial degree

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """
    phi = design_matrix(x, basis, poly_degree=degree, miu=mu, s=s, include_bias=include_bias)

    t_est = phi * w
    test_err_e = np.square(t_est - t) / 2
    test_err_rms =  np.sqrt(2 * np.sum(test_err_e) / len(x))

    return (t_est, test_err_rms)
