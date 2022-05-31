# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Contains several basic functions that are used in the generation of synthetic
inflow and demand timeseries.

"""
import pandas as pd
import numpy as np
import statistics


def standardize(data, columns = True):
    """
    Performs a row- or column-wise standardization of the data.

    Parameters
    ----------
    data : matrix (2D or 1D array)
        A numpy-like matrix of values, or array.

    Returns
    -------
    std_data : matrix (same as input)
        The standardized data.
    data_mean : array
        The means of the columns, a scalar if an array is input.
    data_sd : array
        The standard deviation of the columns, a scalar if an array is input.

    """

    if len(data.shape) == 1 and columns == True:
        n_columns = 1

        std_data = np.zeros(len(data))

        data_sd = statistics.stdev(data)
        data_mean = statistics.mean(data)

        std_data[:] = (data - data_mean) / data_sd

    elif len(data.shape) == 2 and columns == True:
        n_columns = data.shape[1]
        data_sd = np.zeros(shape = (n_columns))
        data_mean = np.zeros(shape = (n_columns))
        std_data = np.zeros((len(data), n_columns))

        for i in range(n_columns):

            data_sd[i] = statistics.stdev(data[:,i])
            data_mean[i] = statistics.mean(data[:,i])

            std_data[:,i] = (data[:,i] - data_mean[i]) / data_sd[i]

    elif len(data.shape) == 2 and columns == False:
        n_rows = data.shape[0]
        n_columns = data.shape[1]

        data_sd = np.zeros(shape = (n_rows))
        data_mean = np.zeros(shape = (n_rows))
        std_data = np.zeros((len(data), n_columns))

        for i in range(n_rows):

            data_sd[i] = statistics.stdev(data[i,:])
            data_mean[i] = statistics.mean(data[i,:])

            std_data[i,:] = (data[i,:] - data_mean[i]) / data_sd[i]

    return std_data, data_mean, data_sd


#################################################################################

def transform_intraannual(Y):
    """
    Transforms matrix Y into Y' according to methods described in Kirsch et al. (2013)

    Parameters
    ----------
    Y : matrix (n_year x 52 weeks)
        Matrix to be transformed.

    Returns
    -------
    Y_prime : matrix (n_year - 1 x 52 weeks)

    """

    # dimensions
    N = np.shape(Y)[0]
    n_col = np.shape(Y)[1]

    # Reorganize Y into Y_prime to preserve interannual correlation
    Y_prime = np.zeros((N-1, n_col))
    flat_Y = Y.flatten()

    for k in range(1,int(len(flat_Y)/52)):

        Y_prime[(k-1),0:26] = flat_Y[(k * 26) : ((k + 1)*26)]
        Y_prime[(k-1),26:52] = flat_Y[((k+1) * 26) : ((k + 2)*26)]

    return Y_prime

###############################################################################
