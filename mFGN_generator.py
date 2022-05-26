# -*- coding: utf-8 -*-
"""
Trevor Amestoy

Cornell University
Spring 2022

Purpose:
    Contains the modified Fractional Gaussian Noise (mFGN) generator that is
    used to produce synthetic streamflow timeseries.

    Follows the synthetic streamflow generation as described by:

    Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013). Evaluating the
    impact of alternative hydro-climate scenarios on transfer agreements:
    Practical improvement for generating synthetic streamflows. Journal of Water
    Resources Planning and Management, 139(4), 396-406.

    A detailed walk-trhough of the method is provided by Julie Quinn here:
    https://waterprogramming.wordpress.com/2017/08/29/open-source-streamflow-generator-part-i-synthetic-generation/
    
"""

# Core libraries
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import random

# Load custom functions of interest
from my_stats_functions import standardize
from my_stats_functions import transform_intraannual

# Functions for Cholsky decomposition
from scipy.linalg import cholesky, eig
from fix_nonpositive_matrix import _is_positive_semidefinite, fix_nonpositive_semidefinite

def FGN_generate(historic_flow, n_years, standardized = True):
    """
    Follows the synthetic streamflow generation as described by:

    Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013). Evaluating the
    impact of alternative hydro-climate scenarios on transfer agreements:
    Practical improvement for generating synthetic streamflows. Journal of Water
    Resources Planning and Management, 139(4), 396-406.

    A detailed walk-trhough of the method is provided by Julie Quinn here:
    https://waterprogramming.wordpress.com/2017/08/29/open-source-streamflow-generator-part-i-synthetic-generation/

    Parameters
    ----------
    historic_flow : matrix (N-years x T-timesteps)
        Historic flow data, arranged with yearly observations occupying the rows.
    n_years : int
        The number of synthetic years to produce.
    Standardized : bool
        Indication of whether the

    Returns
    -------
    if standardized:
        Matrices of 1,000 synthetic streamflow realizations ([1,0000, (N*T)]), Q_s, and standard streamflows, Z_s
    if not standardized:
        A matrix of 1,000 standard synthetic streamflow realizations ([1,0000, (N*T)]), Z_s

    """
    # dimensions
    n_hist = np.shape(historic_flow)[0]
    n_col = np.shape(historic_flow)[1]


    ## Initialize some matrices of interest ##
    Q_h = historic_flow                 # histric inflow matrix (years x weeks)

    M   = np.zeros((n_years + 1, n_col))          # intermediate matrix
    C   = np.zeros((n_years + 1, n_col))          # matrix of uncorrelated bootstrap timeseries

    Z_s = np.zeros((n_years, n_col))            # matrix of standard sythetic inflows
    Y_s = np.zeros((n_years + 1, n_col))            # matrix of synthetic log-flows
    Q_s = np.zeros((n_years, n_col))            # matrix of synthetic flows

    ## Whitening data ##
    if not standardized:
        # Log-transform
        Y_log = np.log(Q_h)

        # Standardize inflow data column(week)- wise
        Z_h, Y_log_mean, Y_log_sd = standardize(Y_log)
    elif standardized:
        Z_h = Q_h

    # modified
    Z_h_prime = transform_intraannual(Z_h)

    ## Bootstrapping ##
    # C is an uncorrelated randomly generated timeseries
    # M stores the bootstrap sampling for use in generating correlated flows at other

    for i in range(n_years + 1):
        for j in range(n_col):
            m = random.randrange(1,n_hist,1)
            M[i,j] = m
            C[i,j] = Z_h[m,j]

    C_prime = transform_intraannual(C)

    ## Impose correlation on X

    # correlation matrix of Z_h (hsitoric log-std inflows), using columns!
    P_h = np.corrcoef(Z_h, rowvar = False)
    P_h_prime = np.corrcoef(Z_h_prime, rowvar = False)

    # Cholesky decomposition

    # Check if the corrlation matrix is positive definite; fix if not.
    P_h = fix_nonpositive_semidefinite(P_h, fix_method = 'diag')
    P_h_prime = fix_nonpositive_semidefinite(P_h_prime, fix_method = 'diag')

    # Take upper diagonal decomposition
    U = cholesky(P_h, lower = False)
    U_prime = cholesky(P_h_prime, lower = False)

    # Impose correlation on bootstrapped timeseries
    Z = C @ U
    Z_prime = C_prime @ U_prime

    Z_s[:, 0:26] = Z_prime[:, 26:52]
    Z_s[:, 26:52] = Z[1:, 26:52]

    if not standardized:
        # Re-apply historical statistics
        for col in range(n_col):
            Q_s[:,col] = np.exp(Y_log_mean[col] + Z_s[:, col] * Y_log_sd[col])
        return Q_s, Z_s
    elif standardized:
        return Z_s
