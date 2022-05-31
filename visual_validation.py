# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Using the validation methods described by J. Quinn (2017):
https://waterprogramming.wordpress.com/2017/08/29/open-source-streamflow-generator-part-ii-validation/

Reference:
    Kirsch, B. R., G. W. Characklis, and H. B. Zeff (2013),
    Evaluating the impact of alternative hydro-climate scenarios on transfer agreements:
    A practical improvement for generating synthetic streamflows,
    J. Water Resour. Plann. Manage., 139(4), 396–406.

"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker


# Load other functions of interest
from my_stats_functions import standardize
from mFGN_generator import FGN_generate

###########################################

def plot_FDC_range(historical, synthetic, filename = 'FlowDurationCurve.jpeg'):
    """
    Plots overlapping flow duration curves.

    Parameters
    ----------
    historical : matrix [N-years x T-timesteps]
        The historic streamflow data.
    synthetic : matrix [M-years x T-timesteps]
        The synthetic streamflow data.
    filename : str
        The name of the output file, including '.jpeg' or '.png'.

    Returns
    -------
    None.

    """

    # constants of interest
    years_hist = np.shape(historical)[0]
    years_syn = np.shape(synthetic)[0]

    # initialize
    FDC_hist = np.empty(np.shape(historical))
    FDC_syn = np.empty(np.shape(synthetic))
    FDC_hist[:] = np.NaN
    FDC_syn[:] = np.NaN

    # rank flows greatest -> lowest
    for yr in range(years_hist):
        FDC_hist[yr, :] = np.sort(historical[yr, :])[::-1]

    for yr in range(years_syn):
        FDC_syn[yr,:] = np.sort(synthetic[yr, :])[::-1]

    # calculate generic exceedance vector
    n = np.shape(historical)[1]
    M = np.array(range(1,n+1))
    P = (M - 0.5) / n



    fig,ax = plt.subplots()
    fig.set_size_inches(5,5)

    ax.semilogy(P, np.min(FDC_syn, 0), c='k', label='Synthetic')
    ax.semilogy(P, np.max(FDC_syn, 0), c='k', label='Synthetic')
    ax.semilogy(P, np.min(FDC_hist, 0), c='#bdbdbd', label='Historical')
    ax.semilogy(P, np.max(FDC_hist, 0), c='#bdbdbd', label='Historical')

    ax.fill_between(P, np.min(FDC_syn, 0), np.max(FDC_syn, 0), color='k')
    ax.fill_between(P, np.min(FDC_hist, 0), np.max(FDC_hist, 0), color='#bdbdbd')

    ax.set_ylabel('Flow (cfs)', fontsize = 14)
    ax.set_xlabel("Exceedance Probability", fontsize = 14)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.grid(True,which='both',ls='-')

    ax.tick_params(axis = 'x', labelbottom = True, labelsize = 12,
                   gridOn= True, grid_linestyle = 'dashed')

    ax.tick_params(axis = 'y', gridOn = True,
                   grid_linestyle = 'dashed')


    ax.set_xlim([0,1])
    #ax.set_yticks([10**-1,10**0,10**1,10**2])

    ax.grid(axis = 'both', which = 'both', linestyle = 'dashed')

    title_string = str('Flow duration curve for historic and synthetic streamflows  \n$n_{historic}$ = ' + str(years_hist) + '\n$n_{synthetic}$ = ' + str(years_syn))
    plt.title(title_string, loc = 'left')

    fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.1), borderaxespad=0, fontsize = 14, loc='lower center',ncol=2, frameon=True)

    fig.savefig(str('./figures/' + filename + '.jpeg'), format = 'jpeg', dpi = 200)

    return None


###########################################

def plot_correlation_map(correlation, output_name, historic_data = True, annual_data = True):
    """
    Plots a heatmap of correlation values.

    Parameters
    ----------
    correlation : matrix
        A [N x M] matrix of correlation values.
    output_name : str
        The name of the output file, excluding '.jpeg' or '.png'.
    historic_data : bool
        Default True.
    annual_data : bool
        If showing correlation across entire annual timeperiod (i.e., 52 weeks),
         then True, if irrigaton season data then False.  Default True.
    annual_standardize =


    Returns
    -------
    None.

    """
    # The number of non_irrigated weeks in spring (1), fall (2), and irrigated weeks in mid-year
    nw = 23 ; nn1 = 16 ; nn2= 13

    if annual_data:
        my_tick_labs = np.linspace(1, 52, int(52/2), dtype = np.int)
        my_ticks = np.arange(52)
        my_xlab = 'Weekly Inflow'
        my_ylab = 'Weekly Demand'

    elif not annual_data:
        correlation = correlation[nn1:nn1+nw, nn1:nn1+nw]

        my_ticks = np.arange(nw)
        my_tick_labs = np.linspace(nn1, nn1+nw, int((nn1+nw)/2), dtype = np.int)
        my_xlab = 'Irrigation Season Weekly Inflow'
        my_ylab = 'Irrigation Season Weekly Demand'

    if historic_data:
        title_string = "Historic Data: Correlation Matrix \n Data standardized across weeks"

    elif not historic_data:
        title_string = "Synthetic Data: Correlation Matrix \n Data standardized across weeks"
    else:
        print('Invalid specifications; check inputs.')
        return

    ## Heatmap of correlation
    plt.figure()
    s_hist = sns.heatmap(correlation, xticklabels = 5, yticklabels = 5, annot = False)


    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals

    s_hist.set(xlabel = my_xlab, ylabel = my_ylab )
    plt.title(title_string, fontsize = 14)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    plt.savefig(str(output_name + '.png'), dpi = 300)

    return None

###########################################

def plot_all_years(historic, synthetic, my_ylabel, filename, my_color = 'blue', highlight_mean = True, standardized = False):
    """
    Plots overlapping synthetic and historic demand ranges.

    Parameters
    ----------



    Returns
    -------
    None.

    """
    # Calculate column averages

    X = np.arange(historic.shape[1])
    min_historic = np.min(historic, axis = 0)
    max_historic = np.max(historic, axis = 0)
    min_synthetic = np.min(synthetic, axis = 0)
    max_synthetic = np.max(synthetic, axis = 0)

    # constants of interest
    years_hist = np.shape(historic)[0]
    years_syn = np.shape(synthetic)[0]

    # initialize
    fig,ax = plt.subplots()
    fig.set_size_inches(7,5)

    ax.fill_between(X, min_historic, max_historic, color = 'grey', label = 'Historic', alpha = 0.5)
    ax.fill_between(X, min_synthetic, max_synthetic, color = 'black', label = 'Synthetic', alpha = 0.5)


    ax.set_ylabel(my_ylabel, fontsize = 14)
    ax.set_xlabel("Week", fontsize = 14)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.grid(True,which='both',ls='-')

    ax.tick_params(axis = 'x', labelbottom = True, labelsize = 12,
                   gridOn= True, grid_linestyle = 'dashed')

    ax.tick_params(axis = 'y', gridOn = True,
                   grid_linestyle = 'dashed')


    ax.set_xlim([0,52])

    ax.grid(axis = 'both', which = 'both', linestyle = 'dashed')

    title_string = str('Comparison of historic and synthetic timeseries ranges.  \n$n_{historic}$ = ' + str(years_hist) + '\n$n_{synthetic}$ = ' + str(years_syn))
    plt.title(title_string, loc = 'left')

    fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.1), borderaxespad=0, fontsize = 14, loc='lower center',ncol=2, frameon=True)

    fig.savefig(str('./figures/'+ filename + '.png'), dpi = 200)

    return
